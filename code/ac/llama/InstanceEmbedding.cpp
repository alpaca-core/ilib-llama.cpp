// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "InstanceEmbedding.hpp"
#include "Model.hpp"
#include "LoraAdapter.hpp"
#include "Logging.hpp"
#include "Session.hpp"
#include "ControlVector.hpp"

#include <llama.h>

#include <astl/throw_stdex.hpp>
#include <astl/iile.h>
#include <astl/move.hpp>
#include <astl/sentry.hpp>

#include <cassert>
#include <span>
#include <fstream>

namespace ac::llama {

namespace {
llama_context_params llamaFromInstanceInitParams(const InstanceEmbedding::InitParams& params) {
    llama_context_params llamaParams = llama_context_default_params();
    llamaParams.n_ctx = params.ctxSize;
    llamaParams.n_batch = params.batchSize;
    llamaParams.n_ubatch = params.ubatchSize;
    llamaParams.flash_attn = params.flashAttn;
    return llamaParams;
}
} // namespace

InstanceEmbedding::InstanceEmbedding(Model& model, InitParams params)
    : m_model(model)
    , m_sampler(model, {})
    , m_lctx(llama_new_context_with_model(model.lmodel(), llamaFromInstanceInitParams(params)), llama_free)
{
    if (!m_lctx) {
        throw_ex{} << "Failed to create llama context";
    }
    assert(model.lmodel() == llama_get_model(m_lctx.get()));

    const auto ctxLen = llama_n_ctx(m_lctx.get());
    const auto ctxTrain = model.trainCtxLength();
    if (ctxLen > ctxTrain) {
        LLAMA_LOG(Warning, "Instance requested context length ", ctxLen, " is greater than the model's training context length ", ctxTrain);
    }

    if (llama_model_has_encoder(m_model.lmodel()) && llama_model_has_decoder(m_model.lmodel())) {
        LLAMA_LOG(Error, "Computing embeddings in encoder-decoder models is not supported");
    }

    // for (auto& lora: model.loras())
    // {
    //     if (llama_lora_adapter_set(m_lctx.get(), lora->adapter(), lora->scale()) < 0) {
    //         LLAMA_LOG(Error, "Failed to set LORA adapter from ", lora->path());
    //     }
    // }
}

InstanceEmbedding::~InstanceEmbedding() = default;

namespace {
llama_batch makeInputBatch(std::span<const Token> tokens) {
    // well, llama.cpp does not touch the tokens for input batches, but llama_batch needs them to be non-const
    // (mostly for stupid C reasons)
    // so... we have to do something evil here
    auto nonConstTokens = const_cast<Token*>(tokens.data());
    return llama_batch_get_one(nonConstTokens, int32_t(tokens.size()));
}

void normalizeEmbedding(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}
}

std::vector<float> InstanceEmbedding::getEmbeddingVector(std::span<const Token> prompt) {
        // count number of embeddings
    int n_embd_count = 0;
    // if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
    //     for (int k = 0; k < n_prompts; k++) {
    //         n_embd_count += inputs[k].size();
    //     }
    // } else {
        n_embd_count = 1;//n_prompts;
    // }

        // allocate output
    const int n_embd = llama_n_embd(m_model.lmodel());
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float* emb = embeddings.data();

    int e = 0;
    // final batch
    float * out = emb + e * n_embd;
    llama_batch batch = makeInputBatch(prompt);

    //batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
    const enum llama_pooling_type pooling_type = llama_pooling_type(m_lctx.get());
    llama_context* ctx = m_lctx.get();
    llama_model* model = m_model.lmodel();

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    // LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (llama_encode(ctx, batch) < 0) {
            // LOG_ERR("%s : failed to encode\n", __func__);
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            // LOG_ERR("%s : failed to decode\n", __func__);
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            // GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            // GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * outRes = out + embd_pos * n_embd;
        // TODO: add normalization option
        int embd_norm = 0; //params.embd_normalize;
        normalizeEmbedding(embd, outRes, n_embd, embd_norm);
    }

    return embeddings;
}

} // namespace ac::llama
