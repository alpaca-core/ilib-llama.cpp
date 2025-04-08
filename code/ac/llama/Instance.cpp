// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Instance.hpp"
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
llama_context_params llamaFromInstanceInitParams(const Instance::InitParams& params) {
    llama_context_params llamaParams = llama_context_default_params();
    llamaParams.n_ctx = params.ctxSize;
    llamaParams.n_batch = params.batchSize;
    llamaParams.n_ubatch = params.ubatchSize;
    llamaParams.flash_attn = params.flashAttn;
    return llamaParams;
}
} // namespace

Instance::Instance(Model& model, InitParams params)
    : m_model(model)
    , m_sampler(new Sampler(model, {
        .grammar = params.grammar,
    }))
    , m_lctx(llama_init_from_model(model.lmodel(), llamaFromInstanceInitParams(params)), llama_free)
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
}

Instance::~Instance() = default;

void Instance::addLora(LoraAdapter& lora, float scale) {
    if (lora.model().lmodel() != m_model.lmodel()) {
        throw_ex{} << "LoraAdapter model does not match the instance model";
    }
    llama_set_adapter_lora(m_lctx.get(), lora.ladapter(), scale);
}

void Instance::clearLoraState() {
    llama_clear_adapter_lora(m_lctx.get());
}

namespace {
llama_batch makeInputBatch(std::span<const Token> tokens) {
    // well, llama.cpp does not touch the tokens for input batches, but llama_batch needs them to be non-const
    // (mostly for stupid C reasons)
    // so... we have to do something evil here
    auto nonConstTokens = const_cast<Token*>(tokens.data());
    return llama_batch_get_one(nonConstTokens, int32_t(tokens.size()));
}
}

void Instance::addControlVector(const ControlVector& ctrlVector) {
    int err = llama_apply_adapter_cvec(m_lctx.get(),
            ctrlVector.data.data(),
            ctrlVector.data.size(),
            ctrlVector.nEmbd,
            ctrlVector.controlVectorLayerStart,
            ctrlVector.controlVectorLayerEnd);

    if (err) {
        throw_ex{} << "Failed to apply control vectors!";
    }
}

void Instance::warmup() {
    LLAMA_LOG(Info, "Running warmup");

    auto lctx = m_lctx.get();
    auto model = m_model.lmodel();

    std::vector<llama_token> tmp;
    llama_token bos = llama_vocab_bos(m_model.vocab().lvocab());
    llama_token eos = llama_vocab_eos(m_model.vocab().lvocab());
    // some models (e.g. T5) don't have a BOS token
    if (bos != LLAMA_TOKEN_NULL) {
        tmp.push_back(bos);
    }
    if (eos != LLAMA_TOKEN_NULL) {
        tmp.push_back(eos);
    }
    if (tmp.empty()) {
        tmp.push_back(0);
    }

    if (llama_model_has_encoder(model)) {
        llama_encode(lctx, makeInputBatch(tmp));
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = bos;
        }
        tmp.clear();
        tmp.push_back(decoder_start_token_id);
    }
    llama_decode(lctx, makeInputBatch(tmp));
    llama_kv_self_clear(lctx);
    llama_synchronize(lctx);
    llama_perf_context_reset(lctx);
}

Session& Instance::startSession(const Session::InitParams params) {
    if (m_session.has_value()) {
        throw_ex{} << "Session is already started. Stop it to start a new one.";
    }

    m_session.emplace(*this, m_lctx.get(), params);
    return *m_session;
}

void Instance::stopSession() noexcept {
    m_session.reset();
}

} // namespace ac::llama
