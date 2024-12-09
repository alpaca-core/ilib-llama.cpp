// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Session.hpp"
#include "Model.hpp"
#include "Instance.hpp"
#include "Logging.hpp"

#include <llama.h>

#include <astl/throw_stdex.hpp>

namespace ac::llama {
namespace {
llama_batch makeInputBatch(std::span<const Token> tokens) {
    // well, llama.cpp does not touch the tokens for input batches, but llama_batch needs them to be non-const
    // (mostly for stupid C reasons)
    // so... we have to do something evil here
    auto nonConstTokens = const_cast<Token*>(tokens.data());
    return llama_batch_get_one(nonConstTokens, int32_t(tokens.size()));
}
}

Session::Session(Instance& instance, InitParams params)
    : m_instance(instance)
    , m_params(std::move(params))
{
    auto lctx = m_instance.ctx();
    auto& sampler = m_instance.sampler();

    llama_kv_cache_clear(lctx);
    llama_synchronize(lctx);
    llama_perf_context_reset(lctx);
    sampler.reset();
    sampler.perfReset();

    const auto ctxLen = llama_n_ctx(lctx);
    maxTokens = ctxLen - 4; // (#16)
}

void Session::setInitialPrompt(std::span<const Token> initialPrompt) {
    Token initialToken; // used to reset the initial prompt to a single token

    auto lctx = m_instance.ctx();
    const auto ctxLen = llama_n_ctx(lctx);
    const auto tokenBos = llama_token_bos(m_instance.model().lmodel());
    numKeep = std::min(uint32_t(initialPrompt.size()), maxTokens); // number of tokens to keep in the context in case we overflow

    if (initialPrompt.empty()) {
        initialToken = tokenBos;
        initialPrompt = {&initialToken, 1};
    }

    if (initialPrompt.empty()) {
        throw_ex{} << "Empty initial prompt";
    }

    if (initialPrompt.size() > maxTokens) {
        throw_ex{} << "Initial prompt too long. Got " << initialPrompt.size() << " tokens, max: " << ctxLen - 4;
    }

    if (m_params.gaFactor != 1) {
        const uint32_t gaFactor = m_params.gaFactor;
        const uint32_t gaWidth = m_params.gaWidth;
        if (gaWidth % gaFactor != 0) {
            throw_ex{} << "Group-attention width " << gaWidth << " must be a multiple of group-attention factor " << gaFactor;
        }
        LLAMA_LOG(Info, "self-extend: train = ", m_instance.model().trainCtxLength(), ", gaFactor = ", gaFactor, ", gaWidth = ", gaWidth);
    }

    if (m_instance.model().hasEncoder()) {
        auto batch = makeInputBatch(initialPrompt);
        auto res = llama_encode(lctx, batch);
        if (res != 0) {
            throw_ex{} << "Failed to encode input";
        }
        auto& vocab = m_instance.model().vocab();
        initialToken = vocab.decoderStartToken();
        initialPrompt = {&initialToken, 1};
    }

    doDecode(initialPrompt, Source::InitialPrompt);
}

void Session::pushPrompt(std::span<const Token> prompt) {
    if (!prompt.empty()) {
        auto& sampler = m_instance.sampler();
        auto& model = m_instance.model();

        // reset sampling and don't allow previous inputs to affect the generation
        sampler.reset();

        if (model.prefixInputsWithBos()) {
            const auto tokenBos = llama_token_bos(m_instance.model().lmodel());
            // add bos token to the prompt
            doDecode({&tokenBos, 1}, Source::InteractivePrompt);
        }

        doDecode(prompt, Source::InteractivePrompt);
    }
}

Token Session::getToken() {
    auto& sampler = m_instance.sampler();
    auto& vocab = m_instance.model().vocab();

    auto token = sampler.sample(m_instance.ctx());

    if (vocab.isEog(token)) {
        return Token_Invalid;
        // don't decode eog tokens in case the the interaction is continued
    }

    // old-comment
    // first yield, then decode, thus we don't decode if the session is aborted
    doDecode({&token, 1}, Source::Generated);
    return token;
}

std::vector<uint8_t> Session::getState() {
    const auto size = llama_state_get_size(m_instance.ctx());
    std::vector<uint8_t> state(size);
    if (llama_state_get_data(m_instance.ctx(), state.data(), size) != size) {
        throw_ex{} << "Failed to get state";
    }
    return state;
}

bool Session::setState(std::span<uint8_t> state) {
    if (llama_state_set_data(m_instance.ctx(), state.data(), state.size()) != state.size()) {
        throw_ex{} << "Failed to set state";
    }
    return true;
}

void Session::doDecode(std::span<const Token> tokens, Source src) {
    // first try to expand the context if needed
    const auto gaFactor = m_params.gaFactor;
    auto lctx = m_instance.ctx();
    const auto ctxLen = llama_n_ctx(lctx);
    auto& sampler = m_instance.sampler();

    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if (tokens.size() > maxTokens) {
        const auto skipped = tokens.size() - maxTokens;
        tokens = tokens.first(maxTokens);
        LLAMA_LOG(Warning, "Input too long. Skipping ", skipped, " tokens");
    }

    bool haveFullContextMitigation = false;
    if (gaFactor == 1) {
        // infinite text generation via context shifting
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via numPast)
        // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
        const auto num = numPast + tokens.size();
        if (num >= ctxLen) {
            if (!m_params.infiniteContext) {
                throw_ex{} << "context limit of " << ctxLen << " reached";
            }

            const auto numLeft = numPast - numKeep;
            const int numDiscard = numLeft / 2; // somewhat arbitrary

            LLAMA_LOG(Debug, "Context is full. Swapping: past = ", numPast, ", numLeft: ", numLeft,
                ", ctxLen: ", ctxLen, ", numKeep: ", numKeep, ", numDiscard: ", numDiscard);

            llama_kv_cache_seq_rm(lctx, 0, numKeep, numKeep + numDiscard);
            llama_kv_cache_seq_add(lctx, 0, numKeep + numDiscard, numPast, -numDiscard);

            numPast -= numDiscard;
            haveFullContextMitigation = true;
        }
    }
    else {
        const uint32_t gaWidth = m_params.gaWidth;

        while (numPast >= gaIndex + gaWidth) {
            // context extension via Self-Extend
            const int ib = (gaFactor * gaIndex) / gaWidth;
            const int bd = (gaWidth / gaFactor) * (gaFactor - 1);
            const int dd = (gaWidth / gaFactor) - ib * bd - gaWidth;

            LLAMA_LOG(Debug, "Group attention shift: ib = ", ib, ", bd = ", bd, ", dd = ", dd);

            llama_kv_cache_seq_add(lctx, 0, gaIndex, numPast, ib * bd);
            llama_kv_cache_seq_div(lctx, 0, gaIndex + ib * bd, gaIndex + ib * bd + gaWidth, gaFactor);
            llama_kv_cache_seq_add(lctx, 0, gaIndex + ib * bd + gaWidth, numPast + ib * bd, dd);

            numPast -= bd;

            gaIndex += gaWidth / gaFactor;
            haveFullContextMitigation = true;
        }
    }

    if (haveFullContextMitigation) {
        LLAMA_LOG(Info, "Context full mitigation performed: past = ", numPast, ", tokens = ", tokens.size());
    }

    // add to sampler
    for (auto t : tokens) {
        // only apply grammar for generated content
        sampler.accept(t, src == Source::Generated);
    }

    // decode
    const auto batchSize = llama_n_batch(lctx);

    // decode with batches of batchSize
    while (!tokens.empty()) {
        auto batchTokens = tokens.size() > batchSize ? tokens.first(batchSize) : tokens;
        tokens = tokens.subspan(batchTokens.size());
        auto batch = makeInputBatch(batchTokens);
        if (llama_decode(lctx, batch) != 0) {
            throw_ex{} << "Failed to decode tokens";
        }
        numPast += uint32_t(batchTokens.size());
    }

}

} // namespace ac::llama
