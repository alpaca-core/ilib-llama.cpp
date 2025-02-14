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

Session::Session(Instance& instance, llama_context* ctx, InitParams params)
    : m_instance(instance)
    , m_ctx(ctx)
    , m_params(std::move(params))
{
    auto& sampler = m_instance.sampler();

    llama_kv_cache_clear(m_ctx);
    llama_synchronize(m_ctx);
    llama_perf_context_reset(m_ctx);
    sampler.reset();
    sampler.perfReset();

    const auto ctxLen = llama_n_ctx(m_ctx);
    m_state.maxTokens = ctxLen - 4; // (#16)
}

void Session::setInitialPrompt(std::span<const Token> initialPrompt) {
    if (m_state.m_phase != State::Phase::Initial) {
        throw_ex{} << "Session already started";
    }

    Token initialToken; // used to reset the initial prompt to a single token

    const auto ctxLen = llama_n_ctx(m_ctx);
    const auto tokenBos = llama_vocab_bos(m_instance.model().vocab().lvocab());
    m_state.numKeep = std::min(uint32_t(initialPrompt.size()), m_state.maxTokens); // number of tokens to keep in the context in case we overflow

    if (initialPrompt.empty()) {
        initialToken = tokenBos;
        initialPrompt = {&initialToken, 1};
    }

    if (initialPrompt.size() > m_state.maxTokens) {
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
        auto res = llama_encode(m_ctx, batch);
        if (res != 0) {
            throw_ex{} << "Failed to encode input";
        }
        auto& vocab = m_instance.model().vocab();
        initialToken = vocab.decoderStartToken();
        initialPrompt = {&initialToken, 1};
    }

    doDecode(initialPrompt, Source::InitialPrompt);
    m_state.m_phase = State::Phase::Generating;
}

void Session::pushPrompt(std::span<const Token> prompt) {
    if (m_state.m_phase != State::Phase::Generating) {
        throw_ex{} << "Session hasn't started yet";
    }

    flushPendingState();

    if (!prompt.empty()) {
        auto& sampler = m_instance.sampler();
        auto& model = m_instance.model();

        // reset sampling and don't allow previous inputs to affect the generation
        sampler.reset();

        if (model.prefixInputsWithBos()) {
            const auto tokenBos = llama_vocab_bos(model.vocab().lvocab());
            // add bos token to the prompt
            doDecode({&tokenBos, 1}, Source::InteractivePrompt);
        }

        doDecode(prompt, Source::InteractivePrompt);
    }
}

Token Session::getToken() {
    if (m_state.m_phase != State::Phase::Generating) {
        throw_ex{} << "Session hasn't started yet";
    }

    flushPendingState();

    auto& sampler = m_instance.sampler();
    auto& vocab = m_instance.model().vocab();

    m_state.m_currToken = sampler.sample(m_ctx);

    if (vocab.isEog(m_state.m_currToken)) {
        // don't decode eog tokens in case the the interaction is continued
        m_state.m_currToken = Token_Invalid;
    }

    return m_state.m_currToken;
}

std::vector<std::pair<Token, float>> Session::getProbs(int32_t topK, float topP) {
    if (m_state.m_phase != State::Phase::Generating) {
        throw_ex{} << "Session hasn't started yet";
    }

    flushPendingState();

    Sampler::Params sParams = {
        .topK = topK,
        .topP = topP,
        .samplerSequence = {
            Sampler::SamplingType::Top_K,
            Sampler::SamplingType::Top_P,
        }
    };
    Sampler sampler(const_cast<Model&>(m_instance.model()), sParams);

    auto logits = sampler.extractProbs(m_ctx);

    return logits;
}

std::vector<uint8_t> Session::getState() {
    if (m_state.m_phase != State::Phase::Generating) {
        throw_ex{} << "Session hasn't started yet";
    }

    flushPendingState();

    const auto size = llama_state_get_size(m_ctx);
    std::vector<uint8_t> state(size);
    if (llama_state_get_data(m_ctx, state.data(), size) != size) {
        throw_ex{} << "Failed to get state";
    }
    return state;
}

bool Session::setState(std::span<uint8_t> state) {
    if (m_state.m_phase != State::Phase::Initial) {
        throw_ex{} << "Session already started";
    }

    if (llama_state_set_data(m_ctx, state.data(), state.size()) != state.size()) {
        throw_ex{} << "Failed to set state";
    }

    m_state.m_phase = State::Phase::Generating;
    return true;
}

void Session::doDecode(std::span<const Token> tokens, Source src) {
    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if (tokens.size() > m_state.maxTokens) {
        const auto skipped = tokens.size() - m_state.maxTokens;
        tokens = tokens.first(m_state.maxTokens);
        LLAMA_LOG(Warning, "Input too long. Skipping ", skipped, " tokens");
    }

    bool haveFullContextMitigation = false;
    const auto gaFactor = m_params.gaFactor;
    const auto ctxLen = llama_n_ctx(m_ctx);
    auto& sampler = m_instance.sampler();

    if (gaFactor == 1) {
        // infinite text generation via context shifting
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via numPast)
        // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
        const auto num = m_state.numPast + tokens.size();
        if (num >= ctxLen) {
            if (!m_params.infiniteContext) {
                throw_ex{} << "context limit of " << ctxLen << " reached";
            }

            const auto numLeft = m_state.numPast - m_state.numKeep;
            const int numDiscard = numLeft / 2; // somewhat arbitrary

            LLAMA_LOG(Debug, "Context is full. Swapping: past = ", m_state.numPast, ", numLeft: ", numLeft,
                ", ctxLen: ", ctxLen, ", numKeep: ", m_state.numKeep, ", numDiscard: ", numDiscard);

            llama_kv_cache_seq_rm(m_ctx, 0, m_state.numKeep, m_state.numKeep + numDiscard);
            llama_kv_cache_seq_add(m_ctx, 0, m_state.numKeep + numDiscard, m_state.numPast, -numDiscard);

            m_state.numPast -= numDiscard;
            haveFullContextMitigation = true;
        }
    }
    else {
        const uint32_t gaWidth = m_params.gaWidth;

        while (m_state.numPast >= m_state.gaIndex + gaWidth) {
            // context extension via Self-Extend
            const int ib = (gaFactor * m_state.gaIndex) / gaWidth;
            const int bd = (gaWidth / gaFactor) * (gaFactor - 1);
            const int dd = (gaWidth / gaFactor) - ib * bd - gaWidth;

            LLAMA_LOG(Debug, "Group attention shift: ib = ", ib, ", bd = ", bd, ", dd = ", dd);

            llama_kv_cache_seq_add(m_ctx, 0, m_state.gaIndex, m_state.numPast, ib * bd);
            llama_kv_cache_seq_div(m_ctx, 0, m_state.gaIndex + ib * bd, m_state.gaIndex + ib * bd + gaWidth, gaFactor);
            llama_kv_cache_seq_add(m_ctx, 0, m_state.gaIndex + ib * bd + gaWidth, m_state.numPast + ib * bd, dd);

            m_state.numPast -= bd;

            m_state.gaIndex += gaWidth / gaFactor;
            haveFullContextMitigation = true;
        }
    }

    if (haveFullContextMitigation) {
        LLAMA_LOG(Info, "Context full mitigation performed: past = ", m_state.numPast, ", tokens = ", tokens.size());
    }

    // add to sampler
    for (auto t : tokens) {
        // only apply grammar for generated content
        sampler.accept(t, src == Source::Generated);
    }

    // decode
    const auto batchSize = llama_n_batch(m_ctx);

    // decode with batches of batchSize
    while (!tokens.empty()) {
        auto batchTokens = tokens.size() > batchSize ? tokens.first(batchSize) : tokens;
        tokens = tokens.subspan(batchTokens.size());
        auto batch = makeInputBatch(batchTokens);
        if (llama_decode(m_ctx, batch) != 0) {
            throw_ex{} << "Failed to decode tokens";
        }
        m_state.numPast += uint32_t(batchTokens.size());
    }

}

void Session::flushPendingState() {
    if (m_state.m_currToken != Token_Invalid) {
        // first yield, then decode, thus we don't decode if the session is aborted
        doDecode({&m_state.m_currToken, 1}, Source::Generated);
        m_state.m_currToken = Token_Invalid;
    }
}
} // namespace ac::llama
