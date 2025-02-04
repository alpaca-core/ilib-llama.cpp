// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Token.hpp"
#include <span>
#include <utility>
#include <exception>
#include <coroutine>
#include <vector>
#include <cassert>

struct llama_context;

namespace ac::llama {
class Instance;

class Session {
public:
    struct InitParams {
        uint32_t gaFactor = 1; // group-attention factor
        uint32_t gaWidth = 512; // group-attention width

        // if true, the inference tries to extend the context by truncating previous tokens
        // only used if gaFactor == 1
        bool infiniteContext = true;
    };
    Session(Instance& instance, llama_context* ctx, InitParams params);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() = default;

    // initial functions to prepare the session
    void setInitialPrompt(std::span<const Token> prompt);
    bool setState(std::span<uint8_t> state);

    // main functions to interact with the model
    void pushPrompt(std::span<const Token> prompt);
    Token getToken();
    std::vector<float> getLogits(int32_t topK, float topP = 0.95f);
    std::vector<uint8_t> getState();
private:
    enum class Source {
        InitialPrompt,
        InteractivePrompt,
        Generated
    };

    void doDecode(std::span<const Token> tokens, Source src);
    void flushPendingState();

    struct State {
        enum class Phase {
            Initial,
            Generating
        };

        Phase m_phase = Phase::Initial;
        Token m_currToken = Token_Invalid;

        unsigned maxTokens = 0;
        unsigned numKeep = 0;
        uint32_t gaIndex = 0; // number of grouped KV tokens (only used if params.gaFactor > 1)
        uint32_t numPast = 0; // number of tokens in the context (that's prompts + generated)
    };

    Instance& m_instance;
    llama_context* m_ctx;
    InitParams m_params;
    State m_state;
};

} // namespace ac::llama
