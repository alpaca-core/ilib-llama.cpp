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
    Session(Instance& instance, InitParams params);

    void setInitialPrompt(std::span<const Token> prompt);

    void pushPrompt(std::span<const Token> prompt);
    Token getToken();
    std::vector<uint8_t> getState();
    bool setState(std::span<uint8_t> state);
private:
    enum class Source {
        InitialPrompt,
        InteractivePrompt,
        Generated
    };

    void doDecode(std::span<const Token> tokens, Source src);

    Instance& m_instance;
    InitParams m_params;
    unsigned maxTokens = 0;
    unsigned numKeep = 0;
    uint32_t gaIndex = 0; // number of grouped KV tokens (only used if params.gaFactor > 1)
    uint32_t numPast = 0; // number of tokens in the context (that's prompts + generated)
};

} // namespace ac::llama
