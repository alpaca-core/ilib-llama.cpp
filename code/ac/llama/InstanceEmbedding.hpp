// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Sampler.hpp"
#include "Session.hpp"
#include <astl/mem_ext.hpp>

#include <vector>

struct llama_context;

namespace ac::llama {
class Model;
class Session;
class StringSession;
class ControlVector;

class AC_LLAMA_EXPORT InstanceEmbedding {
public:
    struct InitParams {
        uint32_t ctxSize = 0; // context size for the model (0 = maximum allowed by model)
        uint32_t batchSize = 2048; // logical batch size for prompt processing (may be silently truncated to ctxSize)
        uint32_t ubatchSize = 512; // physical batch size for prompt processing (0 = batchSize)
        bool flashAttn = false; // enable flash attention
    };

    explicit InstanceEmbedding(Model& model, InitParams params);
    ~InstanceEmbedding();

    std::vector<float> getEmbeddingVector(std::span<const Token> prompt);

    const Model& model() const noexcept { return m_model; }
    Sampler& sampler() noexcept { return m_sampler; }

private:
    Model& m_model;
    Sampler m_sampler;
    astl::c_unique_ptr<llama_context> m_lctx;
    std::optional<Session> m_session;
};

} // namespace ac::llama
