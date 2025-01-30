// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Sampler.hpp"
#include "Session.hpp"
#include <astl/mem_ext.hpp>
#include <optional>

struct llama_context;

namespace ac::llama {
class Model;
class Session;
class StringSession;
class ControlVector;

class AC_LLAMA_EXPORT Instance {
public:
    struct InitParams {
        uint32_t ctxSize = 0; // context size for the model (0 = maximum allowed by model)
        uint32_t batchSize = 2048; // logical batch size for prompt processing (may be silently truncated to ctxSize)
        uint32_t ubatchSize = 512; // physical batch size for prompt processing (0 = batchSize)
        bool flashAttn = false; // enable flash attention
        std::string grammar; // BNF-styled grammar
    };

    explicit Instance(Model& model, InitParams params);
    ~Instance();

    // add control to the context
    void addControlVector(const ControlVector& ctrlVector);

    // do an empty model run to load model data in cache
    void warmup();

    // only one session per instance can be active at a time
    Session& startSession(const Session::InitParams params);
    void stopSession() noexcept;

    const Model& model() const noexcept { return m_model; }
    Sampler& sampler() noexcept { return m_sampler; }

private:
    Model& m_model;
    Sampler m_sampler;
    astl::c_unique_ptr<llama_context> m_lctx;
    std::optional<Session> m_session;
};

} // namespace ac::llama
