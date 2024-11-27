// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Vocab.hpp"
#include <astl/mem_ext.hpp>
#include <astl/ufunction.hpp>
#include <string>
#include <span>
#include <vector>

struct llama_model;
struct llama_lora_adapter;

namespace ac::llama {
class Job;

using ModelLoadProgressCb = astl::ufunction<void(float)>;

class AC_LLAMA_EXPORT Model {
public:
    struct Params {
        bool gpu = true; // try to load data on gpu
        bool vocabOnly = false; // do not load model, only vocab
        bool prefixInputsWithBos = false; // add bos token to interactive inputs (#13)
    };

    Model(const char* pathToGguf, ModelLoadProgressCb loadProgressCb, Params params);
    ~Model();

    const Params& params() const noexcept { return m_params; }

    uint32_t trainCtxLength() const noexcept;
    bool shouldAddBosToken() const noexcept;
    bool hasEncoder() const noexcept;
    bool prefixInputsWithBos() const noexcept { return m_params.prefixInputsWithBos; }

    // fallback to "chatml" if the underlying model does not provide a chat template
    std::string getChatTemplateId() const;

    llama_model* lmodel() noexcept { return m_lmodel.get(); }
    const llama_model* lmodel() const noexcept { return m_lmodel.get(); }

    std::span<llama_lora_adapter*> loras() noexcept { return std::span<llama_lora_adapter*>(m_loras.data(), m_loras.size()); }

    const Vocab& vocab() const noexcept { return m_vocab; }
private:
    const Params m_params;
    astl::c_unique_ptr<llama_model> m_lmodel;
    std::vector<llama_lora_adapter*> m_loras;

    Vocab m_vocab{*this};
};

// Add model to the registry and return the model
// - add loras to the registry too, so if other models need them, they can be reused
// - pass the loras to the model, so when we create instance we can pass them to the instance
class AC_LLAMA_EXPORT ModelRegistry {
public:
    std::shared_ptr<Model> loadModel(
        const std::string& gguf,
        std::span<std::string> loras,
        ModelLoadProgressCb pcb,
         Model::Params params);
private:


    std::unordered_map<std::string, std::weak_ptr<Model>> m_models;
    std::unordered_map<std::string, llama_lora_adapter*> m_loras;
};

} // namespace ac::llama
