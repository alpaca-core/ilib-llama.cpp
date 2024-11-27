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
class LoraAdapter;

using ModelLoadProgressCb = astl::ufunction<void(float)>;

class AC_LLAMA_EXPORT Model {
public:
    struct Params {
        bool gpu = true; // try to load data on gpu
        bool vocabOnly = false; // do not load model, only vocab
        bool prefixInputsWithBos = false; // add bos token to interactive inputs (#13)
    };

    // Model(const char* pathToGguf, ModelLoadProgressCb loadProgressCb, Params params);
    Model(std::shared_ptr<llama_model> lmodel, Params params);
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

    void addLora(std::shared_ptr<LoraAdapter> lora) noexcept { m_loras.push_back(lora); }
    void removeLora(std::shared_ptr<LoraAdapter> lora) noexcept {
        m_loras.erase(std::remove(m_loras.begin(), m_loras.end(), lora), m_loras.end());
    }
    std::span<std::shared_ptr<LoraAdapter>> loras() noexcept { return std::span<std::shared_ptr<LoraAdapter>>(m_loras); }

    const Vocab& vocab() const noexcept { return m_vocab; }
private:
    const Params m_params;
    std::shared_ptr<llama_model> m_lmodel;
    std::vector<std::shared_ptr<LoraAdapter>> m_loras;

    Vocab m_vocab{*this};
};

class AC_LLAMA_EXPORT LoraAdapter {
public:
    LoraAdapter(Model& model, std::string path, float scale = 1.0f);

    llama_lora_adapter* adapter() const noexcept { return m_adapter.get(); }
    float scale() const noexcept { return m_scale; }
    std::string_view path() const noexcept { return m_path; }

private:
    astl::c_unique_ptr<llama_lora_adapter> m_adapter;
    float m_scale;
    std::string m_path;
};

class AC_LLAMA_EXPORT ModelRegistry {
public:
    Model loadModel(
        const std::string& gguf,
        std::span<std::string> loras,
        ModelLoadProgressCb pcb,
         Model::Params params);

    std::shared_ptr<LoraAdapter> loadLora(Model* model, const std::string& loraPath);

private:
    std::unordered_map<std::string, std::weak_ptr<llama_model>> m_models;
    std::unordered_map<llama_model*, std::vector<std::weak_ptr<LoraAdapter>>> m_loras;
};

} // namespace ac::llama
