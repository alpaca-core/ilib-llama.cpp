// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Vocab.hpp"
#include <astl/mem_ext.hpp>
#include <astl/ufunction.hpp>
#include <algorithm>
#include <string>
#include <span>
#include <vector>

struct llama_model;

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

        bool operator==(const Params& other) const noexcept {
            return gpu == other.gpu
                    && vocabOnly == other.vocabOnly
                    && prefixInputsWithBos == other.prefixInputsWithBos;
        }
    };

    Model(std::shared_ptr<llama_model> model, Params params);
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

class AC_LLAMA_EXPORT ModelRegistry {
public:
    static ModelRegistry& getInstance() {
        static ModelRegistry instance;
        return instance;
    };

    std::shared_ptr<llama_model> loadModel(
        const std::string& gguf,
        ModelLoadProgressCb pcb,
        Model::Params params);

    std::shared_ptr<LoraAdapter> loadLora(Model* model, const std::string& loraPath);
private:

    struct ModelKey {
        std::string gguf;
        Model::Params params;

        bool operator==(const ModelKey& other) const noexcept {
            return gguf == other.gguf
                    && params == other.params;
        }
    };

    std::vector<std::pair<ModelKey, std::weak_ptr<llama_model>>> m_models;
    std::unordered_map<llama_model*, std::vector<std::weak_ptr<LoraAdapter>>> m_loras;
};

} // namespace ac::llama
