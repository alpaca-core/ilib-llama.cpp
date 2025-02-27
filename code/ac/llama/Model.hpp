// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Vocab.hpp"

#include "LoraAdapter.hpp"

#include <ac/local/Resource.hpp>
#include <ac/local/ResourceLock.hpp>

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

    Model(std::string gguf, Params params, ModelLoadProgressCb pcb);
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

    void addLora(local::ResourceLock<ac::llama::LoraResource> lora) noexcept { m_loras.push_back(lora); }
    // void removeLora(local::ResourceLock<ac::llama::LoraResource> lora) noexcept {
    //     m_loras.erase(std::remove(m_loras.begin(), m_loras.end(), lora), m_loras.end());
    // }
    std::span<local::ResourceLock<ac::llama::LoraResource>> loras() noexcept { return std::span<local::ResourceLock<ac::llama::LoraResource>>(m_loras); }
    // void addLora(std::shared_ptr<LoraAdapter> lora) noexcept { m_loras.push_back(lora); }
    // void removeLora(std::shared_ptr<LoraAdapter> lora) noexcept {
    //     m_loras.erase(std::remove(m_loras.begin(), m_loras.end(), lora), m_loras.end());
    // }
    // std::span<std::shared_ptr<LoraAdapter>> loras() noexcept { return std::span<std::shared_ptr<LoraAdapter>>(m_loras); }

    const Vocab& vocab() const noexcept { return m_vocab; }
private:
    const Params m_params;
    std::shared_ptr<llama_model> m_lmodel;
    std::vector<local::ResourceLock<ac::llama::LoraResource>> m_loras;

    Vocab m_vocab{*this};
};

struct ModelResource : public Model, public local::Resource {
    using Model::Model;
};

using ModelResoucePtr = std::shared_ptr<ModelResource>;

} // namespace ac::llama
