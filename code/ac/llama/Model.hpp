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

struct LlamaModelResource;

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
    Model(local::ResourceLock<LlamaModelResource> model, Params params);
    ~Model();

    const Params& params() const noexcept { return m_params; }

    uint32_t trainCtxLength() const noexcept;
    bool shouldAddBosToken() const noexcept;
    bool hasEncoder() const noexcept;
    bool prefixInputsWithBos() const noexcept { return m_params.prefixInputsWithBos; }

    // fallback to "chatml" if the underlying model does not provide a chat template
    std::string getChatTemplateId() const;

    llama_model* lmodel() noexcept;
    const llama_model* lmodel() const noexcept;

    void addLora(local::ResourceLock<ac::llama::LoraResource> lora) noexcept { m_loras.push_back(lora); }
    // void removeLora(local::ResourceLock<ac::llama::LoraResource> lora) noexcept {
    //     m_loras.erase(std::remove(m_loras.begin(), m_loras.end(), lora), m_loras.end());
    // }
    std::span<local::ResourceLock<ac::llama::LoraResource>> loras() noexcept { return std::span<local::ResourceLock<ac::llama::LoraResource>>(m_loras); }

    const Vocab& vocab() const noexcept { return m_vocab; }
private:
    const Params m_params;
    // std::shared_ptr<llama_model> m_lmodel;
    local::ResourceLock<LlamaModelResource> m_model;
    std::vector<local::ResourceLock<ac::llama::LoraResource>> m_loras;

    Vocab m_vocab{*this};
};

struct LlamaModelResource : public local::Resource {
    LlamaModelResource(std::string gguf, Model::Params params, ModelLoadProgressCb pcb);

    astl::c_unique_ptr<llama_model> m_model;
};

using LlamaModelResourcePtr = std::shared_ptr<LlamaModelResource>;

} // namespace ac::llama
