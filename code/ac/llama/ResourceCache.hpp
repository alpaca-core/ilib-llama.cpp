// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Model.hpp"
#include "LoraAdapter.hpp"

#include <ac/local/ResourceManager.hpp>

namespace ac::llama {

struct ModelKey {
    std::string gguf;
    Model::Params params;

    bool operator==(const ModelKey& other) const noexcept = default;
};

struct LoraKey {
    llama_model* model;
    std::string path;

    bool operator==(const LoraKey& other) const noexcept = default;
};

class AC_LLAMA_EXPORT ResourceCache {
public:
    local::ResourceLock<LlamaModelResource> getOrCreateModel(std::string_view gguf, Model::Params params, ModelLoadProgressCb pcb);
    local::ResourceLock<LLamaLoraResource> getOrCreateLora(Model& model, std::string_view loraPath);

private:
    local::ResourceManager<ModelKey> m_modelsManager;
    local::ResourceManager<LoraKey> m_lorasManager;
};

}
