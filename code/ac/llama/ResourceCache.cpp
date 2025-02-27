// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ResourceCache.hpp"

#include <llama.h>

namespace ac::llama {

local::ResourceLock<LlamaModelResource> ResourceCache::getOrCreateModel(std::string_view gguf, Model::Params params, ModelLoadProgressCb pcb) {
    return m_modelsManager.findOrCreateResource<LlamaModelResource>(ModelKey{gguf.data(), params}, [&] {
        return std::make_shared<LlamaModelResource>(gguf.data(), params, std::move(pcb));
    });
}

local::ResourceLock<LLamaLoraResource> ResourceCache::getOrCreateLora(Model& model, std::string_view loraPath) {
    return m_lorasManager.findOrCreateResource<LLamaLoraResource>(LoraKey{model.lmodel(), loraPath.data()}, [&] {
        return std::make_shared<LLamaLoraResource>(model, loraPath.data());
    });
}

}
