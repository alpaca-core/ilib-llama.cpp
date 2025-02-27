// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ResourceCache.hpp"

#include <llama.h>

namespace ac::llama {

local::ResourceLock<ModelResource> ResourceCache::getOrCreateModel(std::string_view gguf, Model::Params params, ModelLoadProgressCb pcb) {
    local::ResourceLock<ModelResource> modelResult;
    {
        std::lock_guard l(m_modelsMutex);
        modelResult = m_modelsManager.findResource<ModelResource>(ModelKey{gguf.data(), params});

        if (modelResult) {
            return modelResult;
        }
    }

    ModelResoucePtr modelPtr = std::make_shared<ModelResource>(gguf.data(), params, std::move(pcb));

    {
        std::lock_guard l(m_modelsMutex);
        auto check = m_modelsManager.findResource<ModelResource>(ModelKey{gguf.data(), params});
        if (!check) {
            modelResult = m_modelsManager.addResource(ModelKey{gguf.data(), params}, std::move(modelPtr));
        }
    }

    return modelResult;
}

local::ResourceLock<LoraResource> ResourceCache::getOrCreateLora(Model& model, std::string_view loraPath) {
    local::ResourceLock<LoraResource> loraResult;
    {
        std::lock_guard l(m_lorasMutex);
        loraResult = m_lorasManager.findResource<LoraResource>(LoraKey{model.lmodel(), loraPath.data()});

        if (loraResult) {
            return loraResult;
        }
    }

    LoraResoucePtr loraPtr = std::make_shared<LoraResource>(model, loraPath.data());

    {
        std::lock_guard l(m_lorasMutex);
        auto check = m_lorasManager.findResource<LoraResource>(LoraKey{model.lmodel(), loraPath.data()});
        if (!check) {
            loraResult = m_lorasManager.addResource(LoraKey{model.lmodel(), loraPath.data()}, std::move(loraPtr));
        }
    }

    return loraResult;
}

}
