// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Model.hpp"
#include "LoraAdapter.hpp"

#include <ac/local/ResourceManager.hpp>

namespace ac::llama {

class ResourceCache {
public:
    struct LoraParams {
        std::string path;
        bool operator==(const LoraParams& other) const noexcept = default;
    };
    struct LoraResource : public LoraAdapter, public local::Resource {
        using LoraAdapter::LoraAdapter;
    };

    using LoraLock = local::ResourceLock<LoraResource>;

    struct ModelParams {
        std::string gguf;
        Model::Params params;
        bool operator==(const ModelParams& other) const noexcept = default;
    };

    class ModelResource : public Model, public local::Resource {
    public:
        ModelResource(const ModelParams& params, ModelLoadProgressCb pcb)
            : Model(params.gguf, params.params, std::move(pcb))
        {}

        LoraLock getLora(LoraParams params) {
            return m_loras.findOrCreate(std::move(params), [&](const LoraParams& key) {
                return std::make_shared<LoraResource>(*this, key.path);
            });
        }
    private:
        local::ResourceManager<LoraParams, LoraResource> m_loras;
    };

    using ModelLock = local::ResourceLock<ModelResource>;

    ModelLock getModel(ModelParams params, ModelLoadProgressCb pcb = {}) {
        return m_modelManager.findOrCreate(std::move(params), [&](const ModelParams& key) {
            return std::make_shared<ModelResource>(key, std::move(pcb));
        });
    }

private:
    local::ResourceManager<ModelParams, ModelResource> m_modelManager;

};

}
