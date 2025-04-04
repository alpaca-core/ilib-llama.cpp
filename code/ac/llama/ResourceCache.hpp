// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Model.hpp"
#include "LoraAdapter.hpp"

#include <ac/local/ResourceCache.hpp>

namespace ac::llama {

class ResourceCache {
public:
    ResourceCache(local::ResourceManager& rm)
        : m_modelCache(rm)
    {}

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
        ModelResource(local::ResourceManager& rm, const ModelParams& params, ModelLoadProgressCb pcb)
            : Model(params.gguf, params.params, std::move(pcb))
            , m_loras(rm)
        {}

        LoraLock getLora(LoraParams params) {
            return m_loras.findOrCreate(std::move(params), [&](const LoraParams& key) {
                return std::make_shared<LoraResource>(*this, key.path);
            });
        }
    private:
        local::ResourceCache<LoraParams, LoraResource> m_loras;
    };

    using ModelLock = local::ResourceLock<ModelResource>;

    ModelLock getModel(ModelParams params, ModelLoadProgressCb pcb = {}) {
        return m_modelCache.findOrCreate(std::move(params), [&](const ModelParams& key) {
            return std::make_shared<ModelResource>(m_modelCache.manager(), key, std::move(pcb));
        });
    }

private:
    local::ResourceCache<ModelParams, ModelResource> m_modelCache;
};

}
