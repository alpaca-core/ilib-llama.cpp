// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <ac/local/Resource.hpp>
#include <ac/local/ResourceLock.hpp>

#include <astl/mem_ext.hpp>
#include <string>

struct llama_adapter_lora;

namespace ac::llama {
class Model;
struct LLamaLoraResource;

class AC_LLAMA_EXPORT LoraAdapter {
public:
    LoraAdapter(local::ResourceLock<LLamaLoraResource> resource, float scale = 1.0);

    llama_adapter_lora* adapter() const noexcept;
    float scale() const noexcept { return m_scale; }

private:
    local::ResourceLock<LLamaLoraResource> m_resource;
    float m_scale;
};

struct LLamaLoraResource : public local::Resource {
    LLamaLoraResource(Model& model, std::string path);

    astl::c_unique_ptr<llama_adapter_lora> m_adapter;
};

using LoraResourcePtr = std::shared_ptr<LLamaLoraResource>;

} // namespace ac::llama
