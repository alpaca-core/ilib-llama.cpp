// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <astl/mem_ext.hpp>
#include <string>

struct llama_adapter_lora;

namespace ac::llama {
class Model;
struct LLamaLoraResource;

class AC_LLAMA_EXPORT LoraAdapter {
public:
    LoraAdapter(Model& model, std::string path);

    llama_adapter_lora* ladapter() const noexcept { return m_adapter.get(); }

    const Model& model() const noexcept { return m_model; }

private:
    Model& m_model;
    astl::c_unique_ptr<llama_adapter_lora> m_adapter;
};

} // namespace ac::llama
