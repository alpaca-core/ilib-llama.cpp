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

class AC_LLAMA_EXPORT LoraAdapter {
public:
    LoraAdapter(Model& model, std::string path, float scale = 1.0f);

    llama_adapter_lora* adapter() const noexcept { return m_adapter.get(); }
    float scale() const noexcept { return m_scale; }
    const std::string& path() const noexcept { return m_path; }

private:
    astl::c_unique_ptr<llama_adapter_lora> m_adapter;
    float m_scale;
    std::string m_path;
};

} // namespace ac::llama
