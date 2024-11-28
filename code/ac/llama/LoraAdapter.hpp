// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include <astl/mem_ext.hpp>
#include <string>

struct llama_lora_adapter;

namespace ac::llama {
class Model;

class AC_LLAMA_EXPORT LoraAdapter {
public:
    LoraAdapter(Model& model, std::string path, float scale = 1.0f);

    llama_lora_adapter* adapter() const noexcept { return m_adapter.get(); }
    float scale() const noexcept { return m_scale; }
    std::string_view path() const noexcept { return m_path; }

private:
    astl::c_unique_ptr<llama_lora_adapter> m_adapter;
    float m_scale;
    std::string m_path;
};

} // namespace ac::llama
