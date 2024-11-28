// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "LoraAdapter.hpp"
#include "Model.hpp"

#include <llama.h>

namespace ac::llama {

LoraAdapter::LoraAdapter(Model& model, std::string path, float scale)
    : m_adapter(llama_lora_adapter_init(model.lmodel(), path.c_str()), llama_lora_adapter_free)
    , m_scale(scale)
    , m_path(std::move(path))
{
    if (!m_adapter) {
        throw std::runtime_error("Failed to initialize LORA adapter from " + m_path);
    }
}

} // namespace ac::llama
