// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "LoraAdapter.hpp"
#include "Model.hpp"

#include <llama.h>
#include <stdexcept>

namespace ac::llama {

LoraAdapter::LoraAdapter(Model& model, std::string path, float scale)
    : m_adapter(llama_adapter_lora_init(model.lmodel(), path.c_str()), llama_adapter_lora_free)
    , m_scale(scale)
    , m_path(std::move(path))
{
    if (!m_adapter) {
        throw std::runtime_error("Failed to initialize LORA adapter from " + m_path);
    }
}

} // namespace ac::llama
