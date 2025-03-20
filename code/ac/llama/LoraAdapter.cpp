// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "LoraAdapter.hpp"
#include "Model.hpp"

#include <llama.h>
#include <stdexcept>

namespace ac::llama {

LoraAdapter::LoraAdapter(Model& model, std::string path)
    : m_model(model)
    , m_adapter(llama_adapter_lora_init(model.lmodel(), path.c_str()), llama_adapter_lora_free)
{
    if (!m_adapter) {
        throw std::runtime_error("Failed to create lora adapter");
    }
}

} // namespace ac::llama
