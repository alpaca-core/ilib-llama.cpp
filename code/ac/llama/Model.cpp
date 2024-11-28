    // Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Model.hpp"
#include "LoraAdapter.hpp"
#include <llama.h>
#include <astl/move.hpp>
#include <stdexcept>

namespace ac::llama {
namespace {
llama_model_params llamaFromModelParams(const Model::Params& params, ModelLoadProgressCb& loadProgressCb)
{
    llama_model_params llamaParams = llama_model_default_params();
    if (params.gpu) {
        llamaParams.n_gpu_layers = 10000;
    }
    llamaParams.vocab_only = params.vocabOnly;
#ifndef NDEBUG
    llamaParams.check_tensors = true;
#endif

    if (loadProgressCb) {
        llamaParams.progress_callback = +[](float progress, void* userData) {
            auto progressCallback = reinterpret_cast<ModelLoadProgressCb*>(userData);
            (*progressCallback)(progress);
            return true;
        };
        llamaParams.progress_callback_user_data = &loadProgressCb;
    }

    return llamaParams;
}
} // namespace


Model::Model(const char* pathToGguf, std::vector<std::string> loras, ModelLoadProgressCb loadProgressCb, Params params)
    : m_params(astl::move(params))
{
    m_lmodel = ModelRegistry::getInstance().loadModel(pathToGguf, std::move(loadProgressCb), m_params);
    if (!m_lmodel) {
        throw std::runtime_error("Failed to load model");
    }

    for(auto& loraPath: loras) {
        auto lora = ModelRegistry::getInstance().loadLora(this, loraPath);
        m_loras.push_back(lora);
    }
}

Model::~Model() = default;


uint32_t Model::trainCtxLength() const noexcept {
    return uint32_t(llama_n_ctx_train(m_lmodel.get()));
}

bool Model::shouldAddBosToken() const noexcept {
    return llama_add_bos_token(m_lmodel.get());
}

bool Model::hasEncoder() const noexcept {
    return llama_model_has_encoder(m_lmodel.get());
}

std::string Model::getChatTemplateId() const {
    // load template from model
    constexpr size_t bufSize = 2048; // longest known template is about 1200 bytes
    std::unique_ptr<char[]> tplBuf(new char[bufSize]);

    const char* key = "tokenizer.chat_template";

    int32_t len = llama_model_meta_val_str(m_lmodel.get(), key, tplBuf.get(), bufSize);
    if (len < 0) {
        return "chatml"; // default fallback
    }

    return std::string(tplBuf.get(), len);
}

std::shared_ptr<llama_model> ModelRegistry::loadModel(
    const std::string& gguf,
    ModelLoadProgressCb pcb,
    Model::Params params) {
    std::shared_ptr<llama_model> model = nullptr;
    auto key = ModelKey{gguf, params};
    for (auto& m: m_models) {
        if (m.first == key) {
            return m.second.lock();
        }
    }

    if (!model) {
        model = std::shared_ptr<llama_model>(llama_load_model_from_file(gguf.c_str(), llamaFromModelParams(params, pcb)), llama_free_model);
        m_models.push_back({key, model});
    }

    return model;
}

std::shared_ptr<LoraAdapter> ModelRegistry::loadLora(Model* model, const std::string& loraPath) {
    auto loadedLorasIt = m_loras.find(model->lmodel());
    if (loadedLorasIt == m_loras.end()) {
        m_loras[model->lmodel()] = {};
    }

    auto loadedLoras = m_loras[model->lmodel()];

    loadedLoras.erase(std::find_if(loadedLoras.begin(), loadedLoras.end(), [&](const auto& lora) {
        return lora.expired();
    }), loadedLoras.end());

    for (auto& lora : loadedLoras) {
        if (!lora.expired() && lora.lock()->path() == loraPath) {
            return lora.lock();
            break;
        }
    }

    std::shared_ptr<LoraAdapter> lora = std::make_shared<LoraAdapter>(*model, loraPath);
    loadedLoras.push_back(lora);

    return lora;
}


} // namespace ac::llama
