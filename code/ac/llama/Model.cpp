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
llama_model_params llamaFromModelParams(const Model::Params& params, ModelLoadProgressCb& loadProgressCb) {
    static ggml_backend_dev_t devicesCpu[] = {
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
         nullptr
    };

    static ggml_backend_dev_t devicesGpu[] = {
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU),
         nullptr
    };

    llama_model_params llamaParams = llama_model_default_params();

    if (params.gpu) {
        llamaParams.devices = devicesGpu;
    } else {
        llamaParams.devices = devicesCpu;
    }

    llamaParams.n_gpu_layers = params.gpu ? 10000 : 0;
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


Model::Model(std::shared_ptr<llama_model> lmodel, Params params)
    : m_params(astl::move(params))
    , m_lmodel(std::move(lmodel))
{}

Model::~Model() = default;


uint32_t Model::trainCtxLength() const noexcept {
    return uint32_t(llama_model_n_ctx_train(m_lmodel.get()));
}

bool Model::shouldAddBosToken() const noexcept {
    return llama_vocab_get_add_bos(m_vocab.lvocab());
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

    // clean up expired models
    m_models.erase(std::find_if(m_models.begin(), m_models.end(), [&](const auto& m) {
        return m.second.expired();
    }), m_models.end());

    std::shared_ptr<llama_model> model = nullptr;
    auto key = ModelKey{gguf, params};
    for (auto& m: m_models) {
        if (m.first == key) {
            return m.second.lock();
        }
    }

    if (!model) {
        model = std::shared_ptr<llama_model>(llama_model_load_from_file(gguf.c_str(), llamaFromModelParams(params, pcb)), llama_model_free);
        if (model == nullptr) {
            throw std::runtime_error("Failed to load model");
        }
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
