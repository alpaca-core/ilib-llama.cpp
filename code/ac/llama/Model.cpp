    // Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Model.hpp"
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

Model::Model(const char* pathToGguf, ModelLoadProgressCb loadProgressCb, Params params)
    : m_params(astl::move(params))
    , m_lmodel(llama_load_model_from_file(pathToGguf, llamaFromModelParams(m_params, loadProgressCb)), llama_free_model)
{
    if (!m_lmodel) {
        throw std::runtime_error("Failed to load model");
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


std::shared_ptr<Model> ModelRegistry::loadModel(
    const std::string& gguf,
    std::span<std::string> loras,
    ModelLoadProgressCb pcb,
    Model::Params params) {
        // TOOD: key must include params
        auto it = m_models.find(gguf);
        if (it != m_models.end() && !it->second.expired()) {
            return it->second.lock();
        }

        //astl::c_unique_ptr<llama_model> model(llama_load_model_from_file(gguf.c_str(), llamaFromModelParams(params, pcb)), llama_free_model)

        std::shared_ptr<Model> model = std::make_shared<Model>(gguf.c_str(), pcb, params);

        std::vector<llama_lora_adapter*> adaptersToApply;
        for(auto& loraPath: loras) {
            auto loraIt = m_loras.find(loraPath);
            if (m_loras.find(loraPath) != m_loras.end()) {
                adaptersToApply.push_back(loraIt->second);
                continue;
            }

            llama_lora_adapter* adapter = llama_lora_adapter_init(model->lmodel(), loraPath.c_str());
            if (!adapter) {
                throw std::runtime_error("Failed to initialize LORA adapter from " + loraPath);
            }

            m_loras.emplace(loraPath, adapter);
            adaptersToApply.push_back(adapter);
        }

        m_models.emplace(gguf, model);

        return model;
    }

} // namespace ac::llama
