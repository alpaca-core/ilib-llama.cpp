// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Session.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/AntipromptManager.hpp>
#include <ac/llama/ControlVector.hpp>

#include <ac/local/Instance.hpp>
#include <ac/local/Model.hpp>
#include <ac/local/ModelLoader.hpp>

#include <ac/schema/LlamaCpp.hpp>
#include <ac/schema/DispatchHelpers.hpp>

#include <astl/move.hpp>
#include <astl/move_capture.hpp>
#include <astl/iile.h>
#include <astl/throw_stdex.hpp>
#include <astl/workarounds.h>

#include "aclp-llama-version.h"
#include "aclp-llama-interface.hpp"

namespace ac::local {

namespace {
class ChatSession {
    llama::Session m_session;
    const llama::Vocab& m_vocab;
    std::string m_userPrefix;
    std::string m_assistantPrefix;

    std::vector<llama::Token> m_promptTokens;

    bool m_addUserPrefix = true;
    bool m_addAssistantPrefix = true;
public:
    using Interface = ac::local::schema::LlamaCppInterface;

    ChatSession(llama::Instance& instance, Interface::OpChatBegin::Params& params)
        : m_session(instance.newSession({}))
        , m_vocab(instance.model().vocab())
    {
        m_promptTokens = instance.model().vocab().tokenize(params.setup.value(), true, true);
        m_session.setInitialPrompt(m_promptTokens);

        m_userPrefix = "\n";
        m_userPrefix += params.roleUser;
        m_userPrefix += ":";
        m_assistantPrefix = "\n";
        m_assistantPrefix += params.roleAssistant;
        m_assistantPrefix += ":";
    }

    void pushPrompt(Interface::OpAddChatPrompt::Params& params) {
        auto& prompt = params.prompt.value();

        // prefix with space as the generated content doesn't include it
        prompt = ' ' + prompt;

        if (m_addUserPrefix) {
            // we haven't had an interaction yet, so we need to add the user prefix
            // subsequent interaction will have it generated
            prompt = m_userPrefix + prompt;
        }

        // prepare for the next generation
        prompt += m_assistantPrefix;

        m_promptTokens = m_vocab.tokenize(prompt, false, false);
        m_session.pushPrompt(m_promptTokens);
        m_addAssistantPrefix = false;
    }

    Interface::OpGetChatResponse::Return getResponse() {
        if (m_addAssistantPrefix) {
            // generated responses are requested first, but we haven't yet fed the assistant prefix to the model
            auto prompt = m_assistantPrefix;
            assert(m_promptTokens.empty()); // nothing should be pending here
            m_promptTokens = m_vocab.tokenize(prompt, false, false);
            m_session.pushPrompt(m_promptTokens);
        }


        ac::llama::IncrementalStringFinder finder(m_userPrefix);

        m_addUserPrefix = true;
        Interface::OpGetChatResponse::Return ret;
        auto& response = ret.response.materialize();


        for (int i = 0; i < 1000; ++i) {
            auto t = m_session.getToken();
            if (t == ac::llama::Token_Invalid) {
                // no more tokens
                break;
            }

            auto tokenStr = m_vocab.tokenToString(t);
            response += tokenStr;

            if (finder.feedText(tokenStr)) {
                // user prefix was added by generation, so don't add it again
                m_addUserPrefix = false;

                // and also hide it from the return value
                // note that we assume that m_userPrefix is always the final piece of text in the response
                // TODO: update to better match the cutoff when issue #131 is done
                response.resize(response.size() - m_userPrefix.size());
                break;
            }
        }

        // remove leading space if any
        // we could add the space to the assistant prefix, but most models have a much easier time generating tokens
        // with a leading space, so instead of burdening them with "unorthodox" tokens, we'll clear it here
        if (!response.empty() && response[0] == ' ') {
            response.erase(0, 1);
        }

        return ret;
    }
};


class LlamaInstance final : public Instance {
    std::shared_ptr<llama::Model> m_model;
    llama::Instance m_instance;

    std::optional<ChatSession> m_chatSession;

    schema::OpDispatcherData m_dispatcherData;
public:
    using Schema = ac::local::schema::LlamaCppLoader::InstanceGeneral;
    using Interface = ac::local::schema::LlamaCppInterface;

    LlamaInstance(std::shared_ptr<llama::Model> model, const ac::llama::ControlVector& ctrlVector, llama::Instance::InitParams params)
        : m_model(astl::move(model))
        , m_instance(*m_model, astl::move(params))
    {
        m_instance.addControlVector(ctrlVector);
        schema::registerHandlers<Interface::Ops>(m_dispatcherData, *this);
    }

    Interface::OpRun::Return on(Interface::OpRun, Interface::OpRun::Params&& params) {
        auto& prompt = params.prompt.value();
        const auto maxTokens = params.maxTokens.value();

        auto s = m_instance.newSession({});

        auto promptTokens = m_instance.model().vocab().tokenize(prompt, true, true);
        s.setInitialPrompt(promptTokens);

        auto& model = m_instance.model();
        ac::llama::AntipromptManager antiprompt;

        for (auto& ap : params.antiprompts.value()) {
            antiprompt.addAntiprompt(ap);
        }

        Interface::OpRun::Return ret;
        auto& result = ret.result.materialize();
        for (unsigned int i = 0; i < maxTokens; ++i) {
            auto t = s.getToken();
            if (t == ac::llama::Token_Invalid) {
                break;
            }

            auto tokenStr = model.vocab().tokenToString(t);
            if (antiprompt.feedGeneratedText(tokenStr)) {
                break;
            }

            result += tokenStr;
        }

        return ret;
    }

    Interface::OpChatBegin::Return on(Interface::OpChatBegin, Interface::OpChatBegin::Params&& params) {
        m_chatSession.emplace(m_instance, params);
        return {};
    }

    Interface::OpAddChatPrompt::Return on(Interface::OpAddChatPrompt, Interface::OpAddChatPrompt::Params&& params) {
        if (!m_chatSession) {
            throw_ex{} << "llama: chat not started";
        }
        m_chatSession->pushPrompt(params);
        return {};
    }

    Interface::OpGetChatResponse::Return on(Interface::OpGetChatResponse, Interface::OpGetChatResponse::Params&&) {
        if (!m_chatSession) {
            throw_ex{} << "llama: chat not started";
        }
        return m_chatSession->getResponse();
    }

    virtual Dict runOp(std::string_view op, Dict params, ProgressCb) override {
        auto ret = m_dispatcherData.dispatch(op, astl::move(params));
        if (!ret) {
            throw_ex{} << "llama: unknown op: " << op;
        }
        return *ret;
    }
};

class LlamaModel final : public Model {
    using Schema = ac::local::schema::LlamaCppLoader;

    std::shared_ptr<llama::Model> m_model;
    std::vector<ac::llama::ControlVector::LoadInfo> m_ctrlVectors;

    llama::Instance::InitParams translateInstanceParams(Dict&& params) {
        llama::Instance::InitParams initParams;
        auto schemaParams = schema::Struct_fromDict<Schema::InstanceGeneral::Params>(astl::move(params));

        if (schemaParams.ctxSize.hasValue()) {
            initParams.ctxSize = schemaParams.ctxSize;
        }

        if (schemaParams.batchSize.hasValue()) {
            initParams.batchSize = schemaParams.batchSize;
        }

        if (schemaParams.ubatchSize.hasValue()) {
            initParams.ubatchSize = schemaParams.ubatchSize;
        }

        return initParams;
    }
public:

    LlamaModel(const std::string& gguf, std::span<std::string> loras, std::vector<llama::ControlVector::LoadInfo>& ctrlVectors, llama::ModelLoadProgressCb pcb, llama::Model::Params params)
        : m_model(std::make_shared<llama::Model>(llama::ModelRegistry::getInstance().loadModel(gguf.c_str(), astl::move(pcb), params), astl::move(params)))
        , m_ctrlVectors(astl::move(ctrlVectors))
    {
        for(auto& loraPath: loras) {
            auto lora = llama::ModelRegistry::getInstance().loadLora(m_model.get(), loraPath);
            m_model->addLora(lora);;
        }
    }

    virtual std::unique_ptr<Instance> createInstance(std::string_view type, Dict params) override {
        ac::llama::ControlVector ctrlVector(*m_model, m_ctrlVectors);
        if (type == "general") {
            return std::make_unique<LlamaInstance>(m_model, ctrlVector, translateInstanceParams(astl::move(params)));
        }
        else {
            throw_ex{} << "llama: unknown instance type: " << type;
            MSVC_WO_10766806();
        }
    }
};

class LlamaModelLoader final : public ModelLoader {
public:
    virtual const Info& info() const noexcept override {
        static Info i = {
            .name = "ac llama.cpp",
            .vendor = "Alpaca Core",
        };
        return i;
    }

    virtual bool canLoadModel(const ModelAssetDesc& desc, const Dict&) const noexcept override {
        return desc.type == "llama.cpp gguf";
    }

    virtual ModelPtr loadModel(ModelAssetDesc desc, Dict, ProgressCb progressCb) override {
        auto& gguf = desc.assets.front().path;

        std::vector<llama::ControlVector::LoadInfo> ctrlVectors;
        std::vector<std::string> loras;
        for (auto& asset : desc.assets) {
            if (asset.tag.find("control_vector:")) {
                ctrlVectors.push_back({asset.path, 2});
            }
            if (asset.tag.find("lora:") != std::string::npos) {
                loras.push_back(asset.path);
            }
        }

        llama::Model::Params modelParams;
        std::string progressTag = "loading " + gguf;
        return std::make_shared<LlamaModel>(gguf, loras, ctrlVectors, [movecap(progressTag, progressCb)](float p) {
            if (progressCb) {
                progressCb(progressTag, p);
            }
        }, astl::move(modelParams));
    }
};
} // namespace

} // namespace ac::local

namespace ac::llama {

void init() {
    initLibrary();
}

std::vector<ac::local::ModelLoaderPtr> getLoaders() {
    std::vector<ac::local::ModelLoaderPtr> ret;
    ret.push_back(std::make_unique<local::LlamaModelLoader>());
    return ret;
}

local::PluginInterface getPluginInterface() {
    return {
        .label = "ac llama.cpp",
        .desc = "llama.cpp plugin for ac-local",
        .vendor = "Alpaca Core",
        .version = astl::version{
            ACLP_llama_VERSION_MAJOR, ACLP_llama_VERSION_MINOR, ACLP_llama_VERSION_PATCH
        },
        .init = init,
        .getLoaders = getLoaders,
    };
}

} // namespace ac::llama
