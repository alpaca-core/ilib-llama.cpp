// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Session.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/InstanceEmbedding.hpp>
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/AntipromptManager.hpp>
#include <ac/llama/ControlVector.hpp>
#include <ac/llama/LogitComparer.hpp>
#include <ac/llama/ResourceCache.hpp>

#include <ac/local/Service.hpp>
#include <ac/local/ServiceFactory.hpp>
#include <ac/local/ServiceInfo.hpp>
#include <ac/local/Backend.hpp>
#include <ac/local/BackendWorkerStrand.hpp>

#include <ac/schema/LlamaCpp.hpp>
#include <ac/schema/FrameHelpers.hpp>
#include <ac/schema/StateChange.hpp>
#include <ac/schema/Error.hpp>
#include <ac/schema/OpTraits.hpp>

#include <ac/frameio/IoEndpoint.hpp>

#include <ac/xec/coro.hpp>
#include <ac/xec/co_spawn.hpp>
#include <ac/io/exception.hpp>

#include <astl/move.hpp>
#include <astl/move_capture.hpp>
#include <astl/iile.h>
#include <astl/throw_stdex.hpp>
#include <astl/workarounds.h>

#include "aclp-llama-version.h"
#include "aclp-llama-interface.hpp"

namespace ac::local {

namespace {

namespace sc = schema::llama;
using namespace ac::frameio;

class ChatSession {
    llama::Session& m_session;
    const llama::Vocab& m_vocab;
    llama::Instance& m_instance;
    IoEndpoint& m_io;
    std::string m_userPrefix;
    std::string m_assistantPrefix;

    std::vector<llama::Token> m_promptTokens;

    bool m_addUserPrefix = true;
    bool m_addAssistantPrefix = true;
public:
    using Schema = sc::StateChatInstance;

    ChatSession(llama::Instance& instance, IoEndpoint& io, sc::StateModelLoaded::OpStartInstance::Params& params)
        : m_session(instance.startSession({}))
        , m_vocab(instance.model().vocab())
        , m_instance(instance)
        , m_io(io)
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

    ~ChatSession() {
        m_instance.stopSession();
    }

    xec::coro<void> pushPrompt(Schema::OpAddChatPrompt::Params& params) {
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

        co_await m_io.push(Frame_from(schema::SimpleOpReturn<Schema::OpAddChatPrompt>{}, {}));
    }

    xec::coro<void> getResponse(Schema::ChatResponseParams params, bool isStreaming) {
        int maxTokens = params.maxTokens.value();
        // handle unlimited generation
        if (maxTokens == 0) {
            maxTokens = 1000;
        }

        if (m_addAssistantPrefix) {
            // generated responses are requested first, but we haven't yet fed the assistant prefix to the model
            auto prompt = m_assistantPrefix;
            assert(m_promptTokens.empty()); // nothing should be pending here
            m_promptTokens = m_vocab.tokenize(prompt, false, false);
            m_session.pushPrompt(m_promptTokens);
        }

        ac::llama::AntipromptManager antiprompt;
        antiprompt.addAntiprompt(m_userPrefix);

        m_addUserPrefix = true;
        Schema::OpGetChatResponse::Return ret;
        auto& result = ret.response.materialize();

        for (int i = 0; i < maxTokens; ++i) {
            auto t = m_session.getToken();
            if (t == ac::llama::Token_Invalid) {
                // no more tokens
                break;
            }

            auto tokenStr = m_vocab.tokenToString(t);
            result += tokenStr;

            auto matchedAntiPrompt = antiprompt.feedGeneratedText(tokenStr);
            if (!matchedAntiPrompt.empty()) {
                // user prefix was added by generation, so don't add it again
                m_addUserPrefix = false;

                // and also hide it from the return value
                // note that we assume that m_userPrefix is always the final piece of text in the response
                // TODO: update to better match the cutoff when issue #131 is done
                result.erase(result.size() - matchedAntiPrompt.size());
                m_addUserPrefix = false;
                break;
            }

            if (isStreaming && !antiprompt.hasRunningAntiprompts()) {
                co_await m_io.push(Frame_from(sc::StreamToken{}, result));
                result = {};
            }
        }

        // remove leading space if any
        // we could add the space to the assistant prefix, but most models have a much easier time generating tokens
        // with a leading space, so instead of burdening them with "unorthodox" tokens, we'll clear it here
        if (!result.empty() && result[0] == ' ') {
            result.erase(0, 1);
        }

        if (isStreaming) {
            if (!result.empty()) {
                co_await m_io.push(Frame_from(sc::StreamToken{}, result));
                result = {};
            }
            co_await m_io.push(Frame_from(schema::SimpleOpReturn<Schema::OpStreamChatResponse>{}, {}));
        } else {
            co_await m_io.push(Frame_from(Schema::OpGetChatResponse{}, {
                .response = std::move(result)
            }));
        }
    }
};

namespace sc = schema::llama;

struct LocalLlama {
    Backend& m_backend;
    llama::ResourceCache& m_resourceCache;
public:
    LocalLlama(Backend& backend, llama::ResourceCache& resourceCache)
        : m_backend(backend)
        , m_resourceCache(resourceCache)
    {}

    static Frame unknownOpError(const Frame& f) {
        return Frame_from(schema::Error{}, "llama: unknown op: " + f.op);
    }

    template <typename ReturnParams>
    static ReturnParams InstanceParams_fromSchema(sc::StateModelLoaded::InstanceParams& params) {
        ReturnParams ret;
        if (params.batchSize.hasValue()) {
            ret.batchSize = params.batchSize.valueOr(2048);
        }
        if (params.ctxSize.hasValue()) {
            ret.ctxSize = params.ctxSize.valueOr(1024);
        }
        if (params.ubatchSize.hasValue()) {
            ret.ubatchSize = params.ubatchSize.valueOr(512);
        }
        return ret;
    }

    sc::StateGeneralInstance::OpRun::Return opRun(llama::Instance& instance, const sc::StateGeneralInstance::OpRun::Params& iparams) {
        auto& prompt = iparams.prompt.value();
        auto maxTokens = iparams.maxTokens.valueOr(0);

        auto& session = instance.startSession({});

        auto promptTokens = instance.model().vocab().tokenize(prompt, true, true);
        session.setInitialPrompt(promptTokens);

        ac::llama::AntipromptManager antiprompt;
        for (auto& ap : iparams.antiprompts.value()) {
            antiprompt.addAntiprompt(ap);
        }

        sc::StateGeneralInstance::OpRun::Return ret;
        auto& result = ret.result.materialize();
        for (unsigned int i = 0; i < maxTokens; ++i) {
            auto t = session.getToken();
            if (t == ac::llama::Token_Invalid) {
                break;
            }

            auto tokenStr = instance.model().vocab().tokenToString(t);
            auto matchedAntiPrompt = antiprompt.feedGeneratedText(tokenStr);
            result += tokenStr;
            if (!matchedAntiPrompt.empty()) {
                result.erase(result.size() - matchedAntiPrompt.size());
                break;
            }
        }

        instance.stopSession();

        return ret;
    }

    xec::coro<void> opStream(
        llama::Instance& instance,
        IoEndpoint& io,
        const sc::StateGeneralInstance::OpStream::Params& iparams) {

        auto& prompt = iparams.prompt.value();
        auto maxTokens = iparams.maxTokens.valueOr(0);

        auto& session = instance.startSession({});

        auto promptTokens = instance.model().vocab().tokenize(prompt, true, true);
        session.setInitialPrompt(promptTokens);

        ac::llama::AntipromptManager antiprompt;
        for (auto& ap : iparams.antiprompts.value()) {
            antiprompt.addAntiprompt(ap);
        }

        for (unsigned int i = 0; i < maxTokens; ++i) {
            auto t = session.getToken();
            if (t == ac::llama::Token_Invalid) {
                break;
            }

            auto tokenStr = instance.model().vocab().tokenToString(t);
            auto matchedAntiPrompt = antiprompt.feedGeneratedText(tokenStr);
            co_await io.push(Frame_from(sc::StreamToken{}, tokenStr));
            if (!matchedAntiPrompt.empty()) {
                break;
            }
        }

        instance.stopSession();

        co_await io.push(Frame_from(schema::SimpleOpReturn<sc::StateGeneralInstance::OpStream>{}, {}));
    }

    xec::coro<void>  opGetTokenData(
        llama::Instance& instance,
        IoEndpoint& io,
        const sc::StateGeneralInstance::OpGetTokenData::Params& iparams) {

        auto& s = instance.startSession({});

        constexpr int32_t topKElements = 10;
        auto tokenData = s.getSampledTokenData(topKElements);

        std::vector<int32_t> tokens(tokenData.size());
        std::vector<float> logits(tokenData.size());
        std::vector<float> probs(tokenData.size());
        for (size_t i = 0; i < tokenData.size(); i++) {
            tokens[i] = tokenData[i].token;
            logits[i] = tokenData[i].logit;
            probs[i] = tokenData[i].prob;
        }

        instance.stopSession();

        co_await io.push(Frame_from(sc::StateGeneralInstance::OpGetTokenData{}, {
            .tokens = std::move(tokens),
            .logits = std::move(logits),
            .probs = std::move(probs)
        }));
    }

    xec::coro<void>  opCompareTokenData(
        llama::Instance& instance,
        IoEndpoint& io,
        const sc::StateGeneralInstance::OpCompareTokenData::Params& iparams) {

        auto& l1 = iparams.logits1.value();
        auto& l2 = iparams.logits2.value();
        auto& p1 = iparams.probs1.value();
        auto& p2 = iparams.probs2.value();
        auto& t1 = iparams.tokens1.value();
        auto& t2 = iparams.tokens2.value();
        assert(l1.size() == t1.size() && l1.size() == p1.size());
        assert(l2.size() == t2.size() && l2.size() == p2.size());

        ac::llama::TokenDataVector data1;
        data1.resize(t1.size());
        for (size_t i = 0; i < t1.size(); i++) {
            data1[i] = ac::llama::TokenData{
                .token = t1[i],
                .logit = l1[i],
                .prob = p1[i]
            };
        }

        ac::llama::TokenDataVector data2;
        data2.resize(t2.size());
        for (size_t i = 0; i < t2.size(); i++) {
            data2[i] = ac::llama::TokenData{
                .token = t2[i],
                .logit = l2[i],
                .prob = p2[i]
            };
        }

        co_await io.push(Frame_from(sc::StateGeneralInstance::OpCompareTokenData{}, {
            .equal = ac::llama::LogitComparer::compare(data1, data2)
        }));
    }

    xec::coro<void> runGeneralInstance(IoEndpoint& io, llama::Instance& instance) {
        using Schema = sc::StateGeneralInstance;
        co_await io.push(Frame_from(schema::StateChange{}, Schema::id));

        while(true) {
            auto f = co_await io.poll();
            Frame err;

            try {
                if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpRun>{}, *f)) {
                    co_await io.push(Frame_from(Schema::OpRun{}, opRun(instance, *iparams)));
                } else if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpStream>{}, *f)) {
                    co_await opStream(instance, io, *iparams);
                } else if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpGetTokenData>{}, *f)) {
                    co_await opGetTokenData(instance, io, *iparams);
                } else if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpCompareTokenData>{}, *f)) {
                    co_await opCompareTokenData(instance, io, *iparams);
                } else {
                    err = unknownOpError(*f);
                }
            }
            catch (std::runtime_error& e) {
                err = Frame_from(schema::Error{}, e.what());
            }

            if (!err.op.empty()) {
                co_await io.push(err);
            }
        }
    }

    xec::coro<void> runEmbeddingInstance(IoEndpoint& io, llama::InstanceEmbedding& instance) {
        using Schema = sc::StateEmbeddingInstance;
        co_await io.push(Frame_from(schema::StateChange{}, Schema::id));

        while(true) {
            auto f = co_await io.poll();

            Frame err;

            try {
                if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpRun>{}, *f)) {
                    auto& prompt = iparams->prompt.value();

                    auto promptTokens = instance.model().vocab().tokenize(prompt, true, true);
                    auto embVec = instance.getEmbeddingVector(promptTokens);

                    co_await io.push(Frame_from(Schema::OpRun{}, {
                        .result = std::move(embVec)
                    }));
                } else {
                    err = unknownOpError(*f);
                }
            }
            catch (std::runtime_error& e) {
                err = Frame_from(schema::Error{}, e.what());
            }

            if (!err.op.empty()) {
                co_await io.push(err);
            }
        }
    }

    xec::coro<void> runChatInstance(IoEndpoint& io, llama::Instance& instance, sc::StateModelLoaded::InstanceParams& params) {
        using Schema = sc::StateChatInstance;
        co_await io.push(Frame_from(schema::StateChange{}, Schema::id));

        ChatSession chatSession(instance, io, params);

        while(true) {
            auto f = co_await io.poll();

            Frame err;

            try {
                if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpAddChatPrompt>{}, *f)) {
                    co_await chatSession.pushPrompt(*iparams);
                } else if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpGetChatResponse>{}, *f)) {
                    co_await chatSession.getResponse(*iparams, false);
                } else if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpStreamChatResponse>{}, *f)) {
                    co_await chatSession.getResponse(*iparams, true);
                } else {
                    err = unknownOpError(*f);
                }
            }
            catch (std::runtime_error& e) {
                err = Frame_from(schema::Error{}, e.what());
            }

            if (!err.op.empty()) {
                co_await io.push(err);
            }
        }
    }

    xec::coro<void> runModel(IoEndpoint& io, sc::StateLlama::OpLoadModel::Params& lmParams) {
        auto gguf = lmParams.ggufPath.valueOr("");
        auto loraPaths = lmParams.loraPaths.valueOr({});

        llama::Model::Params lparams;
        lparams.gpu = lmParams.useGpu.valueOr(true);
        lparams.vocabOnly = lmParams.vocabOnly.valueOr(false);
        lparams.prefixInputsWithBos = lmParams.prefixInputsWithBos.valueOr(false);


        auto model = m_resourceCache.getModel({.gguf = gguf, .params = lparams});

        std::vector<llama::ResourceCache::LoraLock> loras;
        for (auto& loraPath : loraPaths) {
            loras.push_back(model->getLora({loraPath}));
        }

        using Schema = sc::StateModelLoaded;
        co_await io.push(Frame_from(schema::StateChange{}, Schema::id));

        while (true) {
            auto f = co_await io.poll();

            Frame err;

            try {
                if (auto iparams = Frame_optTo(schema::OpParams<Schema::OpStartInstance>{}, *f)) {
                    if (iparams->instanceType == "general" || iparams->instanceType == "chat") {
                        llama::Instance instance(*model, InstanceParams_fromSchema<llama::Instance::InitParams>(*iparams));
                        for (auto& lora : loras) {
                            instance.addLora(*lora, 1.f);
                        }
                        auto ctrlVectors = iparams->ctrlVectorPaths.valueOr({});
                        if (ctrlVectors.size()) {
                            std::vector<llama::ControlVector::LoadInfo> ctrlloadInfo;
                            for (auto& path : ctrlVectors) {
                                ctrlloadInfo.push_back({path, 2});
                            }
                            ac::llama::ControlVector ctrl(*model, ctrlloadInfo);
                            instance.addControlVector(ctrl);
                        }
                        if (iparams->instanceType == "chat") {
                            co_await runChatInstance(io, instance, *iparams);
                        }
                        else {
                            co_await runGeneralInstance(io, instance);
                        }
                    }
                    else if (iparams->instanceType == "embedding") {
                        llama::InstanceEmbedding instance(*model, InstanceParams_fromSchema<llama::InstanceEmbedding::InitParams>(*iparams));
                        co_await runEmbeddingInstance(io, instance);
                    }
                    else {
                        err = Frame_from(schema::Error{}, "llama: unknown instance type: " + iparams->instanceType.value());
                    }
                }
                else {
                    err = unknownOpError(*f);
                }
            }
            catch (std::runtime_error& e) {
                err = Frame_from(schema::Error{}, e.what());
            }

            co_await io.push(err);
        }
    }

    xec::coro<void> runSession(IoEndpoint& io) {
        using Schema = sc::StateLlama;

        co_await io.push(Frame_from(schema::StateChange{}, Schema::id));

        while (true) {
            auto f = co_await io.poll();

            Frame err;

            try {
                if (auto lm = Frame_optTo(schema::OpParams<Schema::OpLoadModel>{}, * f)) {
                    co_await runModel(io, *lm);
                }
                else {
                    err = unknownOpError(*f);
                }
            }
            catch (std::runtime_error& e) {
                err = Frame_from(schema::Error{}, e.what());
            }

            co_await io.push(err);
        }
    }

    xec::coro<void> run(frameio::StreamEndpoint ep, xec::strand ex) {
        try {
            IoEndpoint io(std::move(ep), ex);
            co_await runSession(io);
        }
        catch (io::stream_closed_error&) {
            co_return;
        }
    }
};

ServiceInfo g_serviceInfo = {
    .name = "ac llama.cpp",
    .vendor = "Alpaca Core",
};

struct LlamaService final : public Service {
    LlamaService(BackendWorkerStrand& ws) : m_workerStrand(ws) {}

    BackendWorkerStrand& m_workerStrand;
    llama::ResourceCache m_resourceCache{m_workerStrand.resourceManager};
    std::shared_ptr<LocalLlama> llama;

    virtual const ServiceInfo& info() const noexcept override {
        return g_serviceInfo;
    }

    virtual void createSession(frameio::StreamEndpoint ep, Dict) override {
        llama = std::make_shared<LocalLlama>(m_workerStrand.backend, m_resourceCache);
        co_spawn(m_workerStrand.executor(), llama->run(std::move(ep), m_workerStrand.executor()));
    }
};

struct LlamaServiceFactory final : public ServiceFactory {
    virtual const ServiceInfo& info() const noexcept override {
        return g_serviceInfo;
    }
    virtual std::unique_ptr<Service> createService(Backend& backend) const override {
        auto svc = std::make_unique<LlamaService>(backend.gpuWorkerStrand());
        return svc;
    }
};

} // namespace

} // namespace ac::local

namespace ac::llama {

void init() {
    initLibrary();
}

std::vector<const local::ServiceFactory*> getFactories() {
    static local::LlamaServiceFactory factory;
    return {&factory};
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
        .getServiceFactories = getFactories,
    };
}

} // namespace ac::llama
