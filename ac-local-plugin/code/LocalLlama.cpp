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
#include <ac/schema/OpDispatchHelpers.hpp>
#include <ac/schema/FrameHelpers.hpp>

#include <ac/FrameUtil.hpp>
#include <ac/frameio/IoEndpoint.hpp>

#include <ac/xec/coro.hpp>
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

struct BasicRunner {
    schema::OpDispatcherData m_dispatcherData;

    Frame dispatch(Frame& f) {
        try {
            auto ret = m_dispatcherData.dispatch(f.op, std::move(f.data));
            if (!ret) {
                throw_ex{} << "llama: unknown op: " << f.op;
            }
            return {f.op, *ret};
        }
        catch (io::stream_closed_error&) {
            throw;
        }
        catch (std::exception& e) {
            return {"error", e.what()};
        }
    }
};

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
    using Schema = sc::StateChat;

    ChatSession(llama::Instance& instance, IoEndpoint& io, sc::StateInstance::OpChatBegin::Params& params)
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

    void pushPrompt(Schema::OpAddChatPrompt::Params& params) {
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

    xec::coro<std::optional<std::string>> getResponse(Schema::OpGetChatResponse::Params params) {
        using SchemaStreaming = sc::StateStreaming;

        int maxTokens = params.maxTokens.value();
        // handle unlimited generation
        if (maxTokens == 0) {
            maxTokens = 1000;
        }
        const bool isStreaming = params.stream.value();
        if (isStreaming) {
            co_await m_io.push(Frame_stateChange(SchemaStreaming::id));
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
                co_await m_io.push(Frame_fromStreamType(SchemaStreaming::StreamToken{}, result));
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
                co_await m_io.push(Frame_fromStreamType(sc::StateStreaming::StreamToken{}, result));
                result = {};
            }
            co_await m_io.push(Frame_stateChange(Schema::id));
        }

        co_return result;
    }
};

xec::coro<void> Llama_beginChat(IoEndpoint& io, ChatSession& chat) {
    using Schema = sc::StateChat;

    struct Runner : public BasicRunner {
        ChatSession& chatSession;
        xec::coro<std::optional<std::string>> nextCoro;
        bool immediatePush = true;

        enum class State {
            Running,
            End
        } state = State::Running;

        Runner(ChatSession& chat)
            : chatSession(chat)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        Schema::OpChatEnd::Return on(Schema::OpChatEnd, Schema::OpChatEnd::Params&&) {
            state = State::End;
            return {};
        }

        Schema::OpAddChatPrompt::Return on(Schema::OpAddChatPrompt, Schema::OpAddChatPrompt::Params&& params) {
            chatSession.pushPrompt(params);
            return {};
        }

        Schema::OpGetChatResponse::Return on(Schema::OpGetChatResponse, Schema::OpGetChatResponse::Params&& params) {
            nextCoro = chatSession.getResponse(params);
            immediatePush = params.stream.value();
            return {
                .response = ""
            };
        }
    };

    co_await io.push(Frame_stateChange(Schema::id));

    Runner runner(chat);
    while (true) {
        auto f = co_await io.poll();
        auto dispatchRes = runner.dispatch(*f);
        if (runner.nextCoro) {
            if (runner.immediatePush) {
                co_await io.push(dispatchRes);
                runner.immediatePush = false;
            }
            auto coroRes = co_await runner.nextCoro;
            if (coroRes.has_value() && !coroRes.value().empty()) {
                auto d = Struct_toDict(Schema::OpGetChatResponse::Return{.response = coroRes.value()});
                co_await io.push({f->op, d});
            }
        } else {
            co_await io.push(dispatchRes);
        }
        if (runner.state == Runner::State::End) {
            co_return;
        }
    }
}

xec::coro<void> Llama_runInstance(IoEndpoint& io, std::unique_ptr<llama::Instance> instance) {
    using Schema = sc::StateInstance;

    struct Runner : public BasicRunner {
        llama::Instance& m_instance;
        IoEndpoint& io;
        std::optional<ChatSession> m_chatSession;
        xec::coro<std::optional<std::string>> nextCoro;
        bool immediatePush = false;

        enum class State {
            Running,
            End
        } state = State::Running;

        Runner(llama::Instance& instance, IoEndpoint& io)
            : m_instance(instance)
            , io(io)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        Schema::OpRun::Return on(Schema::OpRun, Schema::OpRun::Params&& params) {
            if (m_chatSession) {
                throw_ex{} << "llama: chat already started";
            }

            nextCoro = runOp(params);
            immediatePush = params.stream.value();
            return {
                .result = ""
            };
        }

        xec::coro<std::optional<std::string>> runOp(Schema::OpRun::Params params) {
            using SchemaStreaming = sc::StateStreaming;

            auto& prompt = params.prompt.value();
            const auto maxTokens = params.maxTokens.value();
            const bool isStreaming = params.stream.value();

            auto& s = m_instance.startSession({});

            auto promptTokens = m_instance.model().vocab().tokenize(prompt, true, true);
            s.setInitialPrompt(promptTokens);

            auto& model = m_instance.model();
            ac::llama::AntipromptManager antiprompt;

            for (auto& ap : params.antiprompts.value()) {
                antiprompt.addAntiprompt(ap);
            }

            if (isStreaming) {
                co_await io.push(Frame_stateChange(SchemaStreaming::id));
            }

            Schema::OpRun::Return ret;
            auto& result = ret.result.materialize();
            for (unsigned int i = 0; i < maxTokens; ++i) {
                auto t = s.getToken();
                if (t == ac::llama::Token_Invalid) {
                    break;
                }

                auto tokenStr = model.vocab().tokenToString(t);
                auto matchedAntiPrompt = antiprompt.feedGeneratedText(tokenStr);
                result += tokenStr;
                if (!matchedAntiPrompt.empty()) {
                    result.erase(result.size() - matchedAntiPrompt.size());
                    break;
                }

                if (isStreaming && !antiprompt.hasRunningAntiprompts()) {
                    co_await io.push(Frame_fromStreamType(SchemaStreaming::StreamToken{}, result));
                    result = {};
                }
            }

            m_instance.stopSession();

            if (isStreaming) {
                if (!result.empty()) {
                    co_await io.push(Frame_fromStreamType(sc::StateStreaming::StreamToken{}, result));
                    result = {};
                }
                co_await io.push(Frame_stateChange(Schema::id));
            }

            co_return result;
        }

        Schema::OpGetTokenData::Return on(Schema::OpGetTokenData, Schema::OpGetTokenData::Params&&) {
            if (m_chatSession) {
                throw_ex{} << "llama: chat already started";
            }

            auto& s = m_instance.startSession({});

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

            m_instance.stopSession();

            return {
                .tokens = std::move(tokens),
                .logits = std::move(logits),
                .probs = std::move(probs)
            };
        }

        Schema::OpCompareTokenData::Return on(Schema::OpCompareTokenData, Schema::OpCompareTokenData::Params&& params) {
            assert(params.logits1.value().size() == params.tokens1.value().size() &&
                params.logits1.value().size()== params.probs1.value().size());

            assert(params.logits2.value().size() == params.tokens2.value().size() &&
                params.logits2.value().size()== params.probs2.value().size());

            ac::llama::TokenDataVector data1;
            data1.resize(params.tokens1.value().size());
            for (size_t i = 0; i < params.tokens1.value().size(); i++) {
                data1[i] = ac::llama::TokenData{
                    .token = params.tokens1.value()[i],
                    .logit = params.logits1.value()[i],
                    .prob = params.probs1.value()[i]
                };
            }

            ac::llama::TokenDataVector data2;
            data2.resize(params.tokens2.value().size());
            for (size_t i = 0; i < params.tokens2.value().size(); i++) {
                data2[i] = ac::llama::TokenData{
                    .token = params.tokens2.value()[i],
                    .logit = params.logits2.value()[i],
                    .prob = params.probs2.value()[i]
                };
            }

            return {
                .equal = ac::llama::LogitComparer::compare(data1, data2)
            };
        }

        Schema::OpChatBegin::Return on(Schema::OpChatBegin, Schema::OpChatBegin::Params&& params) {
            m_chatSession.emplace(m_instance, io, params);
            return {};
        }

        Schema::OpStopInstance::Return on(Schema::OpStopInstance, Schema::OpStopInstance::Params&&) {
            state = State::End;
            return {};
        }
    };

    co_await io.push(Frame_stateChange(Schema::id));

    Runner runner(*instance, io);
    while (true) {
        auto f = co_await io.poll();
        auto dispatchRes = runner.dispatch(*f);

        if (runner.nextCoro) {
            if (runner.immediatePush) {
                co_await io.push(dispatchRes);
                runner.immediatePush = false;
            }
            auto coroRes = co_await runner.nextCoro;
            if (coroRes.has_value() && !coroRes.value().empty()) {
                auto d = Struct_toDict(Schema::OpRun::Return{.result = coroRes.value()});
                co_await io.push({f->op, d});
            }
        } else {
            co_await io.push(dispatchRes);
        }

        if (runner.m_chatSession) {
            co_await Llama_beginChat(io, runner.m_chatSession.value());
            runner.m_chatSession.reset();

            co_await io.push(Frame_stateChange(Schema::id));
        }

        if (runner.state == Runner::State::End) {
            co_return;
        }
    }
}

xec::coro<void> Llama_runInstanceEmbedding(IoEndpoint& io, std::unique_ptr<llama::InstanceEmbedding> instance) {
    using Schema = sc::StateEmbeddingInstance;

    struct Runner : public BasicRunner {
        llama::InstanceEmbedding& m_instance;

        Runner(llama::InstanceEmbedding& instance)
            : m_instance(instance)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        Schema::OpRun::Return on(Schema::OpRun, Schema::OpRun::Params&& params) {
            auto& prompt = params.prompt.value();

            auto promptTokens = m_instance.model().vocab().tokenize(prompt, true, true);
            auto embVec = m_instance.getEmbeddingVector(promptTokens);

            return {
                .result = std::move(embVec)
            };
        }
    };

    co_await io.push(Frame_stateChange(Schema::id));

    Runner runner(*instance);
    while (true) {
        auto f = co_await io.poll();
        co_await io.push(runner.dispatch(*f));
    }
}

template <typename Params, typename ReturnParams>
static ReturnParams Params_fromSchema(Params& params) {
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

xec::coro<void> Llama_runModel(IoEndpoint& io, llama::Model& model, std::span<const llama::ResourceCache::LoraLock> loras) {
    using Schema = sc::StateModelLoaded;

    struct Runner : public BasicRunner {
        Runner(llama::Model& model, std::span<const llama::ResourceCache::LoraLock> loras)
            : m_lmodel(model)
            , m_loras(loras)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        llama::Model& m_lmodel;
        std::span<const llama::ResourceCache::LoraLock> m_loras;
        std::unique_ptr<llama::Instance> instance;
        std::unique_ptr<llama::InstanceEmbedding> embeddingInstance;

        Schema::OpStartInstance::Return on(Schema::OpStartInstance, Schema::OpStartInstance::Params iParams) {
            instance = std::make_unique<llama::Instance>(m_lmodel, Params_fromSchema<Schema::OpStartInstance::Params, llama::Instance::InitParams>(iParams));
            for (auto& lora : m_loras) {
                instance->addLora(*lora, 1.f);
            }

            auto ctrlVectors = iParams.ctrlVectorPaths.valueOr({});
            if (ctrlVectors.size()) {
                std::vector<llama::ControlVector::LoadInfo> ctrlloadInfo;
                for (auto& path : ctrlVectors) {
                    ctrlloadInfo.push_back({ path, 2 });
                }
                ac::llama::ControlVector ctrl(m_lmodel, ctrlloadInfo);
                instance->addControlVector(ctrl);
            }

            return {};
        }

        Schema::OpStartEmbeddingInstance::Return on(Schema::OpStartEmbeddingInstance, Schema::OpStartEmbeddingInstance::Params iParams) {
            embeddingInstance = std::make_unique<llama::InstanceEmbedding>(m_lmodel, Params_fromSchema<Schema::OpStartEmbeddingInstance::Params, llama::InstanceEmbedding::InitParams>(iParams));

            return {};
        }
    };

    co_await io.push(Frame_stateChange(Schema::id));

    Runner runner(model, loras);
    while (true) {
        auto f = co_await io.poll();
        co_await io.push(runner.dispatch(*f));
        if (runner.instance) {
            co_await Llama_runInstance(io, std::move(runner.instance));

            runner.instance.reset();
        }

        if (runner.embeddingInstance) {
            co_await Llama_runInstanceEmbedding(io, std::move(runner.embeddingInstance));

            runner.embeddingInstance.reset();
        }
    }
}

xec::coro<void> Llama_runSession(StreamEndpoint ep, llama::ResourceCache& resourceCache) {
    using Schema = sc::StateInitial;

    struct Runner : public BasicRunner {
        Runner(llama::ResourceCache& resourceCache)
            : cache(resourceCache)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        llama::ResourceCache& cache;
        llama::ResourceCache::ModelLock model;
        std::vector<llama::ResourceCache::LoraLock> loras;

        static llama::Model::Params ModelParams_fromSchema(sc::StateInitial::OpLoadModel::Params schemaParams) {
            llama::Model::Params ret;
            ret.gpu = schemaParams.useGpu.valueOr(true);
            ret.vocabOnly = schemaParams.vocabOnly.valueOr(false);
            ret.prefixInputsWithBos = schemaParams.prefixInputsWithBos.valueOr(false);
            return ret;
        }

        Schema::OpLoadModel::Return on(Schema::OpLoadModel, Schema::OpLoadModel::Params params) {
            auto gguf = params.ggufPath.valueOr("");
            auto loraPaths = params.loraPaths.valueOr({});
            auto lparams = ModelParams_fromSchema(params);

            model = cache.getModel({.gguf = gguf, .params = lparams});

            for(auto& loraPath : loraPaths) {
                loras.push_back(model->getLora({loraPath}));
            }

            return {};
        }
    };

    try {
        auto ex = co_await xec::executor{};
        IoEndpoint io(std::move(ep), ex);

        co_await io.push(Frame_stateChange(Schema::id));

        Runner runner(resourceCache);

        while (true) {
            auto f = co_await io.poll();
            co_await io.push(runner.dispatch(*f));
            if (runner.model) {
                co_await Llama_runModel(io, *runner.model, runner.loras);
            }
        }
    }
    catch (io::stream_closed_error&) {
        co_return;
    }
}

ServiceInfo g_serviceInfo = {
    .name = "ac llama.cpp",
    .vendor = "Alpaca Core",
};

struct LlamaService final : public Service {
    LlamaService(BackendWorkerStrand& ws) : m_workerStrand(ws) {}

    BackendWorkerStrand& m_workerStrand;
    llama::ResourceCache m_resourceCache{m_workerStrand.resourceManager};

    virtual const ServiceInfo& info() const noexcept override {
        return g_serviceInfo;
    }

    virtual void createSession(frameio::StreamEndpoint ep, Dict) override {
        co_spawn(m_workerStrand.executor(), Llama_runSession(std::move(ep), m_resourceCache));
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
