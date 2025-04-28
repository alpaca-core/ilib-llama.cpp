// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// trivial example of using alpaca-core's llama inference

// llama
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Session.hpp>
#include <ac/llama/ControlVector.hpp>
#include <ac/llama/ResourceCache.hpp>
#include <ac/llama/LogitComparer.hpp>

// logging
#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/ColorSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>

struct GenerationStepData {
    std::string tokenStr;
    int32_t token;
    ac::llama::TokenDataVector data;
};

ac::local::ResourceManager g_rm;
ac::llama::ResourceCache g_rcache(g_rm);

class Model {
public:
    Model(const std::string& gguf, ac::llama::Model::Params params) {
        m_model = g_rcache.getModel({.gguf = gguf, .params = {params}});
        m_instance.reset(new ac::llama::Instance(*m_model, {
            .ctxSize = 2048,
        }));
    }

    struct GenerationResult {
        std::string initalPrompt;
        std::string result;
        std::vector<GenerationStepData> steps;
    };

    GenerationResult generate(std::string prompt, uint32_t maxTokens) {
        m_session = &m_instance->startSession({});

        auto promptTokens = m_model->vocab().tokenize(prompt, true, true);
        m_session->setInitialPrompt(promptTokens);

        constexpr int32_t topK = 10;
        auto data = m_session->getSampledTokenData(topK);

        auto token = promptTokens.back();
        auto tokenStr = m_model->vocab().tokenToString(token);

        std::vector<GenerationStepData> genSteps;
        genSteps.push_back(GenerationStepData{
            .tokenStr = tokenStr,
            .token = token,
            .data = std::move(data)
        });

        std::string result = "";
        for (size_t i = 0; i < maxTokens; i++) {
            auto token = m_session->getToken();
            if (token == ac::llama::Token_Invalid) {
                // no more tokens
                break;
            }
            tokenStr = m_model->vocab().tokenToString(token);
            result += tokenStr;

            auto data = m_session->getSampledTokenData(topK);

            genSteps.push_back({
                .tokenStr = tokenStr,
                .token = token,
                .data = std::move(data)
            });
        }

        m_instance->stopSession();
        m_session = nullptr;

        return {
            .initalPrompt = prompt,
            .result = result,
            .steps = genSteps
        };
    }

private:
    ac::llama::ResourceCache::ModelLock m_model;
    std::unique_ptr<ac::llama::Instance> m_instance;
    ac::llama::Session* m_session;
};

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    std::vector<GenerationStepData> genSteps;

    // load model
    std::string tmpFolder = AC_TEST_DATA_LLAMA_DIR "/../../../tmp/";
    std::string modelGguf = "Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf";
    std::string modelGguf2 = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";

    Model m1(tmpFolder + modelGguf, {});
    Model m2(tmpFolder + modelGguf2, {});

    std::string prompt = "The first person to";
    std::cout << "Prompt: " << prompt << "\n";

    std::string result = prompt;

    std::cout << "Models to compare:\n" << modelGguf << "\n" << modelGguf2 << "\n";
    std::cout << "Comparing...\n";

    for (int i = 0; i < 1; ++i) {

        auto res = m1.generate(prompt, 100);
        std::cout << "Model 1 generated: " << res.result << "\n";
        std::string genPrompt = res.initalPrompt;
        for (size_t i = 0; i < res.steps.size(); i++) {
            auto& step  = res.steps[i];
            if (i > 0) {
                genPrompt += step.tokenStr;
            }
            auto res2 = m2.generate(genPrompt, 0);
            assert(res2.steps.size() == 1);

            if (ac::llama::LogitComparer::compare(step.data, res2.steps[0].data)) {
                std::cout << "Models are the same. Generated str by now:\n" << genPrompt << "\n\n";
            }
        }
    }
    std::cout << '\n';

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
