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
        m_session = &m_instance->startSession({});
        m_session->setInitialPrompt({}); // empty prompt
    }

    ~Model() {
        m_instance->stopSession();
    }

    struct GenerationResult {
        std::string initalPrompt;
        std::string result;
        std::vector<GenerationStepData> steps;
    };

    GenerationResult generate(std::string prompt, uint32_t maxTokens) {
        auto promptTokens = m_model->vocab().tokenize(prompt, true, true);
        return generate_impl(promptTokens, maxTokens);
    }

    GenerationResult generate(std::span<ac::llama::Token> prompt, uint32_t maxTokens) {
        return generate_impl(prompt, maxTokens);
    }

    std::vector<ac::llama::Token> tokenize(std::string prompt) {
        return m_model->vocab().tokenize(prompt, true, true);
    }

    bool tokenExists(ac::llama::Token token) {
        return m_model->vocab().nTokens() > token;
    }

private:
    GenerationResult generate_impl(std::span<ac::llama::Token> promptTokens, uint32_t maxTokens) {
        if (!promptTokens.empty()) {
            m_session->pushPrompt(promptTokens, {});
        }

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

        std::string initialPrompt = "";
        for (size_t i = 0; i < promptTokens.size(); i++){
            initialPrompt += m_model->vocab().tokenToString(promptTokens[i], false);
        }

        return {
            .initalPrompt = std::move(initialPrompt),
            .result = std::move(result),
            .steps = std::move(genSteps)
        };
    }

private:
    ac::llama::ResourceCache::ModelLock m_model;
    std::unique_ptr<ac::llama::Instance> m_instance;
    ac::llama::Session* m_session;
};

// -- Helper function to compute normalized entropy --
float normalizedEntropy(const ac::llama::TokenDataVector& data) {
    std::vector<float> probs(data.size());
    float sum = 0.0f;

    // Calculate softmax probabilities
    for (auto& val : data) {
        sum += std::exp(val.logit);
    }
    for (size_t i = 0; i < data.size(); ++i) {
        probs[i] = std::exp(data[i].logit) / sum;
    }

    // Calculate entropy
    float entropy = 0.0f;
    for (float p : probs) {
        if (p > 0.0f) {
            entropy -= p * std::log(p);
        }
    }

    // Normalize entropy by maximum possible entropy (log(number of classes))
    float maxEntropy = std::log(float(probs.size()));
    return entropy / maxEntropy;
}




int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    std::vector<GenerationStepData> genSteps;

    // load model
    std::string tmpFolder = AC_TEST_DATA_LLAMA_DIR "/../../../tmp/";
    // std::string modelGguf = "Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf";
    std::string modelGguf = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";
    // std::string modelGguf = "BgGPT-Gemma-2-2B-IT-v1.0.Q8_0.gguf";
    // std::string modelGguf = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";
    std::string modelGguf2 = "Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf";
    // std::string modelGguf2 = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";

    Model m1(tmpFolder + modelGguf, {});
    Model m2(tmpFolder + modelGguf2, {});

    std::string prompt = "The first person to";
    std::cout << "Prompt: " << prompt << "\n";

    std::string result = prompt;

    std::cout << "Models to compare:\n" << modelGguf << "\n" << modelGguf2 << "\n";
    std::cout << "Comparing...\n";

    std::vector<float> jsdResults;
    for (int i = 0; i < 1; ++i) {

        auto res = m1.generate(prompt, 100);
        std::cout << "Model 1 generated: " << res.result << "\n";
        std::string genPrompt = res.initalPrompt;

        auto genPromptTokens = m2.tokenize(genPrompt);

        float totalWeightedDist = 0.0f;
        float totalWeight = 0.0f;

        for (size_t i = 0; i < res.steps.size(); i++) {
            auto& step  = res.steps[i];
            if (i > 0) {
                if (m2.tokenExists(step.token)) {
                    genPromptTokens.push_back(step.token);
                }
                else {
                    // Instead of skipping, penalize fully
                    float fakeDist = 1.0f; // Maximum possible distance
                    float weight = 1.0f;    // Assume maximum confidence since we can't know entropy
                    totalWeightedDist += weight * fakeDist;
                    totalWeight += weight;

                    jsdResults.push_back(1);

                    std::cout << "Token not found in model 2: " << step.tokenStr << "\n";
                    continue;
                }
            }

            Model::GenerationResult res2;
            if (i == 0) {
                res2 = m2.generate(genPromptTokens, 0);
            } else {
                std::vector<ac::llama::Token> token{step.token};
                res2 = m2.generate(token, 0);
            }

            assert(res2.steps.size() == 1);

            {
                // Step 1: Compare logits
                float dist = ac::llama::LogitComparer::cosineDistance(step.data, res2.steps[0].data);

                // Step 2: Calculate confidence weight
                float entropy = normalizedEntropy(step.data);
                float weight = 1.0f - entropy; // high confidence = high weight

                // Step 3: Accumulate weighted distance
                totalWeightedDist += weight * dist;
                totalWeight += weight;
            }

            {
                float jsd = ac::llama::LogitComparer::JSD(step.data, res2.steps[0].data);
                jsdResults.push_back(jsd);
            }

        }

        // Final step: Normalize

        // Score range | Interpretation
        // 0.0 | Perfect match (identical predictions)
        // 0.0001 - 0.001 | Practically indistinguishable
        // 0.001 - 0.01 | Very close, slight variation
        // 0.01 - 0.1 | Moderate variation, likely different versions/settings
        // 0.1 - 1.0 | Large differences, likely different models
        float finalScore = (totalWeight > 0.0f) ? (totalWeightedDist / totalWeight) : 0.0f;
        std::cout << "Final weighted distance score: " << finalScore << "\n";

        // Final score interpretation
        // average JSD score
        // 0.0 | Perfect match (identical predictions)
        // 0.0001 - 0.001 | Practically indistinguishable
        // 0.001 - 0.01 | Moderate variation, likely different versions/settings
        // 0.01 - 0.1 | Large differences, likely different models
        float jsdSum = 0.0f;
        for (const auto& jsd : jsdResults) {
            jsdSum += jsd;
        }
        float jsdAvg = jsdSum / jsdResults.size();
        std::cout << "Average JSD score: " << jsdAvg << "\n";

    }
    std::cout << '\n';

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
