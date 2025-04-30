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
#include <fstream>
#include <string>
#include <filesystem>

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

    GenerationResult generate(std::string_view prompt, uint32_t maxTokens) {
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


std::vector<Model::GenerationResult> modelGeneration(Model& m1, Model& m2, std::string_view prompt, uint32_t maxTokens) {
    auto res = m1.generate(prompt, maxTokens);

    auto genPromptTokens = m2.tokenize(res.initalPrompt);

    Model::GenerationResult res2;
    for (size_t i = 0; i < res.steps.size(); i++) {
        auto& step  = res.steps[i];
        if (i > 0) {
            if (m2.tokenExists(step.token)) {
                genPromptTokens.push_back(step.token);
            }
            else {
                std::cout << "Token not found in model 2: " << step.tokenStr << "\n";
                throw std::runtime_error("Token not found in model 2");
            }
        }

        if (i == 0) {
            res2 = m2.generate(genPromptTokens, 0);
        } else {
            Model::GenerationResult tempRes;
            std::vector<ac::llama::Token> token{step.token};
            tempRes = m2.generate(token, 0);
            res2.steps.push_back(tempRes.steps[0]);
        }
    }

    res2.result = res.result;

    return {res, res2};
}

// function to serialize the generation result in a file, so I can read it later
void serialize(std::string_view filename, std::string_view gguf, Model::GenerationResult& res) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return;
    }

    size_t ggufSize = gguf.size();
    f.write(reinterpret_cast<const char*>(&ggufSize), sizeof(ggufSize));
    f.write(gguf.data(), gguf.size());

    size_t initialPromptSize = res.initalPrompt.size();
    f.write(reinterpret_cast<const char*>(&initialPromptSize), sizeof(initialPromptSize));
    f.write(res.initalPrompt.c_str(), res.initalPrompt.size());

    size_t resultSize = res.result.size();
    f.write(reinterpret_cast<const char*>(&resultSize), sizeof(resultSize));
    f.write(res.result.c_str(), res.result.size());

    size_t stepsCount = res.steps.size();
    f.write(reinterpret_cast<const char*>(&stepsCount), sizeof(stepsCount));
    for (const auto& step : res.steps) {
        size_t tokenStrSize = step.tokenStr.size();
        f.write(reinterpret_cast<const char*>(&tokenStrSize), sizeof(tokenStrSize));

        f.write(step.tokenStr.c_str(), step.tokenStr.size());
        f.write(reinterpret_cast<const char*>(&step.token), sizeof(step.token));

        size_t tokenCount = step.data.size();
        f.write(reinterpret_cast<const char*>(&tokenCount), sizeof(tokenCount));
        f.write(reinterpret_cast<const char*>(step.data.data()), sizeof(ac::llama::TokenData) * tokenCount);
    }
}

Model::GenerationResult deserialize(std::string_view filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Error opening file for reading: " << filename << "\n";
        return {};
    }

    Model::GenerationResult res;

    size_t ggufSize = 0;
    f.read(reinterpret_cast<char*>(&ggufSize), sizeof(ggufSize));

    std::string gguf;
    gguf.resize(ggufSize);
    f.read(gguf.data(), ggufSize);

    size_t initialPromptSize = 0;
    f.read(reinterpret_cast<char*>(&initialPromptSize), sizeof(initialPromptSize));

    res.initalPrompt.resize(initialPromptSize);
    f.read(res.initalPrompt.data(), initialPromptSize);

    size_t resultSize;
    f.read(reinterpret_cast<char*>(&resultSize), sizeof(resultSize));

    res.result.resize(resultSize);
    f.read(res.result.data(), resultSize);

    size_t stepsCount;
    f.read(reinterpret_cast<char*>(&stepsCount), sizeof(stepsCount));
    res.steps.reserve(stepsCount);
    for (size_t i = 0; i < stepsCount; ++i) {
        GenerationStepData step;

        size_t tokenStrSize;
        f.read(reinterpret_cast<char*>(&tokenStrSize), sizeof(tokenStrSize));

        step.tokenStr.resize(tokenStrSize);
        f.read(step.tokenStr.data(), tokenStrSize);

        f.read(reinterpret_cast<char*>(&step.token), sizeof(step.token));

        size_t tokenCount;
        f.read(reinterpret_cast<char*>(&tokenCount), sizeof(tokenCount));

        step.data.resize(tokenCount);
        f.read(reinterpret_cast<char*>(step.data.data()), sizeof(ac::llama::TokenData) * tokenCount);

        res.steps.push_back(step);
    }

    return res;
}

void runCompare(Model::GenerationResult& r1, Model::GenerationResult& r2) {
    std::vector<float> jsdResults;
    std::vector<float> similarityResults;
    float totalWeightedDist = 0.0f;
    float totalWeight = 0.0f;

    for (size_t i = 0; i < r1.steps.size(); i++) {
        auto& step1 = r1.steps[i];
        auto& step2 = r2.steps[i];

        // Calculate distance
        float dist = ac::llama::LogitComparer::cosineDistance(step1.data, step2.data);

        // Calculate weight based on normalized entropy
        float weight = normalizedEntropy(step1.data);
        totalWeightedDist += weight * dist;
        totalWeight += weight;

        // Calculate JSD
        float jsd = ac::llama::LogitComparer::JSD(step1.data, step2.data);
        jsdResults.push_back(jsd);

        // Calculate similarity
        float similarity = ac::llama::LogitComparer::logitSimilarity(step1.data, step2.data);
        similarityResults.push_back(similarity);

        std::cout << "Token: " << step1.tokenStr
                << ", Weight: " << weight
                << ", JSD: " << jsd
                << ", Similarity: " << similarity
                << ", Distance: " << dist
                << "\n";
    }


    {
        // Final step: Normalize

        // Score range | Interpretation
        // 0.0 | Perfect match (identical predictions)
        // 0.0001 - 0.001 | Practically indistinguishable
        // 0.001 - 0.01 | Very close, slight variation
        // 0.01 - 0.1 | Moderate variation, likely different versions/settings
        // 0.1 - 1.0 | Large differences, likely different models
        float finalScore = (totalWeight > 0.0f) ? (totalWeightedDist / totalWeight) : 0.0f;
        std::cout << "Final weighted distance score: " << finalScore << "\n";
    }

    {
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

    {
        float similaritySum = 0.0f;
        for (const auto& similarity : similarityResults) {
            similaritySum += similarity;
        }
        float similarityAvg = similaritySum / similarityResults.size();
        std::cout << "Average similarity score: " << similarityAvg << "\n";
    }
}


int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    std::vector<GenerationStepData> genSteps;

    // load model
    std::string tmpFolder = AC_TEST_DATA_LLAMA_DIR "/../../../tmp/";
    std::string modelGguf = "Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf";
    // std::string modelGguf = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";
    // std::string modelGguf = "BgGPT-Gemma-2-2B-IT-v1.0.Q8_0.gguf";
    // std::string modelGguf = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";
    // std::string modelGguf2 = "Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf";
    std::string modelGguf2 = "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf";

    // std::string prompt = "The first person to ";
    std::string prompt = "Explain quantum physics in simple terms.";
    std::cout << "Prompt: " << prompt << "\n";

    std::string res1fn = "gen-res_" + modelGguf + "_" + prompt + ".bin";
    std::string res2fn = "gen-res_" + modelGguf2 + "_" + prompt + ".bin";
    bool shouldRunGenerate = !(std::filesystem::exists(res1fn) && std::filesystem::exists(res2fn));

    Model::GenerationResult r1;
    Model::GenerationResult r2;
    if (shouldRunGenerate) {
        Model m1(tmpFolder + modelGguf, {});
        Model m2(tmpFolder + modelGguf2, {});

        auto genRes = modelGeneration(m1, m2, prompt, 100);
        r1 = std::move(genRes[0]);
        r2 = std::move(genRes[1]);
        serialize(res1fn, modelGguf, r1);
        serialize(res2fn, modelGguf2, r2);
    } else {
        r1 = deserialize(res1fn);
        r2 = deserialize(res2fn);
    }

    std::string result = prompt;

    std::cout << "Models to compare:\n" << modelGguf << "\n" << modelGguf2 << "\n";
    std::cout << "Comparing...\n";

    runCompare(r1, r2);

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
