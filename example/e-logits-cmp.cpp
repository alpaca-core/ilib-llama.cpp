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

// logging
#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/ColorSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>
#include <fstream>

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    // load model
    std::vector<std::string> modelGgufs = {
        AC_TEST_DATA_LLAMA_DIR "/../../.." "/Thoth-Llama3.2-3B-IQ4_NL.gguf",
        AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
    };
    std::vector<std::string> names = {
        "llam3.2-3b-iq4_nl",
        "gpt2-117m-q6_k"
    };

    // std::string prompt = "President George W.";
    std::string prompt = "In the 23th centuty";

    using ProbVector = std::vector<std::pair<ac::llama::Token, float>>;
    std::pair<ProbVector, ProbVector> results[modelGgufs.size()];

    for (size_t i = 0; i < modelGgufs.size(); i++)
    {
        ac::llama::Model::Params modelParams = {
            .gpu = true,
        };

        // ac::llama::Model::Params modelParamsCpu = {
        //     .gpu = false,
        // };

        auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(modelGgufs[i], nullptr, modelParams);
        ac::llama::Model model(lmodel, modelParams);
        ac::llama::Instance instance(model, {});

        // auto lmodelCpu = ac::llama::ModelRegistry::getInstance().loadModel(modelGgufs[i], cb, modelParamsCpu);
        // ac::llama::Model modelCpu(lmodel, modelParamsCpu);
        // ac::llama::Instance instanceCpu(modelCpu, {});

        // start session
        auto& session = instance.startSession({});
        session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));
        auto a = session.getProbs(10);

        // 117m
        // auto& sessionCpu = instanceCpu.startSession({});
        // sessionCpu.setInitialPrompt(modelCpu.vocab().tokenize(prompt, true, true));
        // auto b = sessionCpu.getProbs(10);

        // if (a.size() != b.size()) {
        //     std::cerr << "Logits size mismatch for " << modelGgufs[i] << std::endl;
        //     continue;
        // }

        results[i].first = a;
        // results[i].second = b;
    };

    std::cout << "Prompt: " << prompt << std::endl;
    for (size_t i = 0; i < modelGgufs.size(); i++) {
        std::cerr << "Model: " << modelGgufs[i] << std::endl;
        auto& a = results[i].first;

#if 0 // GPU
        {
            std::ofstream out(names[i] + "-gpu"+ ".logits", std::ios::binary);
            size_t s = a.size();
            out.write((char*)&s, sizeof(s));
            out.write((char*)a.data(), a.size() * sizeof(a[0]));
            out.close();
        }
#else
        ProbVector b;
        {
            std::ifstream out(names[i] + "-gpu"+ ".logits", std::ios::binary);
            size_t s;
            out.read((char*)&s, sizeof(s));
            b.resize(s);
            out.read((char*)b.data(), b.size() * sizeof(b[0]));
            out.close();
        }

        std::cout << "\t\t\tCPU vs GPU" << std::endl;

        {
            float errSum = 0.0;
            for (size_t j = 0; j < a.size(); j++) {
                auto err = std::pow(a[j].second - b[j].second, 2.0);
                errSum += err;
            }

            float sqErr = std::sqrt(errSum);
            std::cout << names[i] << " err: " << errSum << ", sqErr: " << sqErr << std::endl;
        }

        for (size_t j = 0; j < a.size(); j++) {
            auto err = std::abs(a[j].second - b[j].second);
            std::cout << "[" << i << "]" << (err < 0.001 ? " OK" : " MISMATCH") << " (err < 0.001)."
                        <<" t: [" << a[j].first  << "] vs [" << b[j].first << "],"
                        <<" p: [" << a[j].second << "] vs [" << b[j].second << "]"
                        << std::endl;
        }
#endif
    }

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
