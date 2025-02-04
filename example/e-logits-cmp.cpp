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
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/../../.." "/Thoth-Llama3.2-3B-IQ4_NL.gguf";
    std::string modelGguf2 = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    // std::string modelGguf2 = AC_TEST_DATA_LLAMA_DIR "/../../.." "/DeepSeek-R1-Distill-Qwen-7B-IQ4_XS.gguf";

    ac::llama::Model::Params modelParams;
    auto modelLoadProgressCallback = [](float progress) {
        const int barWidth = 50;
        static float currProgress = 0;
        auto delta = int(progress * barWidth) - int(currProgress * barWidth);
        for (int i = 0; i < delta; i++) {
            std::cout.put('=');
        }
        currProgress = progress;
        if (progress == 1.f) {
            std::cout << '\n';
        }
        return true;
    };
    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(modelGguf, modelLoadProgressCallback, modelParams);
    auto lmodel2 = ac::llama::ModelRegistry::getInstance().loadModel(modelGguf2, modelLoadProgressCallback, modelParams);
    ac::llama::Model model(lmodel, modelParams);
    ac::llama::Model modelDeep(lmodel2, modelParams);


    // create inference instance
    ac::llama::Instance instance(model, {});
    ac::llama::Instance instanceDeep(modelDeep, {});

    // To add control vector uncomment the following lines
    // ac::llama::ControlVector ctrlVector(model, {{ctrlVectorGguf, 2.f}});
    // instance.addControlVector(ctrlVector);

    std::string prompt = "In the 23th century the world was ";

    // start session
    auto& session = instance.startSession({});
    session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));
    auto a = session.getLogits(10);

    // 117m
    auto& sessionDeep = instanceDeep.startSession({});
    sessionDeep.setInitialPrompt(modelDeep.vocab().tokenize(prompt, true, true));
    auto b = sessionDeep.getLogits(10);

    std::string files[] = {
        "Thoth-Llama3.2-3B-IQ4_NL-logits.bin",
        "gpt2-117m-q6_k-logits.bin",
    };

    std::vector<float> logits[] = {a, b};

    // for (size_t i = 0; i < 2; i++) {
    //     std::ofstream f(files[i], std::ios::binary);
    //     size_t size = logits[i].size();
    //     f.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    //     f.write(reinterpret_cast<const char*>(logits[i].data()), logits[i].size() * sizeof(float));
    //     f.close();
    // }

    std::vector<float> gpuData[2];

    for (size_t i = 0; i < 2; i++) {
        std::ifstream f(files[i], std::ios::binary);

        size_t size = 0;
        f.read((char*)&size, sizeof(size_t));

        gpuData[i].resize(size);
        f.read((char*)gpuData[i].data(), size * sizeof(float));

        f.close();

        for (size_t j = 0; j < gpuData[i].size(); j++) {
            if (std::abs(logits[i][j] - gpuData[i][j]) > 0.00001) {
                std::cerr << "[" << i << "] Logits mismatch at index " << j << ". "<< logits[i][j] << " vs " << gpuData[i][j] <<std::endl;
            }
        }

    }




    // if (a.size() != b.size()) {
    //     std::cerr << "Logits size mismatch" << std::endl;
    //     return 1;
    // }

    for (size_t i = 0; i < a.size(); i++)
    {
        std::cout << a[i] << " " << b[i] << std::endl;
    }

    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > 0.00001) {
            std::cerr << "Logits mismatch at index " << i << std::endl;
            return 1;
        }
    }

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
