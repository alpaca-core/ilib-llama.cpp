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
#include <ac/llama/LogitComparer.hpp>

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

    // initialize the library
    ac::llama::initLibrary();

    // load model
    std::vector<std::string> modelGgufs = {
        //AC_TEST_DATA_LLAMA_DIR "/../../../tmp/Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf",
        AC_TEST_DATA_LLAMA_DIR "/../../../tmp/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf",
    };
    //std::vector<std::string> names = {
    //    //"Meta-Llama-3.1-70B",
    //    "Meta-Llama-3.1-8B"
    //};

    // std::string prompt = "President George W.";
    std::string prompt = "In the 23th centuty";


    std::vector<std::string> names = {
        "Meta-Llama-3.1-8B-cuda",
        "Meta-Llama-3.1-8B-cpu-amd",
        "Meta-Llama-3.1-8B-metal",
        "Meta-Llama-3.1-8B-cpu-mac",
        "Meta-Llama-3.1-70B-metal",
        "Meta-Llama-3.1-70B-cpu-mac",
    };

    std::vector<ac::llama::TokenDataVector> results;
    results.resize(names.size());

    // 8b cuda
    results[0] = {
        {11, 16.9658},
        {22706, 13.6525},
        {279, 13.2836},
        {12966, 12.5076},
    };

    // 8b cpu amd
    results[1] = {
        {11, 17.0296},
        {22706, 13.5018},
        {279, 13.4278},
        {264, 12.7805},
        {12966, 12.7211},
    };

    // 8b metal
    results[2] = {
        {11, 16.9334},
        {22706, 13.4646},
        {279, 13.3156},
        {12966, 12.6787},
    };

    // 8b cpu mac
    results[3] = {
        { 11, 17.0128 },
        { 279, 13.4753 },
        { 22706, 13.452 },
        { 264, 12.6863 },
    };

    // 70b metal
    results[4] = {
        {11, 15.7284},
        {279, 13.7213},
        {22706, 13.208},
        {1174, 12.4862},
        {264, 12.4639},
        {584, 11.6679}
    };

    // 70b cpu mac
    results[5] = {
        { 11, 15.6762 },
        { 279, 13.6525 },
        { 22706, 13.2836 },
        { 264, 12.5076 },
        { 1174, 12.4008 },
        { 1070, 11.6473 }
    };

     //for (size_t i = 0; i < modelGgufs.size(); i++) {
     //    ac::llama::Model::Params modelParams = {
     //        .gpu = true,
     //    };

     //    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(modelGgufs[i], modelLoadProgressCallback, modelParams);
     //    ac::llama::Model model(lmodel, modelParams);
     //    ac::llama::Instance instance(model, {
     //        .ctxSize = 1024
     //    });

     //    // start session
     //    auto& session = instance.startSession({});
     //    session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));
     //    auto a = session.getProbs(10);

     //    results[i] = a;
     //};

    for (size_t i = 0; i < results.size(); i++) {
        auto& a = results[i];

        for (size_t j = i + 1; j < results.size(); j++)
        {
            auto& b = results[j];

            std::cerr << names[i] << " vs " << names[j] << std::endl;

            ac::llama::LogitComparer c;
            const auto size = std::min(a.size(), b.size());
            auto res = c.compare(a, b, size);

            bool shouldBeSame = names[i].find("8B") == names[j].find("8B");

            std::cout << "The models should be "<< (shouldBeSame ? "EQUAL" : "DIFFERENT") << std::endl;
            std::cout << "The models are " << (res ? "EQUAL" : "DIFFERENT") << std::endl;

            std::cout << "\n===========================================================\n\n";
        }
    }

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
