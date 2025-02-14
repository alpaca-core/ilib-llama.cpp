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
        AC_TEST_DATA_LLAMA_DIR "/../../../tmp/Meta-Llama-3.1-70B-Instruct-Q5_K_S.gguf",
        AC_TEST_DATA_LLAMA_DIR "/../../../tmp/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf",
    };
    std::vector<std::string> names = {
        "Meta-Llama-3.1-70B",
        "Meta-Llama-3.1-8B"
    };

    // std::string prompt = "President George W.";
    std::string prompt = "In the 23th centuty";

    using ProbVector = std::vector<std::pair<ac::llama::Token, float>>;
    ProbVector results[modelGgufs.size()];

    results[0] = {
        {11, 15.7284},
        {279, 13.7213},
        {22706, 13.208},
        {1174, 12.4862},
        {264, 12.4639},
        {584, 11.6679}
    };

    results[1] = {
        { 11, 15.6762 },
        { 279, 13.6525 },
        { 22706, 13.2836 },
        { 264, 12.5076 },
        { 1174, 12.4008 },
        { 1070, 11.6473 }
    };

    // for (size_t i = 0; i < modelGgufs.size(); i++) {
    //     ac::llama::Model::Params modelParams = {
    //         .gpu = true,
    //     };

    //     auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(modelGgufs[i], modelLoadProgressCallback, modelParams);
    //     ac::llama::Model model(lmodel, modelParams);
    //     ac::llama::Instance instance(model, {
    //         .ctxSize = 1024
    //     });

    //     // start session
    //     auto& session = instance.startSession({});
    //     session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));
    //     auto a = session.getProbs(10);

    //     results[i] = a;
    // };

    for (size_t i = 1; i < modelGgufs.size(); i++) {
        std::cerr << "Model: " << modelGgufs[i - 1] << " vs " << modelGgufs[i] << std::endl;
        auto& a = results[i - 1];
        auto& b = results[i];

        ac::llama::LogitComparer c;
        const auto size = std::min(a.size(), b.size());
        auto res = c.compare(a, b, size);

        std::cout << "The models are " << (res ? "equal" : "different") << std::endl;
    }


    {
//         std::cout << "Prompt: " << prompt << std::endl;
//         for (size_t i = 0; i < modelGgufs.size(); i++) {
//             std::cerr << "Model: " << modelGgufs[i] << std::endl;
//             auto& a = results[i].first;

// #if 0 // GPU
//             {
//                 std::ofstream out(names[i] + "-gpu"+ ".logits", std::ios::binary);
//                 size_t s = a.size();
//                 out.write((char*)&s, sizeof(s));
//                 out.write((char*)a.data(), a.size() * sizeof(a[0]));
//                 out.close();
//             }
// #else
//             ProbVector b;
//             {
//                 std::ifstream out(names[i] + "-gpu"+ ".logits", std::ios::binary);
//                 size_t s;
//                 out.read((char*)&s, sizeof(s));
//                 b.resize(s);
//                 out.read((char*)b.data(), b.size() * sizeof(b[0]));
//                 out.close();
//             }

//             std::cout << "\t\t\tCPU vs GPU" << std::endl;

//             {
//                 float errSum = 0.0;
//                 for (size_t j = 0; j < a.size(); j++) {
//                     auto err = std::pow(a[j].second - b[j].second, 2.0);
//                     errSum += err;
//                 }

//                 float sqErr = std::sqrt(errSum);
//                 std::cout << names[i] << " err: " << errSum << ", sqErr: " << sqErr << std::endl;
//             }

//             for (size_t j = 0; j < a.size(); j++) {
//                 auto err = std::abs(a[j].second - b[j].second);
//                 std::cout << "[" << i << "]" << (err < 0.001 ? " OK" : " MISMATCH") << " (err < 0.001)."
//                             <<" t: [" << a[j].first  << "] vs [" << b[j].first << "],"
//                             <<" p: [" << a[j].second << "] vs [" << b[j].second << "]"
//                             << std::endl;
//             }
// #endif
//         }
    }

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
