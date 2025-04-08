// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// trivial example of using alpaca-core's llama embedding API

// llama
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/InstanceEmbedding.hpp>
#include <ac/llama/ResourceCache.hpp>

// logging
#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/ColorSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    // load model
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/bge-small-en-v1.5-f16.gguf";
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

    ac::local::ResourceManager rm;
    ac::llama::ResourceCache cache(rm);
    auto model = cache.getModel({.gguf = modelGguf, .params = {}}, modelLoadProgressCallback);

    // create inference instance
    ac::llama::InstanceEmbedding instance(*model, {});

    std::string prompt = "The main character in the story loved to eat pineapples.";
    std::vector<ac::llama::Token> tokens = model->vocab().tokenize(prompt, true, true);

    auto embeddings = instance.getEmbeddingVector(tokens);

    std::cout << "Embedding vector for prompt(" << prompt<< "): ";
    for (uint64_t i = 0; i < embeddings.size(); i++) {
        std::cout << embeddings[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
