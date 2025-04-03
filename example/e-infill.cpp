// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// Code completion Example of using alpaca-core's llama inference

// llama
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Session.hpp>
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
    // download better model for good code completion results such as
    // https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/tree/main
    // std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/../../../models/qwen2.5-coder-3b-instruct-q8_0.gguf";
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";

    ac::local::ResourceManager rm;
    ac::llama::ResourceCache cache(rm);
    auto model = cache.getModel({.gguf = modelGguf, .params = {}});

    // create inference instance
    ac::llama::Instance instance(*model, {});

    // start session
    auto& session = instance.startSession({});
    session.setInitialPrompt({});

    std::string input_prefix = "def helloworld():\n    print(\"hell";
    std::string input_suffix = "\n    print(\"goodbye world\")\n";
    std::cout << "<prefix>\n" << input_prefix << "\n</prefix> +\n <place_to_fill> + \n" << "<postfix>\n" << input_suffix << "\n</postfix>\n";

    session.pushPrompt(
        model->vocab().tokenize(input_prefix, true, true),
        model->vocab().tokenize(input_suffix, true, true));

    std::cout << "Final result: \n" << input_prefix;

    // generate and print 100 tokens
    for (int i = 0; i < 100; ++i) {
        auto token = session.getToken();
        if (token == ac::llama::Token_Invalid) {
            // no more tokens
            break;
        }

        auto str = model->vocab().tokenToString(token);
        std::cout << str;
    }
    std::cout << input_suffix << "\n";

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
