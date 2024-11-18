// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local/Model.hpp>
#include <ac/local/Instance.hpp>
#include <ac/local/ModelLoaderRegistry.hpp>
#include <ac/local/Lib.hpp>

#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/DefaultSink.hpp>

#include <iostream>

#include "ac-test-data-llama-dir.h"
#include "aclp-llama-info.h"

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::DefaultSink>();

    ac::local::Lib::loadPlugin(ACLP_llama_PLUGIN_FILE);

    auto model = ac::local::Lib::modelLoaderRegistry().createModel(
        {
            .inferenceType = "llama",
            .assets = {
                {.path = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"}
            }
        },
        {},
        [](std::string_view tag, float) {
            if (tag.empty()) {
                std::cout.put('*');
            }
            else {
                std::cout.put(tag[0]);
            }
            return true;
        }
    );


    auto instance = model->createInstance("general", {});

    const std::string prompt = "The first person to";

    std::vector<std::string> antiprompts;
    antiprompts.push_back("user:"); // change it to "name" to break the token generation with the default input

    std::cout << "Prompt: " << prompt << "\n";
    for (size_t i = 0; i < antiprompts.size(); i++) {
        std::cout << "Antiprompt "<<"[" << i << "]" <<": \"" << antiprompts[i] << "\"\n";
    }
    std::cout << "Generation: " << "<prompt>" << prompt << "</prompt> ";

    auto result = instance->runOp("run", {{"prompt", prompt}, {"max_tokens", 20}, {"antiprompts", antiprompts}}, {});

    std::cout << result.at("result").get<std::string_view>() << '\n';

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
