// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local/Lib.hpp>

#include <ac/frameio/local/LocalIoRunner.hpp>
#include <ac/frameio/local/BlockingIo.hpp>

#include <ac/schema/BlockingIoHelper.hpp>
#include <ac/schema/FrameHelpers.hpp>

#include <ac/schema/LlamaCpp.hpp>

#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/DefaultSink.hpp>

#include <iostream>

#include "ac-test-data-llama-dir.h"
#include "aclp-llama-info.h"

namespace schema = ac::schema::llama;

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::DefaultSink>();

    ac::local::Lib::loadPlugin(ACLP_llama_PLUGIN_FILE);

    ac::frameio::LocalIoRunner io;
    ac::schema::BlockingIoHelper llama(io.connectBlocking(ac::local::Lib::createSessionHandler("llama.cpp")));

    llama.expectState<schema::StateInitial>();
    llama.call<schema::StateInitial::OpLoadModel>({
        .ggufPath = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
    });

    llama.expectState<schema::StateModelLoaded>();
    llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "general"
    });

    llama.expectState<schema::StateInstance>();

    constexpr std::string prompt = "The first person to";

    std::vector<std::string> antiprompts;
    antiprompts.push_back("user:"); // change it to "name" to break the token generation with the default input

    auto result = llama.call<schema::StateInstance::OpRun>({
        .prompt = prompt,
        .maxTokens = 20,
        .antiprompts = antiprompts
    });

    std::cout << "Prompt: " << prompt << "\n";
    for (size_t i = 0; i < antiprompts.size(); i++) {
        std::cout << "Antiprompt "<<"[" << i << "]" <<": \"" << antiprompts[i] << "\"\n";
    }
    std::cout   << "Generation: "
                << "<prompt>" << prompt << "</prompt> "
                << result.result.value() << '\n';

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
