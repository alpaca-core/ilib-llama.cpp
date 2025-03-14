// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

#include <ac/local/Lib.hpp>
#include <ac/local/DefaultBackend.hpp>
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

    ac::local::DefaultBackend backend;
    ac::schema::BlockingIoHelper llama(backend.connect("llama.cpp"));

    llama.expectState<schema::StateInitial>();

    llama.call<schema::StateInitial::OpLoadModel>({
        .ggufPath = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
    });
    llama.expectState<schema::StateModelLoaded>();

    llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "general"
    });
    llama.expectState<schema::StateInstance>();

    const std::string roleUser = "user";
    const std::string roleAssistant = "assistant";
    llama.call<schema::StateInstance::OpChatBegin>({
        .setup = "A chat between a human user and a helpful AI assistant.",
        .roleUser = roleUser,
        .roleAssistant = roleAssistant
    });
    llama.expectState<schema::StateChat>();

    while (true) {
        std::cout << roleUser <<": ";
        std::string user;
        while (user.empty()) {
            std::getline(std::cin, user);
        }
        if (user == "/quit") break;
        user = ' ' + user;
        llama.call<schema::StateChat::OpAddChatPrompt>({
            .prompt = user
        });

        constexpr bool shouldStream = true;
        auto res = llama.call<schema::StateChat::OpGetChatResponse>({
            .stream = shouldStream
        });

        if (shouldStream) {
            llama.expectState<schema::StateStreaming>();
            std::cout << roleAssistant << ": ";
            for(auto t : llama.runStream<schema::StateStreaming::StreamToken, schema::StateChat>()) {
                std::cout << t << std::flush;
            };
            std::cout << '\n';
        } else {
            std::cout << roleAssistant << ": " << res.response.value() << '\n';
        }

    }

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
