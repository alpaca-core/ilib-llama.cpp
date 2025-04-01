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
    ac::schema::BlockingIoHelper llama(backend.connect("llama.cpp", {}));

    auto sid = llama.poll<ac::schema::StateChange>();
    std::cout << "Initial state: " << sid << '\n';

    auto load = llama.stream<schema::StateLlama::OpLoadModel>({
        .ggufPath = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
    });
    sid = load.rval();
    std::cout << "Model loaded: " << sid << '\n';

    const std::string roleUser = "user";
    const std::string roleAssistant = "assistant";

    sid = llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "chat",
        .setup = "A chat between a human user and a helpful AI assistant.",
        .roleUser = roleUser,
        .roleAssistant = roleAssistant
    });
    std::cout << "Instance started: " << sid << '\n';

    while (true) {
        std::cout << roleUser <<": ";
        std::string user;
        while (user.empty()) {
            std::getline(std::cin, user);
        }
        if (user == "/quit") break;
        user = ' ' + user;
        llama.call<schema::StateChatInstance::OpAddChatPrompt>({
            .prompt = user
        });

        auto stream = llama.stream<schema::StateChatInstance::OpStreamChatResponse>({});

        std::cout << roleAssistant << ": ";
        for(auto t : stream) {
            std::cout << t << std::flush;
        };
        std::cout << '\n';
    }

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
