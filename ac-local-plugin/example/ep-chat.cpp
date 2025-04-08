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

    for (auto x : llama.stream<schema::StateLlama::OpLoadModel>({
        // .ggufPath = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
        .ggufPath = AC_TEST_DATA_LLAMA_DIR "/../../../tmp/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
    })) {
        std::cout << "Model loaded: " << x.tag.value() << " " << x.progress.value() << '\n';
    }

    const std::string roleUser = "user";
    const std::string roleAssistant = "assistant";

    sid = llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "chat",
        .setup = "A chat between a human user and a helpful AI assistant.",
        // .roleUser = roleUser,
        // .roleAssistant = roleAssistant
    });
    std::cout << "Instance started: " << sid << '\n';

    std::vector<schema::Message> initMessages = {
        {roleUser, "I need assistance for API design"},
        {roleAssistant, "What aspect of API design are you looking for help with? Do you have a specific problem or question in mind?"},
        {roleUser, "It's a C++ implementation of a class"},
    };

    llama.call<schema::StateChatInstance::OpSendMessages>({
        .messages = initMessages
    });

    std::vector<schema::Message> messages;

    while (true) {
        std::cout << roleUser <<": ";
        std::string user;
        while (user.empty()) {
            std::getline(std::cin, user);
        }
        if (user == "/quit") break;
        user = ' ' + user;
        messages.push_back({roleUser, user});

        llama.call<schema::StateChatInstance::OpAddChatPrompt>({
            .prompt = user
        });

        std::string text;
        std::cout << roleAssistant << ": ";
        constexpr bool streamChat = false;
        if (streamChat) {
            for(auto t: llama.stream<schema::StateChatInstance::OpStreamChatResponse>({})) {
                text += t;
                std::cout << t << std::flush;
            }
        } else {
            auto res = llama.call<schema::StateChatInstance::OpGetChatResponse>({});
            text += res.response.value();
            std::cout << res.response.value() << std::flush;
        }
        messages.push_back({roleUser, text});
        std::cout << "\n";
    }

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
