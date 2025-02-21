// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

#include <ac/local/Lib.hpp>
#include <ac/local/IoCtx.hpp>
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

    ac::frameio::BlockingIoCtx blockingCtx;
    ac::local::IoCtx io;
    auto& llamaProvider = ac::local::Lib::getProvider("llama.cpp");
    ac::schema::BlockingIoHelper llama(io.connect(llamaProvider), blockingCtx);

    llama.expectState<schema::StateInitial>();

    llama.call<schema::StateInitial::OpLoadModel>({
        .ggufPath = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
    });
    llama.expectState<schema::StateModelLoaded>();

    llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "general"
    });
    llama.expectState<schema::StateInstance>();

    llama.call<schema::StateInstance::OpChatBegin>({
        .setup = "A chat between a human user and a helpful AI assistant.",
        .roleUser = "user",
        .roleAssistant = "assistant"
    });
    llama.expectState<schema::StateChat>();

    // Change state back from chat to instance
    llama.call<schema::StateChat::OpChatEnd>({});
    llama.expectState<schema::StateInstance>();

    llama.call<schema::StateInstance::OpChatBegin>({
        .setup = "A chat between a human user and a helpful AI assistant.",
        .roleUser = "user",
        .roleAssistant = "assistant"
    });
    llama.expectState<schema::StateChat>();

    while (true) {
        std::cout << "User: ";
        std::string user;
        std::getline(std::cin, user);
        if (user == "/quit") break;
        user = ' ' + user;
        llama.call<schema::StateChat::OpAddChatPrompt>({
            .prompt = user
        });

        auto res = llama.call<schema::StateChat::OpGetChatResponse>({});
        std::cout << "AI: " << res.response.value() << '\n';
    }

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
