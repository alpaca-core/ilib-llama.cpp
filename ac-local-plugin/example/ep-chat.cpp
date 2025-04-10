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
        .ggufPath = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"
    })) {
        std::cout << "Model loaded: " << x.tag.value() << " " << x.progress.value() << '\n';
    }

    const std::string roleUser = "user";
    const std::string roleAssistant = "assistant";
    const std::string chatTemplate =
                        "{% for message in messages %}"
                        "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>' + '\\n' }}"
                        "{% endfor %}"
                        "{% if add_generation_prompt %}"
                        "{{ '<|' + assistant_role + '|>\\n' }}"
                        "{% endif %}";

    constexpr bool useChatTemplate = false;
    sid = llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "chat",
        .setup = "A chat between a human user and a helpful AI assistant.",
        .chatTemplate = useChatTemplate ? chatTemplate : "",
        .roleUser = roleUser,
        .roleAssistant = roleAssistant,
    });
    std::cout << "Instance started: " << sid << '\n';

    constexpr bool addPreviousMessages = true;
    if (addPreviousMessages) {
        std::vector<schema::Message> msgs = {
            {roleUser, "Hey, I need help planning a surprise weekend getaway."},
            {roleAssistant, "Sure! Are you thinking of something outdoorsy, a relaxing spa weekend, or maybe a city adventure?"},
            {roleUser, "A quiet nature retreat would be perfect."},
            {roleAssistant, "Great choice. I can suggest a few scenic cabin locations and even help you build a checklist for the trip."}
        };

        llama.call<schema::StateChatInstance::OpAddChatMessages>({
            .messages = msgs
        });

        for (auto& m : msgs) {
            std::cout << m.role.value() << ": " << m.content.value() << '\n';
        }
    }

    while (true) {
        std::cout << roleUser <<": ";
        std::string user;
        while (user.empty()) {
            std::getline(std::cin, user);
        }
        if (user == "/quit") break;

        llama.call<schema::StateChatInstance::OpAddChatMessages>({
            .messages = std::vector<schema::Message>{
                { roleUser, user}
            }
        });

        std::cout << roleAssistant << ": ";
        constexpr bool streamChat = true;
        if (streamChat) {
            for(auto t: llama.stream<schema::StateChatInstance::OpStreamChatResponse>({})) {
                std::cout << t << std::flush;
            }
        } else {
            auto res = llama.call<schema::StateChatInstance::OpGetChatResponse>({});
            std::cout << res.response.value() << std::flush;
        }
        std::cout << "\n";
    }

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
