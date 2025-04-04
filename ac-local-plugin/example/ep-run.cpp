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

    sid = llama.call<schema::StateModelLoaded::OpStartInstance>({
        .instanceType = "general",
    });
    std::cout << "Instance started: " << sid << '\n';

    const std::string prompt = "The first person to";

    std::vector<std::string> antiprompts;
    antiprompts.push_back("user:"); // change it to "name" to break the token generation with the default input

    auto stream = llama.stream<schema::StateGeneralInstance::OpStream>({
        .prompt = prompt,
        .antiprompts = antiprompts,
        .maxTokens = 20
    });

    std::cout << "Prompt: " << prompt << "\n";
    for (size_t i = 0; i < antiprompts.size(); i++) {
        std::cout << "Antiprompt "<<"[" << i << "]" <<": \"" << antiprompts[i] << "\"\n";
    }

    std::cout << "Generation: <prompt>" << prompt << "</prompt> ";

    for(auto t : stream) {
        std::cout << t << std::flush;
    };

    std::cout << std::endl;

    auto result2 = llama.call<schema::StateGeneralInstance::OpGetTokenData>({});

    auto result3 = llama.call<schema::StateGeneralInstance::OpCompareTokenData>({
        .tokens1 = result2.tokens,
        .logits1 = result2.logits,
        .probs1 = result2.probs,
        .tokens2 = result2.tokens,
        .logits2 = result2.logits,
        .probs2 = result2.probs
    });

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
