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

    const std::string prompt = "The first person to";

    std::vector<std::string> antiprompts;
    antiprompts.push_back("user:"); // change it to "name" to break the token generation with the default input

    constexpr bool shouldStream = true;
    auto result = llama.call<schema::StateInstance::OpRun>({
        .prompt = prompt,
        .antiprompts = antiprompts,
        .maxTokens = 20,
        .stream = shouldStream
    });

    std::cout << "Prompt: " << prompt << "\n";
    for (size_t i = 0; i < antiprompts.size(); i++) {
        std::cout << "Antiprompt "<<"[" << i << "]" <<": \"" << antiprompts[i] << "\"\n";
    }

    std::cout   << "Generation: <prompt>" << prompt << "</prompt> ";

    if (shouldStream) {
        llama.runStream<schema::StateStreaming, schema::StateStreaming::StreamToken>([&](const std::string& token) {
            std::cout << token << std::flush;
        });
    } else {
        std::cout << result.result.value();
    }

    std::cout << std::endl;

    auto result2 = llama.call<schema::StateInstance::OpGetTokenData>({});

    auto result3 = llama.call<schema::StateInstance::OpCompareTokenData>({
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
