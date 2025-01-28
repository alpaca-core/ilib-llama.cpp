// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local/Lib.hpp>
#include <ac/frameio/local/LocalIoRunner.hpp>
#include <ac/frameio/local/BlockingIo.hpp>

#include <ac/jalog/Instance.hpp>
#include <ac/jalog/sinks/DefaultSink.hpp>

#include <iostream>

#include "ac-test-data-llama-dir.h"
#include "aclp-llama-info.h"

int main() try {
    ac::jalog::Instance jl;
    jl.setup().add<ac::jalog::sinks::DefaultSink>();

    ac::local::Lib::loadPlugin(ACLP_llama_PLUGIN_FILE);
    auto llamaHandler = ac::local::Lib::createSessionHandler("llama.cpp");
    ac::frameio::LocalIoRunner runner;

    auto io = runner.connectBlocking(llamaHandler);

    io.push({ "load", {{"gguf",  AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"}} });
    io.push({ "create", {} });
    io.push({ "run", {{"prompt", "The first person to"}, {"max_tokens", 20}} });

    auto res = io.poll();
    auto resp = ac::Dict_optValueAt(res.frame.data, "result", std::string());
    std::cout << resp << '\n';

    io.push({ "run", {{"prompt", "The first person to"}, {"max_tokens", 20}} });
    res = io.poll();

    io.push({ "begin-chat", {} });
    res = io.poll();

    io.push({ "end-chat", {} });
    res = io.poll();

    io.push({ "run", {{"prompt", "The first person to"}, {"max_tokens", 20}} });
    res = io.poll();

    io.push({ "begin-chat", {} });

    // auto instance = model->createInstance("general", {});

    // const std::string prompt = "The first person to";

    // std::vector<std::string> antiprompts;
    // antiprompts.push_back("user:"); // change it to "name" to break the token generation with the default input

    // std::cout << "Prompt: " << prompt << "\n";
    // for (size_t i = 0; i < antiprompts.size(); i++) {
    //     std::cout << "Antiprompt "<<"[" << i << "]" <<": \"" << antiprompts[i] << "\"\n";
    // }
    // std::cout << "Generation: " << "<prompt>" << prompt << "</prompt> ";

    // auto result = instance->runOp("run", {{"prompt", prompt}, {"max_tokens", 20}, {"antiprompts", antiprompts}}, {});

    // std::cout << result.at("result").get<std::string_view>() << '\n';

    return 0;
}
catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
