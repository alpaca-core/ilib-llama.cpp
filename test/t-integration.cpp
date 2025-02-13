// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Session.hpp>
#include <ac/llama/ControlVector.hpp>

#include <doctest/doctest.h>

#include "ac-test-data-llama-dir.h"

struct GlobalFixture {
    GlobalFixture() {
        ac::llama::initLibrary();
    }
};

GlobalFixture globalFixture;

const char* Model_117m_q6_k = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";

TEST_CASE("vocab only") {
    ac::llama::Model::Params iParams = { .vocabOnly = true };
    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(Model_117m_q6_k, {}, iParams);
    ac::llama::Model model(lmodel, iParams);
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK(params.vocabOnly);

    CHECK(model.trainCtxLength() == 0); // no weights - no training context
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());

    // vocab works
    auto& vocab = model.vocab();
    CHECK(vocab.tokenToString(443) == " le");
    CHECK(vocab.tokenize("hello world", true, true) == std::vector<ac::llama::Token>{31373, 995});
}

TEST_CASE("inference") {
    ac::llama::Model::Params iParams = {};
    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(Model_117m_q6_k, {}, iParams);
    ac::llama::Model model(lmodel, iParams);
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model.trainCtxLength() == 1024);
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());

    // general inference
    {
        ac::llama::Instance inst(model, {});
        inst.warmup(); // should be safe

        std::vector<ac::llama::Token> tokens;

        // choose a very, very suggestive prompt and hope that all architectures will agree
        auto& s = inst.startSession({});
        tokens = model.vocab().tokenize("President George W.", true, true);
        s.setInitialPrompt(tokens);
        {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model.vocab().tokenToString(t);
                CHECK(text == " Bush");
        }

        SUBCASE("default sampler") {
            // add more very suggestive stuff
            tokens = model.vocab().tokenize(" sent troops to Cleveland which was hit by torrential", false, false);
            s.pushPrompt(tokens);
            {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model.vocab().tokenToString(t);
                CHECK(text.starts_with(" rain")); // could be rains
            }
        }

        SUBCASE("custom sampler") {
            ac::llama::Sampler::Params samplerParams = {};
            samplerParams.rngSeed = 1717;
            samplerParams.minP = 0.2f;
            samplerParams.topK = 100;
            samplerParams.topP = 0.2f;
            samplerParams.minKeep = 1000;
            samplerParams.temp = 10.0f;
            samplerParams.tempExp = 5.0f;
            samplerParams.samplerSequence = {
                ac::llama::Sampler::SamplingType::Min_P,
                ac::llama::Sampler::SamplingType::Temperature,
                ac::llama::Sampler::SamplingType::Top_K,
                ac::llama::Sampler::SamplingType::Top_P,
                };
            inst.resetSampler(samplerParams);

            // add more very suggestive stuff
            tokens = model.vocab().tokenize(" sent troops to Cleveland which was hit by torrential", false, false);
            s.pushPrompt(tokens);
            {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model.vocab().tokenToString(t);
                CHECK(text.starts_with(" down"));
            }
        }
    }
}

TEST_CASE("session") {
    ac::llama::Model::Params iParams = {};
    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(Model_117m_q6_k, {}, iParams);
    ac::llama::Model model(lmodel, iParams);
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model.trainCtxLength() == 1024);
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());
    ac::llama::Instance inst(model, {});
    inst.warmup(); // should be safe
    SUBCASE("no initalization") {
        auto& s = inst.startSession({});
        SUBCASE("getToken") {
            CHECK_THROWS_WITH(s.getToken(), "Session hasn't started yet");
        }

        SUBCASE("pushPrompt") {
            auto tokens = model.vocab().tokenize("President George W.", true, true);
            CHECK_THROWS_WITH(s.pushPrompt(tokens), "Session hasn't started yet");
        }

        SUBCASE("getState") {
            CHECK_THROWS_WITH(s.getState(), "Session hasn't started yet");
        }
    }

    SUBCASE("double initialization") {
        auto& s = inst.startSession({});
        auto tokens = model.vocab().tokenize("President George W.", true, true);
        s.setInitialPrompt(tokens);
        CHECK_THROWS_WITH(s.setState({}), "Session already started");
    }

    SUBCASE("generating phase") {
        auto& s = inst.startSession({});
        {
            auto tokens = model.vocab().tokenize("President George W.", true, true);
            s.setInitialPrompt(tokens);
        }
        {
            auto t = s.getToken();
            CHECK(model.vocab().tokenToString(t) == " Bush");
        }
        {
            auto tokens = model.vocab().tokenize(" usually goes to Washington to", true, true);
            s.pushPrompt(tokens);
            auto t = s.getToken();
            auto text = model.vocab().tokenToString(t);
            CHECK(text.starts_with(" meet")); // could be rains
        }
        {
            CHECK(s.getState().size() > 0);
        }
    }

    SUBCASE("single session") {
        auto& s = inst.startSession({});
        (void)s;
        CHECK_THROWS_WITH(inst.startSession({}), "Session is already started. Stop it to start a new one.");
    }
}

// commented out because it relies on specific calc
//TEST_CASE("session states") {
//    ac::llama::Model::Params iParams = {};
//    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(Model_117m_q6_k, {}, iParams);
//    ac::llama::Model model(lmodel, iParams);
//    CHECK(!!model.lmodel());
//
//    auto& params = model.params();
//    CHECK(params.gpu);
//    CHECK_FALSE(params.vocabOnly);
//
//    CHECK(model.trainCtxLength() == 1024);
//    CHECK_FALSE(model.shouldAddBosToken());
//    CHECK_FALSE(model.hasEncoder());
//    {
//        std::string ctrlVectorGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6-control_vector.gguf";
//
//        {
//            ac::llama::ControlVector ctrlVector(model, {{ctrlVectorGguf, -2.f}});
//            ac::llama::Instance inst(model, {});
//            inst.addControlVector(ctrlVector);
//            inst.warmup(); // should be safe
//            auto& s = inst.startSession({});
//            std::vector<ac::llama::Token> tokens = model.vocab().tokenize("My car's fuel consumption is", true, true);
//            s.setInitialPrompt(tokens);
//            std::string text;
//            for (int i = 0; i < 5; ++i) {
//                auto t = s.getToken();
//                REQUIRE(t != ac::llama::Token_Invalid);
//                text += model.vocab().tokenToString(t);
//            }
//            CHECK(text == " lower than mine's.");
//        }
//
//        {
//            ac::llama::ControlVector ctrlVector(model, {{ctrlVectorGguf, 2.f}});
//            ac::llama::Instance inst(model, {});
//            inst.addControlVector(ctrlVector);
//            inst.warmup(); // should be safe
//            auto& s = inst.startSession({});
//            std::vector<ac::llama::Token> tokens = model.vocab().tokenize("My car's fuel consumption is", true, true);
//            s.setInitialPrompt(tokens);
//            std::string text;
//            for (int i = 0; i < 5; ++i) {
//                auto t = s.getToken();
//                REQUIRE(t != ac::llama::Token_Invalid);
//                text += model.vocab().tokenToString(t);
//            }
//            CHECK(text == " more or less constant,");
//        }
//    }
//}

TEST_CASE("control_vector") {
    ac::llama::Model::Params iParams = {};
    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(Model_117m_q6_k, {}, iParams);
    ac::llama::Model model(lmodel, iParams);
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model.trainCtxLength() == 1024);
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());

    ac::llama::Instance inst(model, {});
    inst.warmup(); // should be safe

    const uint32_t nPredict = 30;

    std::vector<uint8_t> initialState;
    std::vector<uint8_t> sessionMiddleState;

    std::string prompt = "France has a long history of";
    std::string generatedStr;
    std::string generatedStr2;

    // create an original session which we'll use to store states
    {
        // session 1

        auto& s = inst.startSession({});
        auto tokens = model.vocab().tokenize(prompt, true, true);
        s.setInitialPrompt(tokens);

        // save the initial state
        initialState = s.getState();

        for (size_t i = 0; i < nPredict; i++) {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            auto text = model.vocab().tokenToString(t);
            generatedStr += text;

            if (i == nPredict / 2) {
                // save state after half of the tokens are generated
                sessionMiddleState = s.getState();
            }

            if (i > nPredict / 2) {
                // save generated string after after we've saved the state
                generatedStr2 += text;
            }
        }
        inst.stopSession();
    }

    // test restoring the initial state
    // since the sampler is in the initial state we should get the same string
    {
        auto& s = inst.startSession({});
        s.setState(initialState);
        std::string restoredStr;

        for (size_t i = 0; i < nPredict; i++) {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            auto text = model.vocab().tokenToString(t);
            restoredStr += text;
        }

        CHECK(restoredStr == generatedStr);
        inst.stopSession();
    }

    // Test restoring the middle state
    // In the middle state the sampler's RNG was not in the initial state, so
    // we should get a different string
    // However, the string should be the same for each session we start from that state
    {
        //restores session 1
        std::string restoredStr;
        {
            auto& s = inst.startSession({});
            s.setState(sessionMiddleState);

            for (size_t i = 0; i < nPredict / 2; i++) {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model.vocab().tokenToString(t);
                restoredStr += text;
            }
            inst.stopSession();
        }

        // Test that it's not the same as original due to samplers RNG state
        CHECK(restoredStr != generatedStr2);

        //restores session 2
        std::string restoredStr2;
        {
            auto& s = inst.startSession({});
            s.setState(sessionMiddleState);

            for (size_t i = 0; i < nPredict / 2; i++) {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model.vocab().tokenToString(t);
                restoredStr2 += text;
            }

            // Test that each session started from the same state produces the same string
            CHECK(restoredStr == restoredStr2);
            inst.stopSession();
        }
    }
}

TEST_CASE("grammar") {
    ac::llama::Model::Params iParams = {};
    auto lmodel = ac::llama::ModelRegistry::getInstance().loadModel(Model_117m_q6_k, {}, iParams);
    ac::llama::Model model(lmodel, iParams);
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model.trainCtxLength() == 1024);
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());

    SUBCASE("Numbers 6-9 only") {
        ac::llama::Instance::InitParams iparams;
        iparams.grammar =  R""""(
root        ::= en-char+ ([ \t\n] en-char+)*
en-char     ::= digit | letter
letter      ::= [a-zA-Z]
digit       ::= [6-9]
            )"""";

        ac::llama::Instance inst(model, iparams);
        inst.warmup(); // should be safe

        auto& s = inst.startSession({});
        std::vector<ac::llama::Token> tokens = model.vocab().tokenize("My name is Daniel and my age is", true, true);
        s.setInitialPrompt(tokens);
        std::string text;
        for (int i = 0; i < 5; ++i) {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            text += model.vocab().tokenToString(t);
        }

        CHECK(text == "s about 9 years old");
    }

    SUBCASE("Numbers 1-5 only") {
        ac::llama::Instance::InitParams iparams;
        iparams.grammar =  R""""(
root        ::= en-char+ ([ \t\n] en-char+)*
en-char     ::= digit | letter
letter      ::= [a-zA-Z]
digit       ::= [1-5]
            )"""";

        ac::llama::Instance inst(model, iparams);
        inst.warmup(); // should be safe

        auto& s = inst.startSession({});
        std::vector<ac::llama::Token> tokens = model.vocab().tokenize("My name is Daniel and my age is", true, true);
        s.setInitialPrompt(tokens);
        std::string text;
        for (int i = 0; i < 5; ++i) {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            text += model.vocab().tokenToString(t);
        }

        CHECK(text == "54 and I am an");
    }

    SUBCASE("All capital letters") {
        ac::llama::Instance::InitParams iparams;
        iparams.grammar =  R""""(
root        ::= en-char+ ([ \t\n] en-char+)*
en-char     ::= letter
letter      ::= [A-Z]
            )"""";

        ac::llama::Instance inst(model, iparams);
        inst.warmup(); // should be safe

        auto& s = inst.startSession({});
        std::vector<ac::llama::Token> tokens = model.vocab().tokenize("My name is Daniel and my age is", true, true);
        s.setInitialPrompt(tokens);
        std::string text;
        for (int i = 0; i < 5; ++i) {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            text += model.vocab().tokenToString(t);
        }

        CHECK(text == "ELLIE JONES");
    }
}
