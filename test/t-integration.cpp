// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/InstanceEmbedding.hpp>
#include <ac/llama/Session.hpp>
#include <ac/llama/ControlVector.hpp>
#include <ac/llama/ResourceCache.hpp>

#include <doctest/doctest.h>

#include "ac-test-data-llama-dir.h"

struct GlobalFixture {
    GlobalFixture() {
        ac::llama::initLibrary();
    }
};

GlobalFixture globalFixture;
ac::local::ResourceManager rm;
ac::llama::ResourceCache resourceCache(rm);

const char* Model_117m_q6_k = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";

TEST_CASE("vocab only") {
    auto model = resourceCache.getModel({
        .gguf = Model_117m_q6_k,
        .params = {.vocabOnly = true}
    });
    CHECK(!!model->lmodel());

    auto& params = model->params();
    CHECK(params.gpu);
    CHECK(params.vocabOnly);

    CHECK(model->trainCtxLength() == 0); // no weights - no training context
    CHECK_FALSE(model->shouldAddBosToken());
    CHECK_FALSE(model->hasEncoder());

    // vocab works
    auto& vocab = model->vocab();
    CHECK(vocab.tokenToString(443) == " le");
    CHECK(vocab.tokenize("hello world", true, true) == std::vector<ac::llama::Token>{31373, 995});
}

TEST_CASE("inference") {
    auto model = resourceCache.getModel({.gguf = Model_117m_q6_k, .params = {}});
    CHECK(!!model->lmodel());

    auto& params = model->params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model->trainCtxLength() == 1024);
    CHECK_FALSE(model->shouldAddBosToken());
    CHECK_FALSE(model->hasEncoder());

    // general inference
    {
        ac::llama::Instance inst(*model, {});
        inst.warmup(); // should be safe

        std::vector<ac::llama::Token> tokens;

        // choose a very, very suggestive prompt and hope that all architectures will agree
        auto& s = inst.startSession({});
        tokens = model->vocab().tokenize("President George W.", true, true);
        s.setInitialPrompt(tokens);
        {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model->vocab().tokenToString(t);
                CHECK(text == " Bush");
        }

        SUBCASE("default sampler") {
            // add more very suggestive stuff
            tokens = model->vocab().tokenize(" sent troops to Cleveland which was hit by torrential", false, false);
            s.pushPrompt(tokens);
            {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model->vocab().tokenToString(t);
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
            tokens = model->vocab().tokenize(" sent troops to Cleveland which was hit by torrential", false, false);
            s.pushPrompt(tokens);
            {
                auto t = s.getToken();
                REQUIRE(t != ac::llama::Token_Invalid);
                auto text = model->vocab().tokenToString(t);
                CHECK(text.starts_with(" down"));
            }
        }
    }
}

TEST_CASE("session") {
    auto model = resourceCache.getModel({.gguf = Model_117m_q6_k, .params = {}});
    CHECK(!!model->lmodel());

    auto& params = model->params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model->trainCtxLength() == 1024);
    CHECK_FALSE(model->shouldAddBosToken());
    CHECK_FALSE(model->hasEncoder());
    ac::llama::Instance inst(*model, {});
    inst.warmup(); // should be safe
    SUBCASE("no initalization") {
        auto& s = inst.startSession({});
        SUBCASE("getToken") {
            CHECK_THROWS_WITH(s.getToken(), "Session hasn't started yet");
        }

        SUBCASE("pushPrompt") {
            auto tokens = model->vocab().tokenize("President George W.", true, true);
            CHECK_THROWS_WITH(s.pushPrompt(tokens), "Session hasn't started yet");
        }

        SUBCASE("getState") {
            CHECK_THROWS_WITH(s.getState(), "Session hasn't started yet");
        }
    }

    SUBCASE("double initialization") {
        auto& s = inst.startSession({});
        auto tokens = model->vocab().tokenize("President George W.", true, true);
        s.setInitialPrompt(tokens);
        CHECK_THROWS_WITH(s.setState({}), "Session already started");
    }

    SUBCASE("generating phase") {
        auto& s = inst.startSession({});
        {
            auto tokens = model->vocab().tokenize("President George W.", true, true);
            s.setInitialPrompt(tokens);
        }
        {
            auto t = s.getToken();
            CHECK(model->vocab().tokenToString(t) == " Bush");
        }
        {
            auto tokens = model->vocab().tokenize(" usually goes to Washington to", true, true);
            s.pushPrompt(tokens);
            auto t = s.getToken();
            auto text = model->vocab().tokenToString(t);
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
//    CHECK(!!model->lmodel());
//
//    auto& params = model->params();
//    CHECK(params.gpu);
//    CHECK_FALSE(params.vocabOnly);
//
//    CHECK(model->trainCtxLength() == 1024);
//    CHECK_FALSE(model->shouldAddBosToken());
//    CHECK_FALSE(model->hasEncoder());
//    {
//        std::string ctrlVectorGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6-control_vector.gguf";
//
//        {
//            ac::llama::ControlVector ctrlVector(model, {{ctrlVectorGguf, -2.f}});
//            ac::llama::Instance inst(model, {});
//            inst.addControlVector(ctrlVector);
//            inst.warmup(); // should be safe
//            auto& s = inst.startSession({});
//            std::vector<ac::llama::Token> tokens = model->vocab().tokenize("My car's fuel consumption is", true, true);
//            s.setInitialPrompt(tokens);
//            std::string text;
//            for (int i = 0; i < 5; ++i) {
//                auto t = s.getToken();
//                REQUIRE(t != ac::llama::Token_Invalid);
//                text += model->vocab().tokenToString(t);
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
//            std::vector<ac::llama::Token> tokens = model->vocab().tokenize("My car's fuel consumption is", true, true);
//            s.setInitialPrompt(tokens);
//            std::string text;
//            for (int i = 0; i < 5; ++i) {
//                auto t = s.getToken();
//                REQUIRE(t != ac::llama::Token_Invalid);
//                text += model->vocab().tokenToString(t);
//            }
//            CHECK(text == " more or less constant,");
//        }
//    }
//}

TEST_CASE("control_vector") {
    auto model = resourceCache.getModel({.gguf = Model_117m_q6_k, .params = {}});
    CHECK(!!model->lmodel());

    auto& params = model->params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model->trainCtxLength() == 1024);
    CHECK_FALSE(model->shouldAddBosToken());
    CHECK_FALSE(model->hasEncoder());

    ac::llama::Instance inst(*model, {});
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
        auto tokens = model->vocab().tokenize(prompt, true, true);
        s.setInitialPrompt(tokens);

        // save the initial state
        initialState = s.getState();

        for (size_t i = 0; i < nPredict; i++) {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            auto text = model->vocab().tokenToString(t);
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
            auto text = model->vocab().tokenToString(t);
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
                auto text = model->vocab().tokenToString(t);
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
                auto text = model->vocab().tokenToString(t);
                restoredStr2 += text;
            }

            // Test that each session started from the same state produces the same string
            CHECK(restoredStr == restoredStr2);
            inst.stopSession();
        }
    }
}

// commented out because it relies on specific calc
//TEST_CASE("grammar") {
//    auto model = resourceCache.getModel({.gguf = Model_117m_q6_k, .params = {}});
//    CHECK(!!model->lmodel());
//
//    auto& params = model->params();
//    CHECK(params.gpu);
//    CHECK_FALSE(params.vocabOnly);
//
//    CHECK(model->trainCtxLength() == 1024);
//    CHECK_FALSE(model->shouldAddBosToken());
//    CHECK_FALSE(model->hasEncoder());
//
//    SUBCASE("Numbers 6-9 only") {
//        ac::llama::Instance::InitParams iparams;
//        iparams.grammar =  R""""(
//root        ::= en-char+ ([ \t\n] en-char+)*
//en-char     ::= digit | letter
//letter      ::= [a-zA-Z]
//digit       ::= [6-9]
//            )"""";
//
//        ac::llama::Instance inst(*model, iparams);
//        inst.warmup(); // should be safe
//
//        auto& s = inst.startSession({});
//        std::vector<ac::llama::Token> tokens = model->vocab().tokenize("My name is Daniel and my age is", true, true);
//        s.setInitialPrompt(tokens);
//        std::string text;
//        for (int i = 0; i < 5; ++i) {
//            auto t = s.getToken();
//            REQUIRE(t != ac::llama::Token_Invalid);
//            text += model->vocab().tokenToString(t);
//        }
//
//        CHECK(text == "s about 9 years old");
//    }
//
//    SUBCASE("Numbers 1-5 only") {
//        ac::llama::Instance::InitParams iparams;
//        iparams.grammar =  R""""(
//root        ::= en-char+ ([ \t\n] en-char+)*
//en-char     ::= digit | letter
//letter      ::= [a-zA-Z]
//digit       ::= [1-5]
//            )"""";
//
//        ac::llama::Instance inst(*model, iparams);
//        inst.warmup(); // should be safe
//
//        auto& s = inst.startSession({});
//        std::vector<ac::llama::Token> tokens = model->vocab().tokenize("My name is Daniel and my age is", true, true);
//        s.setInitialPrompt(tokens);
//        std::string text;
//        for (int i = 0; i < 5; ++i) {
//            auto t = s.getToken();
//            REQUIRE(t != ac::llama::Token_Invalid);
//            text += model->vocab().tokenToString(t);
//        }
//
//        CHECK(text == "54 and I am an");
//    }
//
//    SUBCASE("All capital letters") {
//        ac::llama::Instance::InitParams iparams;
//        iparams.grammar =  R""""(
//root        ::= en-char+ ([ \t\n] en-char+)*
//en-char     ::= letter
//letter      ::= [A-Z]
//            )"""";
//
//        ac::llama::Instance inst(*model, iparams);
//        inst.warmup(); // should be safe
//
//        auto& s = inst.startSession({});
//        std::vector<ac::llama::Token> tokens = model->vocab().tokenize("My name is Daniel and my age is", true, true);
//        s.setInitialPrompt(tokens);
//        std::string text;
//        for (int i = 0; i < 5; ++i) {
//            auto t = s.getToken();
//            REQUIRE(t != ac::llama::Token_Invalid);
//            text += model->vocab().tokenToString(t);
//        }
//
//        CHECK(text == "ELLIE JONES");
//    }
//}

TEST_CASE("embedding") {
    const char* Model_bge_small_en = AC_TEST_DATA_LLAMA_DIR "/bge-small-en-v1.5-f16.gguf";
    auto model = resourceCache.getModel({.gguf = Model_bge_small_en, .params = {}});
    CHECK(model->trainCtxLength() == 512);
    CHECK_FALSE(model->hasEncoder());

    ac::llama::InstanceEmbedding inst(*model, {});
    auto tokens = model->vocab().tokenize("The main character in the story loved to eat pineapples.", true, true);
    auto embeddings = inst.getEmbeddingVector(tokens);
    CHECK(embeddings.size() == 384);

    std::vector<double> expected = { 0.00723457, 0.0672964, 0.00372222, -0.0458788, 0.00874835, 0.00432054, 0.109124, 0.00175256, 0.0172868, 0.0279001, -0.0223953, -0.00486074, 0.0112226, 0.0423849, 0.0285155, -0.00827027, 0.0247047, 0.0291312, -0.0786626, 0.0228906, 0.00884803, -0.0545553, 0.00242499, -0.0371614, 0.0145663, 0.0217592, -0.0379476, -0.012417, -0.031311, -0.0907524, -0.00270661, 0.0225516, 0.0166742, -0.023172, -0.0234313, 0.0518579, -0.00522299, 0.0011265, 0.00472722, -0.00702098, 0.0576354, 0.00290366, 0.0278902, -0.0283858, -0.00852266, -0.0349532, -0.0258749, 0.00864892, 0.0944385, -0.032376, -0.102357, -0.0570537, -0.0630057, -0.0366031, 0.0250621, 0.098078, 0.0734987, -0.0411082, -0.0521881, 0.00953602, 0.00460035, 0.014422, -0.0135636, 0.0487354, 0.0659704, -0.0510038, -0.0432206, 0.0347124, 0.000337169, 0.00681155, -0.0349383, 0.0462863, 0.0538792, 0.0218382, 0.0313523, 0.0300653, -0.00807435, -0.0203202, -0.0387424, 0.0531275, -0.0327624, 0.0274246, -0.000469622, 0.0148036, -0.0624161, -0.024254, 0.00340036, -0.0639136, -0.0116692, 0.0111668, 0.0197133, -0.0172656, -0.00784806, 0.0131758, -0.0579778, -0.00333637, -0.0446055, -0.0315641, -0.00882497, 0.354434, 0.0259944, -0.00811709, 0.060054, -0.0282549, -0.0194096, 0.0259942, -0.010753, -0.0537825, 0.0373867, 0.0552687, -0.0193146, 0.0116561, -0.00876264, 0.0234502, 0.0116844, 0.05702, 0.0531629, -0.0222655, -0.0866693, 0.0299643, 0.0295443, 0.0653484, -0.0565965, -0.00480344, -0.0103601, -0.0158926, 0.0853524, 0.0103825, 0.0322511, -0.0413097, 0.00330726, -0.0114999, -0.0119125, 0.0362464, 0.0276722, 0.0352711, 0.00796944, -0.0262156, -0.0402713, -0.0239314, -0.0561523, -0.0660272, -0.0442701, -0.0105944, 0.0156493, -0.0800205, 0.0467227, 0.0380684, -0.0314222, 0.109449, -0.031353, 0.0298688, -0.00155366, -0.00118869, 0.019166, -0.005014, 0.0258291, 0.0608314, 0.025612, 0.0432555, -0.010526, 0.0102892, 0.006778, -0.0804542, 0.0300636, 0.0019367, -0.00946688, 0.0633147, 0.00758261, 5.33199e-05, 0.034628, 0.0540261, -0.125455, 0.0102287, 0.00555666, 0.0565227, 0.00660611, 0.0497022, -0.0642718, -0.0175176, 0.0052292, -0.0916462, -0.0293923, 0.035024, 0.0503401, -0.0244895, 0.0903103, -0.007599, 0.039994, -0.0427364, 0.086443, 0.0564919, -0.0789255, -0.0167457, -0.0495721, -0.102541, 0.00512145, 0.00380079, -0.0334622, -0.00113675, -0.0529158, -0.0167595, -0.0920621, -0.0877459, 0.13931, -0.0685575, -0.00105833, 0.0327333, -0.0313494, -0.00404531, -0.0188106, 0.0216038, 0.0198488, 0.0505344, -0.00976201, 0.0336061, 0.0362691, 0.074989, 0.0155995, -0.0351994, 0.0128507, -0.0593599, 0.0247995, -0.265298, -0.0213482, -0.00865759, -0.0900854, -0.021827, 0.0103148, -0.0650073, -0.064416, 0.0544336, -0.0180563, -0.0126009, -0.0752656, 0.0396613, 0.0599272, 0.0281464, 0.0102912, 0.0458024, -0.058047, 0.0391549, 0.0234603, -0.00715374, -0.0155389, 0.0115466, -0.00202032, -0.0387425, 0.00196627, 0.189942, 0.138904, -0.031122, 0.00910502, -0.0201774, -0.00269432, -0.0330239, -0.0526063, 0.0205691, 0.0440849, 0.0738484, -0.0430935, -0.0378577, 0.00628437, 0.0127056, 0.0740211, -0.0536525, -0.0183475, -0.0520914, -0.0588744, 0.0223303, 0.0162849, 0.0259296, 0.0510308, 0.0436266, 0.0286193, -0.00156158, 0.0123141, -0.0173283, -0.030903, -0.0197604, 0.00607057, -0.055449, 0.0341534, -0.069812, 0.00289869, 0.000113235, -0.00571824, 0.00992975, -0.0031352, 0.00464151, -0.00241301, -0.0168796, 0.0110532, -0.0204679, -0.0672177, -0.0340668, -0.0370501, 0.0311332, 0.0710521, 0.0382394, -0.115705, -0.0437406, 0.00240175, -0.0409236, -0.00446289, -0.016308, 0.0365087, 0.0138439, -0.0697056, -0.00489864, 1.96082e-05, -0.00335489, -0.0200612, 0.058619, -2.70922e-05, -0.0262538, -0.0136708, 0.0375921, 0.0739009, -0.278277, 0.0240451, -0.0747427, 0.0138804, -0.00663228, 0.0299832, 0.028293, 0.0287869, -0.0257129, 0.0193498, 0.0975099, -0.0386528, 0.0509279, -0.0456842, -0.0403165, 0.0030311, -0.0409809, 0.017794, 0.0191697, -0.0300541, 0.0511827, 0.0638279, 0.148544, -0.0117107, -0.0472298, -0.0296059, -0.0162564, 0.0123344, -0.0239339, 0.0448291, 0.0605528, 0.0288511, 0.0759243, 0.0195688, 0.0373413, 0.0402353, 0.00830747, 0.000708879, 0.00346375, 0.0104776, -0.0347978, 0.0630426, -0.0580485, -0.0384997, 0.00238404, 0.00442908, -0.0406986, -0.00532351, -0.0112028, -0.0070308, 0.0222813, -0.0732604, 0.0689749, 0.0287737, 0.0242196, -0.0179569, -0.109264, 0.00263097, -0.0182948, -0.0285666, 0.00388148, -0.000162523, 0.00822485, 0.0211785, -0.00316543 };

    CHECK(embeddings.size() == expected.size());

    for (size_t i = 0; i < embeddings.size(); i++)
    {
        REQUIRE(embeddings[i] == doctest::Approx(expected[i]).epsilon(0.001));
    }
}
