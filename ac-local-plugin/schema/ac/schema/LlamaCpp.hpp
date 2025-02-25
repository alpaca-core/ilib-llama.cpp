// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include <ac/schema/Field.hpp>
#include <ac/Dict.hpp>
#include <vector>
#include <string>
#include <tuple>

namespace ac::schema {

inline namespace llama {

struct StateInitial {
    static constexpr auto id = "initial";
    static constexpr auto desc = "Initial state";

    struct OpLoadModel {
        static constexpr auto id = "load-model";
        static constexpr auto desc = "Load the llama.cpp model";

        struct Params{
            Field<std::string> ggufPath;
            Field<std::vector<std::string>> loraPaths = Default();
            Field<bool> useGpu = Default(true);
            Field<bool> vocabOnly = Default(false);
            Field<bool> prefixInputsWithBos = Default(false);

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(ggufPath, "gguf", "Path to the file with model data.");
                v(loraPaths, "loras", "Paths to lora adapters.");
                v(useGpu, "useGpu", "Try to load data on gpu.");
                v(vocabOnly, "vocabOnly", "Load only model vocabulary");
                v(prefixInputsWithBos, "prefixInputsWithBos", "Add bos token to interactive inputs.");

            }
        };

        using Return = nullptr_t;
    };

    using Ops = std::tuple<OpLoadModel>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct StateModelLoaded {
    static constexpr auto id = "model-loaded";
    static constexpr auto desc = "Model loaded state";

    struct OpStartInstance {
        static constexpr auto id = "start-instance";
        static constexpr auto desc = "Start a new instance of the llama.cpp model";

        struct Params {
            Field<std::string> instanceType = Default("general");
            Field<std::vector<std::string>> ctrlVectorPaths = Default();
            Field<uint32_t> ctxSize = Default(0);
            Field<uint32_t> batchSize = Default(2048);
            Field<uint32_t> ubatchSize = Default(512);

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(instanceType, "instance_type", "Type of the instance to start");
                v(ctrlVectorPaths, "ctrl-vectors", "Paths to the control vectors.");
                v(ctxSize, "ctx_size", "Size of the context");
                v(batchSize, "batch_size", "Size of the single batch");
                v(ubatchSize, "ubatch_size", "Size of the context");
            }
        };

        using Return = nullptr_t;
    };

    struct OpStartEmbeddingInstance {
        static constexpr auto id = "start-embedding-instance";
        static constexpr auto desc = "Start a new embedding instance of the llama.cpp model";

        struct Params {
            Field<std::string> instanceType = Default("general");
            Field<uint32_t> ctxSize = Default(0);
            Field<uint32_t> batchSize = Default(2048);
            Field<uint32_t> ubatchSize = Default(512);

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(instanceType, "instance_type", "Type of the instance to start");
                v(ctxSize, "ctx_size", "Size of the context");
                v(batchSize, "batch_size", "Size of the single batch");
                v(ubatchSize, "ubatch_size", "Size of the context");
            }
        };

        using Return = nullptr_t;
    };

    using Ops = std::tuple<OpStartInstance, OpStartEmbeddingInstance>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct StateInstance {
    static constexpr auto id = "instance";
    static constexpr auto desc = "Instance state";

    struct OpRun {
        static inline constexpr std::string_view id = "run";
        static inline constexpr std::string_view desc = "Run the llama.cpp inference and produce some output";

        struct Params {
            Field<std::string> prompt;
            Field<std::vector<std::string>> antiprompts = Default();
            Field<uint32_t> maxTokens = Default(0);
            Field<bool> stream = Default(true);

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(prompt, "prompt", "Prompt to complete");
                v(antiprompts, "antiprompts", "Antiprompts to trigger stop");
                v(maxTokens, "max_tokens", "Maximum number of tokens to generate. 0 for unlimited");
                v(stream, "stream", "Stream the output");
            }
        };

        struct Return {
            Field<std::string> result;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(result, "result", "Generated result (completion of prompt)");
            }
        };
    };

    struct OpGetTokenData {
        static inline constexpr std::string_view id = "get-token-data";
        static inline constexpr std::string_view desc = "Get the current state of the token context";

        using Params = nullptr_t;
        struct Return {
            Field<std::vector<int32_t>> tokens;
            Field<std::vector<float>> logits;
            Field<std::vector<float>> probs;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(tokens, "tokens", "Tokens in the context");
                v(logits, "logits", "Logits for the tokens");
                v(probs, "probs", "Probabilities for the tokens");
            }
        };
    };

    struct OpCompareTokenData {
        static inline constexpr std::string_view id = "compare-tokens";
        static inline constexpr std::string_view desc = "Compare two sets of tokens";

        struct Params {
            Field<std::vector<int32_t>> tokens1;
            Field<std::vector<float>> logits1;
            Field<std::vector<float>> probs1;
            Field<std::vector<int32_t>> tokens2;
            Field<std::vector<float>> logits2;
            Field<std::vector<float>> probs2;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(tokens1, "tokens1", "Tokens in the first set");
                v(logits1, "logits1", "Logits for the first set");
                v(probs1, "probs1", "Probabilities for the first set");
                v(tokens2, "tokens2", "Tokens in the second set");
                v(logits2, "logits2", "Logits for the second set");
                v(probs2, "probs2", "Probabilities for the second set");
            }
        };

        struct Return {
            Field<bool> equal;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(equal, "equal", "Whether the two sets are equal");
            }
        };
    };

    struct OpChatBegin {
        static inline constexpr std::string_view id = "begin-chat";
        static inline constexpr std::string_view desc = "Begin a chat session";

        struct Params {
            Field<std::string> setup = Default();
            Field<std::string> roleUser = Default("User");
            Field<std::string> roleAssistant = Default("Assistant");

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(setup, "setup", "Initial setup for the chat session");
                v(roleUser, "role_user", "Role name for the user");
                v(roleAssistant, "role_assistant", "Role name for the assistant");
            }
        };

        using Return = nullptr_t;
    };

    struct OpStopInstance {
        static inline constexpr std::string_view id = "stop-instance";
        static inline constexpr std::string_view desc = "Stop the current instance";

        using Params = nullptr_t;
        using Return = nullptr_t;
    };

    using Ops = std::tuple<OpRun, OpGetTokenData, OpCompareTokenData, OpChatBegin, OpStopInstance>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct StateStreaming {
    static constexpr auto id = "streaming";
    static constexpr auto desc = "Streaming state";

    struct OpAbort {
        static constexpr auto id = "abort";
        static constexpr auto desc = "Abort the streaming";
        using Params = nullptr_t;
        using Return = nullptr_t;
    };

    struct StreamToken {
        static constexpr auto id = "token";
        static constexpr auto desc = "Token stream";
        using Type = std::string;
    };

    using Ops = std::tuple<OpAbort>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<StreamToken>;
};

struct StateChat {
    static constexpr auto id = "chat";
    static constexpr auto desc = "Chat state";

    struct OpChatEnd {
        static inline constexpr std::string_view id = "end-chat";
        static inline constexpr std::string_view desc = "End a chat session";

        using Params = nullptr_t;
        using Return = nullptr_t;
    };

    struct OpAddChatPrompt {
        static inline constexpr std::string_view id = "add-chat-prompt";
        static inline constexpr std::string_view desc = "Add a prompt to the chat session as a user";

        struct Params {
            Field<std::string> prompt = Default();

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(prompt, "prompt", "Prompt to add to the chat session");
            }
        };

        using Return = nullptr_t;
    };

    struct OpGetChatResponse {
        static inline constexpr std::string_view id = "get-chat-response";
        static inline constexpr std::string_view desc = "Get a response from the chat session";

        using Params = nullptr_t;

        struct Return {
            Field<std::string> response;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(response, "response", "Response from the chat session");
            }
        };
    };

    using Ops = std::tuple<OpChatEnd, OpAddChatPrompt, OpGetChatResponse>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
};

struct StateEmbeddingInstance {
    static constexpr auto id = "embedding-instance";
    static constexpr auto desc = "Embedding instance state";

    struct OpRun {
        static inline constexpr std::string_view id = "run";
        static inline constexpr std::string_view description = "Run to produce an embedding vector";

        struct Params {
            Field<std::string> prompt;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(prompt, "prompt", "Prompt to generate the embedding for");
            }
        };

        struct Return {
            Field<std::vector<float>> result;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(result, "result", "Generated result (embedding vector)");
            }
        };
    };

    using Ops = std::tuple<OpRun>;
};

struct Interface {
    static inline constexpr std::string_view id = "llama.cpp";
    static inline constexpr std::string_view desc = "Inference based on our fork of https://github.com/ggerganov/llama.cpp";

    using States = std::tuple<StateInitial, StateModelLoaded, StateInstance, StateChat>;
};

} // namespace llama

} // namespace ac::local::schema
