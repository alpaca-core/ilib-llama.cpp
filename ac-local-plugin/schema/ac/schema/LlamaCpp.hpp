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
            Field<std::string> ggufPath = std::nullopt;
            Field<std::vector<std::string>> loraPaths = Default();

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(ggufPath, "gguf", "Path to the file with model data.");
                v(loraPaths, "loras", "Paths to lora adapters.");
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

    using Ops = std::tuple<OpStartInstance>;
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

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(prompt, "prompt", "Prompt to complete");
                v(antiprompts, "antiprompts", "Antiprompts to trigger stop");
                v(maxTokens, "max_tokens", "Maximum number of tokens to generate. 0 for unlimited");
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

    using Ops = std::tuple<OpRun, OpChatBegin>;
    using Ins = std::tuple<>;
    using Outs = std::tuple<>;
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

struct Interface {
    static inline constexpr std::string_view id = "llama.cpp";
    static inline constexpr std::string_view desc = "Inference based on our fork of https://github.com/ggerganov/llama.cpp";

    using States = std::tuple<StateInitial, StateModelLoaded, StateInstance, StateChat>;
};

} // namespace llama

} // namespace ac::local::schema
