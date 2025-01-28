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

struct LlamaCppInterface {
    static inline constexpr std::string_view id = "llama.cpp";
    static inline constexpr std::string_view description = "ilib-llama.cpp-specific interface";

    struct OpRun {
        static inline constexpr std::string_view id = "run";
        static inline constexpr std::string_view description = "Run the llama.cpp inference and produce some output";

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
        static inline constexpr std::string_view description = "Begin a chat session";

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

    struct OpChatEnd {
        static inline constexpr std::string_view id = "end-chat";
        static inline constexpr std::string_view description = "End a chat session";

        using Params = nullptr_t;
        using Return = nullptr_t;
    };

    struct OpAddChatPrompt {
        static inline constexpr std::string_view id = "add-chat-prompt";
        static inline constexpr std::string_view description = "Add a prompt to the chat session as a user";

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
        static inline constexpr std::string_view description = "Get a response from the chat session";

        using Params = nullptr_t;

        struct Return {
            Field<std::string> response;

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(response, "response", "Response from the chat session");
            }
        };
    };

    using Ops = std::tuple<OpRun, OpChatBegin, OpChatEnd, OpAddChatPrompt, OpGetChatResponse>;
};

struct LlamaCppProvider {
    static inline constexpr std::string_view id = "llama.cpp";
    static inline constexpr std::string_view description = "Inference based on our fork of https://github.com/ggerganov/llama.cpp";

    using Params = nullptr_t;

    struct InstanceGeneral {
        static inline constexpr std::string_view id = "general";
        static inline constexpr std::string_view description = "General instance";

        struct Params {
            Field<uint32_t> ctxSize = Default(0);
            Field<uint32_t> batchSize = Default(2048);
            Field<uint32_t> ubatchSize = Default(512);

            template <typename Visitor>
            void visitFields(Visitor& v) {
                v(ctxSize, "ctx_size", "Size of the context");
                v(batchSize, "batch_size", "Size of the single batch");
                v(ubatchSize, "ubatch_size", "Size of the context");
            }
        };

        using Interfaces = std::tuple<LlamaCppInterface>;
    };

    using Instances = std::tuple<InstanceGeneral>;
};

} // namespace ac::local::schema
