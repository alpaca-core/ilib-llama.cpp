// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "ChatMsg.hpp"

#include <span>
#include <unordered_map>

struct llama_chat_message;

namespace minja {
class chat_template;
}
namespace ac::llama {
class Model;

class AC_LLAMA_EXPORT ChatFormat {
public:
    struct Params {
        std::string chatTemplate;
        std::string bosToken;
        std::string eosToken;
    };

    explicit ChatFormat(Params params);

    ~ChatFormat();

    const std::string& tpl() const noexcept { return m_templateStr; }

    // wrapper around llama_chat_apply_template
    // throw an error on unsupported template
    std::string formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt);

    // format single message taking history into account
    std::string formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt = false);

private:
    std::string apply(std::span<const llama_chat_message> chat, size_t chatSize, bool addAssistantPrompt) const;

    std::string m_templateStr;
    int m_templateId;
    std::unique_ptr<minja::chat_template> m_minjaTemplate;
};

ChatFormat::Params getChatParams(const Model& model);
} // namespace ac::llama
