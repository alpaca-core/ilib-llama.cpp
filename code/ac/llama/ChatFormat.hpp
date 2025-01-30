// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "ChatMsg.hpp"

#include <jinja2cpp/template.h>

#include <span>
#include <unordered_map>

struct llama_chat_message;

namespace ac::llama {
class AC_LLAMA_EXPORT ChatFormat {
public:
    // the template string here can be either an id or a markup
    explicit ChatFormat(std::string tpl);

    // create a custom jinja template
    // it should have a loop over 'messages' field
    // each message must have a 'role' and 'content' field
    explicit ChatFormat(std::string tpl, std::span<std::string> roles);

    const std::string& tpl() const noexcept { return m_template; }
    const char* templateId() const noexcept;

    // wrapper around llama_chat_apply_template
    // throw an error on unsupported template
    std::string formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt);

    // wrapper around jinja template formatting
    // params are options for the template
    // TODO: Do not expose jinj2::ValuesMap to the user
    std::string formatChat(std::span<const ChatMsg> chat, jinja2::ValuesMap params = {});

    // format single message taking history into account
    std::string formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt = false);

    static std::vector<const char*> getSupportedTemplates();

private:
    std::string apply(std::span<const llama_chat_message> chat, size_t chatSize, bool addAssistantPrompt) const;

    std::string m_template;
    int m_templateId;
    jinja2::Template m_jTemplate;
};
} // namespace ac::llama
