// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ChatFormat.hpp"

#include "Logging.hpp"
#include "Model.hpp"

#include <llama.h>
#include <llama-chat.h>

#include <ac/vendor/nlohmann/json.hpp>
namespace nlohmann = acnl;
#include <minja/chat-template.hpp>

#include <astl/throw_stdex.hpp>
#include <astl/move.hpp>

#include <vector>
#include <cassert>
#include <map>

namespace ac::llama {
namespace {

#define CHATML_TEMPLATE_SRC \
    "{%- for message in messages -%}\n" \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n" \
    "{%- if add_generation_prompt -%}\n" \
    "  {{- '<|im_start|>assistant\n' -}}\n" \
    "{%- endif -%}"

std::pair<std::vector<llama_chat_message>, size_t> ac2llamaChatMessages(std::span<const ChatMsg> chat) {
    std::vector<llama_chat_message> lchat;
    size_t size = 0;
    lchat.reserve(chat.size());
    for (const auto& msg : chat) {
        lchat.push_back({msg.role.c_str(), msg.text.c_str()});
        size += msg.role.size();
        size += msg.text.size();
    }
    return {lchat, size};
}

std::pair<acnl::json, size_t> ac2jsonChatMessages(std::span<const ChatMsg> chat) {
    acnl::json messages = acnl::json::array();
    size_t size = 0;
    for (const auto& msg : chat) {
        acnl::json jmsg = {
            {"role", msg.role},
            {"content", msg.text},
        };
        messages.push_back(jmsg);
        size += msg.role.size();
        size += msg.text.size();
    }
    return {messages, size};
}


std::string applyLlama(const std::string& templateStr, std::span<llama_chat_message> lchat, size_t size, bool addAssistantPrompt) {
    auto allocSize = (size * 5) / 4; // optimistic 25% more than the original size
    std::string fmt(allocSize, '\0');

    // run the first time and get the total output length
    int32_t res = llama_chat_apply_template(templateStr.c_str(), lchat.data(), lchat.size(),
        addAssistantPrompt, fmt.data(), int32_t(fmt.size()));

    if (res > int32_t(fmt.size())) {
        // optimistic size was not enough
        fmt.resize(res);
        res = llama_chat_apply_template(templateStr.c_str(), lchat.data(), lchat.size(),
            addAssistantPrompt, fmt.data(), int32_t(fmt.size()));
    }

    assert(res >= 0);

    fmt.resize(res);
    return fmt;
}

std::string applyJinja(minja::chat_template* minjaTemplate, acnl::json jChat, bool /*addAssistantPrompt*/) {
    auto startsWith = [](const std::string& str, const std::string& prefix) {
        return str.rfind(prefix, 0) == 0;
    };

    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = jChat;

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto result = minjaTemplate->apply(tmpl_inputs, tmpl_opts);
    if (startsWith(result, minjaTemplate->bos_token())) {
        result = result.substr(minjaTemplate->bos_token().size());
    }
    if (startsWith(result, minjaTemplate->eos_token())) {
        result = result.substr(0, result.size() - minjaTemplate->eos_token().size());
    }
    return result;
}


} // namespace

ChatFormat::ChatFormat(std::string templateStr)
    : m_templateStr(std::move(templateStr))
    , m_templateId(llm_chat_detect_template(m_templateStr.c_str()))
    , m_useJinja(false)
{
    if (m_templateId == LLM_CHAT_TEMPLATE_UNKNOWN) {
        throw_ex{} << "Unsupported chat template: " << m_templateStr;
    }
}

ChatFormat::ChatFormat(Params params)
    : m_useJinja(true)
{
    m_templateStr = std::move(params.chatTemplate);

    try {
        m_minjaTemplate = std::make_unique<minja::chat_template>(m_templateStr, params.bosToken, params.eosToken);
    } catch (const std::exception & e) {
        throw_ex{} << "Unsupported jinja template. Error: " << e.what();
    }
}

ChatFormat::~ChatFormat() {}

std::string ChatFormat::formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) {
    acnl::json jchat;
    std::vector<llama_chat_message> lchat;
    size_t size = 0;
    if (m_useJinja) {
        auto res = ac2jsonChatMessages(chat);
        jchat = std::move(res.first);
        size = res.second;
    } else {
        auto res = ac2llamaChatMessages(chat);
        lchat = std::move(res.first);
        size = res.second;
    }

    if (size == 0) return {};

    return m_useJinja ?
    applyJinja(m_minjaTemplate.get(), jchat, addAssistantPrompt) :
    applyLlama(m_templateStr, lchat, size, addAssistantPrompt);
}

std::string ChatFormat::formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) {
    if (history.empty()) {
        return formatChat({&msg, 1}, addAssistantPrompt);
    }

    acnl::json jchat;
    std::vector<llama_chat_message> lchat;
    size_t size = 0;
    if (m_useJinja) {
        auto res = ac2jsonChatMessages(history);
        jchat = std::move(res.first);
        size = res.second;
    } else {
        auto res = ac2llamaChatMessages(history);
        lchat = std::move(res.first);
        size = res.second;
    }

    auto fmtHistory = m_useJinja ?
        applyJinja(m_minjaTemplate.get(), jchat, false) :
        applyLlama(m_templateStr, lchat, size, false);

    std::string ret;

    // if the formatted past messages end with a newline,
    // we must preserve it
    if (addAssistantPrompt && fmtHistory.ends_with('\n')) {
        ret = "\n";
    };

    if (m_useJinja) {
        jchat.push_back({
            {"role", msg.role},
            {"content", msg.text},
        });
    } else {
        lchat.push_back({msg.role.c_str(), msg.text.c_str()});
        size += msg.role.size() + msg.text.size();
    }

    auto fmtNew = m_useJinja ?
        applyJinja(m_minjaTemplate.get(), jchat, addAssistantPrompt) :
        applyLlama(m_templateStr, lchat, size, addAssistantPrompt);

    // apply diff
    ret += fmtNew.substr(fmtHistory.size());
    return ret;
}

ChatFormat::Params getChatParams(const Model& model) {
    ChatFormat::Params chatParams;
    chatParams.chatTemplate = llama_model_chat_template(model.lmodel(), nullptr);

    const auto getTokenStr = [&](llama_token token, const char * name, const char * jinja_variable_name) {
        if (token == LLAMA_TOKEN_NULL) {
            if (chatParams.chatTemplate.find(jinja_variable_name) != std::string::npos) {
                LLAMA_LOG(Warning, "Vocab doesn't have a \"%s\" token, jinja template won't work as intended.\n", name);
            }
            return std::string();
        }
        return model.vocab().tokenToString(token, true);
    };

    const auto * vocab = llama_model_get_vocab(model.lmodel());
    chatParams.bosToken = getTokenStr(llama_vocab_bos(vocab), "BOS", "bos_token");
    chatParams.eosToken = getTokenStr(llama_vocab_eos(vocab), "EOS", "eos_token");

    return chatParams;
}


} // namespace ac::llama
