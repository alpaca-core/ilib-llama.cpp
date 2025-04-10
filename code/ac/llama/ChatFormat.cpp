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

class ChatFormat::impl {
public:
    virtual ~impl() = default;
    virtual std::string formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) const = 0;
    virtual std::string formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) const = 0;
};

class LlamaImpl final : public ChatFormat::impl {
public:
    LlamaImpl(std::string templateStr)
        : m_templateStr(std::move(templateStr))
        , m_templateId(llm_chat_detect_template(m_templateStr.c_str()))
    {
        if (m_templateId == LLM_CHAT_TEMPLATE_UNKNOWN) {
            throw_ex{} << "Unsupported chat template: " << m_templateStr;
        }
    }

    virtual std::string formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) const override{
        auto [lchat, size] = ac2llamaChatMessages(chat);
        return size != 0 ? applyLlama(lchat, size, addAssistantPrompt) : "";
    }

    virtual std::string formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) const override {
        if (history.empty()) {
            return formatChat({&msg, 1}, addAssistantPrompt);
        }

        auto [lchat, size] = ac2llamaChatMessages(history);
        auto fmtHistory = applyLlama(lchat, size, false);

        lchat.push_back({msg.role.c_str(), msg.text.c_str()});
        size += msg.role.size() + msg.text.size();

        std::string ret;
        // if the past_msg ends with a newline, we must preserve it in the formatted version
        if (addAssistantPrompt && fmtHistory.ends_with('\n')) {
            ret = "\n";
        };

        auto fmtNew = applyLlama(lchat, size, addAssistantPrompt);
        return ret + fmtNew.substr(fmtHistory.size());
    }

    ~LlamaImpl() {}

private:
    std::pair<std::vector<llama_chat_message>, size_t> ac2llamaChatMessages(std::span<const ChatMsg> chat) const {
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

    std::string applyLlama(std::span<llama_chat_message> lchat, size_t size, bool addAssistantPrompt) const {
        auto allocSize = (size * 5) / 4; // optimistic 25% more than the original size
        std::string fmt(allocSize, '\0');

        // run the first time and get the total output length
        int32_t res = llama_chat_apply_template(m_templateStr.c_str(), lchat.data(), lchat.size(),
            addAssistantPrompt, fmt.data(), int32_t(fmt.size()));

        if (res > int32_t(fmt.size())) {
            // optimistic size was not enough
            fmt.resize(res);
            res = llama_chat_apply_template(m_templateStr.c_str(), lchat.data(), lchat.size(),
                addAssistantPrompt, fmt.data(), int32_t(fmt.size()));
        }

        assert(res >= 0);

        fmt.resize(res);
        return fmt;
    }

    std::string m_templateStr;
    int m_templateId;
};

class JinjaImpl final : public ChatFormat::impl {
public:
    JinjaImpl(ChatFormat::Params params)
    {
        m_templateStr = std::move(params.chatTemplate);
        m_assistantRole = std::move(params.roleAssistant);

        try {
            m_minjaTemplate = std::make_unique<minja::chat_template>(m_templateStr, params.bosToken, params.eosToken);
        } catch (const std::exception & e) {
            throw_ex{} << "Unsupported jinja template. Error: " << e.what();
        }
    }

    ~JinjaImpl() {}

    virtual std::string formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) const override {
        auto [jChat, size] = ac2jsonChatMessages(chat);
        return size == 0 ? std::string{} : applyJinja(jChat, addAssistantPrompt);
    }

    virtual std::string formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) const override {
        if (history.empty()) {
            return formatChat({&msg, 1}, addAssistantPrompt);
        }

        auto [jchat, size] = ac2jsonChatMessages(history);
        auto fmtHistory = applyJinja(jchat, addAssistantPrompt);

        jchat.push_back({{"role", msg.role}, {"content", msg.text}});
        auto fmtNew = applyJinja(jchat, addAssistantPrompt);

        return fmtNew.substr(fmtHistory.size());
    }

private:
    std::pair<acnl::json, size_t> ac2jsonChatMessages(std::span<const ChatMsg> chat) const {
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

    std::string applyJinja(acnl::json jChat, bool addAssistantPrompt) const {
        auto startsWith = [](const std::string& str, const std::string& prefix) {
            return str.rfind(prefix, 0) == 0;
        };

        minja::chat_template_inputs tmpl_inputs;
        tmpl_inputs.messages = jChat;
        tmpl_inputs.add_generation_prompt = addAssistantPrompt;
        tmpl_inputs.extra_context = {
            {"assistant_role",  m_assistantRole}
        };

        // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
        // instead of using `chat_template_options.use_bos_token = false`, since these tokens
        // may be needed inside the template / between messages too.
        auto result = m_minjaTemplate->apply(tmpl_inputs);
        if (startsWith(result, m_minjaTemplate->bos_token())) {
            result = result.substr(m_minjaTemplate->bos_token().size());
        }
        if (startsWith(result, m_minjaTemplate->eos_token())) {
            result = result.substr(0, result.size() - m_minjaTemplate->eos_token().size());
        }
        return result;
    }

    std::unique_ptr<minja::chat_template> m_minjaTemplate;
    std::string m_templateStr;
    std::string m_assistantRole;
};


ChatFormat::ChatFormat(std::string templateStr)
: m_templateStr(templateStr)
, m_impl(std::make_unique<LlamaImpl>(std::move(templateStr)))
{}

ChatFormat::ChatFormat(Params params)
    : m_templateStr(params.chatTemplate)
    , m_impl(std::make_unique<JinjaImpl>(params))
{}

ChatFormat::~ChatFormat() {}

std::string ChatFormat::formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) const {
    return m_impl->formatChat(chat, addAssistantPrompt);
}

std::string ChatFormat::formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) const {
    return m_impl->formatMsg(msg, history, addAssistantPrompt);
}

ChatFormat::Params ChatFormat::getChatParams(const Model& model) {
    ChatFormat::Params chatParams;
    if (auto tmpl = llama_model_chat_template(model.lmodel(), nullptr)) {
        chatParams.chatTemplate = tmpl;
    }

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
