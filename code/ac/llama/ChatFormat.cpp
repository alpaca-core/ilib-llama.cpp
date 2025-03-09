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
#include <stdexcept>
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

std::pair<std::vector<llama_chat_message>, size_t> fromChatMsg(std::span<const ChatMsg> chat) {
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

acnl::json transformMessages(std::span<const llama_chat_message> chat) {
    acnl::json messages = acnl::json::array();
    for (const auto& msg : chat) {
        acnl::json jmsg = {
            {"role", msg.role},
            {"content", msg.content},
        };
        messages.push_back(jmsg);
    }
    return messages;
}

// The maximum characters that are in the template but not in the message that will be added in the final message
// All those are calculated by running empty message with each of roles ["system", "user", "assistant"] and got their maximum
// Check calculateSingleMessageMaxSize in t-ChatFormat.cpp
static const std::map<llm_chat_template, uint32_t>::value_type MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA[] = {
    { LLM_CHAT_TEMPLATE_CHATML ,            46 },
    { LLM_CHAT_TEMPLATE_LLAMA_2,            11 },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS,        20 },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,    20 },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,  20 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V1 ,        12 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V3 ,        10 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN , 9 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V7 ,        26 },
    { LLM_CHAT_TEMPLATE_PHI_3,              27 },
    { LLM_CHAT_TEMPLATE_PHI_4,              63 },
    { LLM_CHAT_TEMPLATE_FALCON_3,           20 },
    { LLM_CHAT_TEMPLATE_ZEPHYR ,            33 },
    { LLM_CHAT_TEMPLATE_MONARCH ,           19 },
    { LLM_CHAT_TEMPLATE_GEMMA ,             51 },
    { LLM_CHAT_TEMPLATE_ORION ,             20 },
    { LLM_CHAT_TEMPLATE_OPENCHAT ,          53 },
    { LLM_CHAT_TEMPLATE_VICUNA ,            17 },
    { LLM_CHAT_TEMPLATE_VICUNA_ORCA ,       17 },
    { LLM_CHAT_TEMPLATE_DEEPSEEK ,          28 },
    { LLM_CHAT_TEMPLATE_DEEPSEEK_2,         39 },
    { LLM_CHAT_TEMPLATE_DEEPSEEK_3,         52 },
    { LLM_CHAT_TEMPLATE_COMMAND_R ,         94 },
    { LLM_CHAT_TEMPLATE_LLAMA_3,            95 },
    { LLM_CHAT_TEMPLATE_CHATGML_3,          29 },
    { LLM_CHAT_TEMPLATE_CHATGML_4,          30 },
    { LLM_CHAT_TEMPLATE_GLMEDGE,            30 },
    { LLM_CHAT_TEMPLATE_MINICPM ,           8 },
    { LLM_CHAT_TEMPLATE_EXAONE_3,           31 },
    { LLM_CHAT_TEMPLATE_RWKV_WORLD ,        14 },
    { LLM_CHAT_TEMPLATE_GRANITE ,           90 },
    { LLM_CHAT_TEMPLATE_GIGACHAT ,          99 },
    { LLM_CHAT_TEMPLATE_MEGREZ ,            73 },
};

static const std::map<llm_chat_template, uint32_t> MAX_SINGLE_MESSAGE_TEMPLATE_SIZE(
    std::begin(MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA), std::end(MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA));

// Check if we have sizes for all supported templates
static_assert(std::size(MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA) == size_t(LLM_CHAT_TEMPLATE_UNKNOWN));

} // namespace

ChatFormat::ChatFormat(Params params) {
    m_templateStr = std::move(params.chatTemplate);

    try {
        m_minjaTemplate = std::make_unique<minja::chat_template>(m_templateStr, params.bosToken, params.eosToken);
    } catch (const std::exception & e) {
        LLAMA_LOG(Error, "Failed to parse jinja chat template (defaulting to chatml): %s \n", e.what());
        m_minjaTemplate = std::make_unique<minja::chat_template>(CHATML_TEMPLATE_SRC, params.bosToken, params.eosToken);
    }
}

ChatFormat::~ChatFormat() {}

std::string ChatFormat::formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) {
    assert(m_templateId != LLM_CHAT_TEMPLATE_UNKNOWN);

    auto [lchat, size] = fromChatMsg(chat);
    return apply(lchat, size, addAssistantPrompt);
}

std::string ChatFormat::formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) {
    if (history.empty()) {
        return formatChat({&msg, 1}, addAssistantPrompt);
    }

    auto [lchat, size] = fromChatMsg(history.subspan(history.size() - 1, 1));
    auto fmtHistory = apply(lchat, size, false);

    std::string ret;

    // if the formatted past messages end with a newline,
    // we must preserve it
    if (addAssistantPrompt && fmtHistory.ends_with('\n')) {
        ret = "\n";
    }

    lchat.push_back({msg.role.c_str(), msg.text.c_str()});
    size += msg.role.size() + msg.text.size();
    auto fmtNew = apply(lchat, size, addAssistantPrompt);

    // apply diff
    ret += fmtNew.substr(fmtHistory.size());
    return ret;
}

std::string ChatFormat::apply(std::span<const llama_chat_message> chat, size_t size, bool addAssistantPrompt) const {
    if (size == 0) return {};

    // // TODO: take assistant prompt into account
    // auto msgSize = MAX_SINGLE_MESSAGE_TEMPLATE_SIZE.at(llm_chat_template(m_templateId));
    // auto formattedChatAllocSize = size + chat.size() * msgSize;
    // std::string fmt(formattedChatAllocSize, '\0');

    // // run the first time and get the total output length
    // int32_t res = llama_chat_apply_template(m_templateStr.c_str(), chat.data(), chat.size(),
    //     addAssistantPrompt, fmt.data(), int32_t(fmt.size()));

    // if (res > int32_t(fmt.size())) {
    //     // The assert should never happen, the case when it might occur is
    //     // - not updated value in MAX_SINGLE_MESSAGE_TEMPLATE_SIZE, because a template has been changed in llama.cpp
    //     assert(false && "The max template size is not calculated properly! Will re-run the template with the correct size.");
    //     // optimistic size was not enough
    //     fmt.resize(res);
    //     res = llama_chat_apply_template(m_templateStr.c_str(), chat.data(), chat.size(),
    //         addAssistantPrompt, fmt.data(), int32_t(fmt.size()));
    // }

    // assert(res >= 0);

    // fmt.resize(res);
    // return fmt;

    auto startsWith = [](const std::string& str, const std::string& prefix) {
        return str.rfind(prefix, 0) == 0;
    };

    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = transformMessages(chat);
    // tmpl_inputs.tools = tools;
    // tmpl_inputs.add_generation_prompt = true;
    // tmpl_inputs.extra_context = extra_context;
    // TODO: add flag to control date/time, if only for testing purposes.
    // tmpl_inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto result = m_minjaTemplate->apply(tmpl_inputs, tmpl_opts);
    if (startsWith(result, m_minjaTemplate->bos_token())) {
        result = result.substr(m_minjaTemplate->bos_token().size());
    }
    if (startsWith(result, m_minjaTemplate->eos_token())) {
        result = result.substr(0, result.size() - m_minjaTemplate->eos_token().size());
    }
    return result;
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
