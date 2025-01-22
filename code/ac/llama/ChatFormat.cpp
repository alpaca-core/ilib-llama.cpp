// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ChatFormat.hpp"

#include <llama.h>
#include <llama-chat.h>

#include <astl/throw_stdex.hpp>
#include <astl/move.hpp>

#include <vector>
#include <cassert>
#include <stdexcept>
#include <map>
namespace ac::llama {
namespace {

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

// The maximum characters that are in the template but not in the message that will be added in the final message
static const std::map<llm_chat_template, uint32_t>::value_type MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA[] = {
    { LLM_CHAT_TEMPLATE_CHATML ,            60 },
    { LLM_CHAT_TEMPLATE_LLAMA_2,            35 },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS,        35 },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,    35 },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,  35 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V1 ,        35 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V3 ,        35 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN , 35 },
    { LLM_CHAT_TEMPLATE_MISTRAL_V7 ,        35 },
    { LLM_CHAT_TEMPLATE_PHI_3,              35 },
    { LLM_CHAT_TEMPLATE_PHI_4,              60 },
    { LLM_CHAT_TEMPLATE_FALCON_3,           35 },
    { LLM_CHAT_TEMPLATE_ZEPHYR ,            35 },
    { LLM_CHAT_TEMPLATE_MONARCH ,           35 },
    { LLM_CHAT_TEMPLATE_GEMMA ,             60 },
    { LLM_CHAT_TEMPLATE_ORION ,             35 },
    { LLM_CHAT_TEMPLATE_OPENCHAT ,          35 },
    { LLM_CHAT_TEMPLATE_VICUNA ,            35 },
    { LLM_CHAT_TEMPLATE_VICUNA_ORCA ,       35 },
    { LLM_CHAT_TEMPLATE_DEEPSEEK ,          35 },
    { LLM_CHAT_TEMPLATE_DEEPSEEK_2,         35 },
    { LLM_CHAT_TEMPLATE_DEEPSEEK_3,         35 },
    { LLM_CHAT_TEMPLATE_COMMAND_R ,         60 },
    { LLM_CHAT_TEMPLATE_LLAMA_3,            60 },
    { LLM_CHAT_TEMPLATE_CHATGML_3,          35 },
    { LLM_CHAT_TEMPLATE_CHATGML_4,          35 },
    { LLM_CHAT_TEMPLATE_MINICPM ,           35 },
    { LLM_CHAT_TEMPLATE_EXAONE_3,           35 },
    { LLM_CHAT_TEMPLATE_RWKV_WORLD ,        35 },
    { LLM_CHAT_TEMPLATE_GRANITE ,           60 },
    { LLM_CHAT_TEMPLATE_GIGACHAT ,          60 },
    { LLM_CHAT_TEMPLATE_MEGREZ ,            60 },
};

static const std::map<llm_chat_template, bool>::value_type HAS_SINGLE_MESSAGE_NEWLINE_DATA[] = {
    { LLM_CHAT_TEMPLATE_CHATML ,            true },
    { LLM_CHAT_TEMPLATE_LLAMA_2,            false },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS,        false },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,    false },
    { LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,  false },
    { LLM_CHAT_TEMPLATE_MISTRAL_V1 ,        false },
    { LLM_CHAT_TEMPLATE_MISTRAL_V3 ,        false },
    { LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN , false },
    { LLM_CHAT_TEMPLATE_MISTRAL_V7 ,        false },
    { LLM_CHAT_TEMPLATE_PHI_3,              false },
    { LLM_CHAT_TEMPLATE_PHI_4,              false },
    { LLM_CHAT_TEMPLATE_FALCON_3,           false },
    { LLM_CHAT_TEMPLATE_ZEPHYR ,            false },
    { LLM_CHAT_TEMPLATE_MONARCH ,           false },
    { LLM_CHAT_TEMPLATE_GEMMA ,             true },
    { LLM_CHAT_TEMPLATE_ORION ,             false },
    { LLM_CHAT_TEMPLATE_OPENCHAT ,          false },
    { LLM_CHAT_TEMPLATE_VICUNA ,            false },
    { LLM_CHAT_TEMPLATE_VICUNA_ORCA ,       false },
    { LLM_CHAT_TEMPLATE_DEEPSEEK ,          false },
    { LLM_CHAT_TEMPLATE_DEEPSEEK_2,         false },
    { LLM_CHAT_TEMPLATE_DEEPSEEK_3,         false },
    { LLM_CHAT_TEMPLATE_COMMAND_R ,         false },
    { LLM_CHAT_TEMPLATE_LLAMA_3,            false },
    { LLM_CHAT_TEMPLATE_CHATGML_3,          false },
    { LLM_CHAT_TEMPLATE_CHATGML_4,          false },
    { LLM_CHAT_TEMPLATE_MINICPM ,           false },
    { LLM_CHAT_TEMPLATE_EXAONE_3,           false },
    { LLM_CHAT_TEMPLATE_RWKV_WORLD ,        false },
    { LLM_CHAT_TEMPLATE_GRANITE ,           false },
    { LLM_CHAT_TEMPLATE_GIGACHAT ,          false },
    { LLM_CHAT_TEMPLATE_MEGREZ ,            false },
};

static const std::map<llm_chat_template, uint32_t> MAX_SINGLE_MESSAGE_TEMPLATE_SIZE(
    std::begin(MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA), std::end(MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA));

static const std::map<llm_chat_template, bool> HAS_SINGLE_MESSAGE_NEWLINE(
    std::begin(HAS_SINGLE_MESSAGE_NEWLINE_DATA), std::end(HAS_SINGLE_MESSAGE_NEWLINE_DATA));

// Check if we have sizes for all supported templates
static_assert(std::size(MAX_SINGLE_MESSAGE_TEMPLATE_SIZE_DATA) == size_t(LLM_CHAT_TEMPLATE_UNKNOWN));
static_assert(std::size(HAS_SINGLE_MESSAGE_NEWLINE_DATA) == size_t(LLM_CHAT_TEMPLATE_UNKNOWN));

} // namespace

ChatFormat::ChatFormat(std::string tpl)
    : m_template(astl::move(tpl))
    , m_templateId(llm_chat_detect_template(m_template))
{
    if (m_templateId == LLM_CHAT_TEMPLATE_UNKNOWN) {
        throw_ex{} << "Unsupported template: " << m_template;
    }
}

std::string ChatFormat::formatChat(std::span<const ChatMsg> chat, bool addAssistantPrompt) const {
    auto [lchat, size] = fromChatMsg(chat);
    return apply(lchat, size, addAssistantPrompt);
}

std::string ChatFormat::formatMsg(const ChatMsg& msg, std::span<const ChatMsg> history, bool addAssistantPrompt) const {
    std::string ret;

    // if the formatted past messages end with a newline,
    // we must preserve it
    if (history.size() && addAssistantPrompt && HAS_SINGLE_MESSAGE_NEWLINE.at(llm_chat_template(m_templateId))) {
        ret = "\n";
    };
    auto [lchat, size ] = std::pair<std::vector<llama_chat_message>, size_t>();
    lchat.push_back({msg.role.c_str(), msg.text.c_str()});
    size += msg.role.size() + msg.text.size();
    ret += apply(lchat, size, addAssistantPrompt);

    // apply diff
    // ret += fmtNew.substr(fmtHistory.size());
    return ret;
}

std::string ChatFormat::apply(std::span<const llama_chat_message> chat, size_t size, bool addAssistantPrompt) const {
    if (size == 0) return {};

    // TODO: take assistant prompt into account
    auto msgSize = MAX_SINGLE_MESSAGE_TEMPLATE_SIZE.at(llm_chat_template(m_templateId));
    auto formattedChatAllocSize = size + chat.size() * msgSize;
    std::string fmt(formattedChatAllocSize, '\0');

    // run the first time and get the total output length
    int32_t res = llama_chat_apply_template(m_template.c_str(), chat.data(), chat.size(),
        addAssistantPrompt, fmt.data(), int32_t(fmt.size()));

    if (res > int32_t(fmt.size())) {
        // The assert should never happen, the case when it might occur is
        // - not updated value in MAX_SINGLE_MESSAGE_TEMPLATE_SIZE, because a template has been changed in llama.cpp
        //assert(false && "The max template size is not calculated properly! Will re-run the template with the correct size.");
        // optimistic size was not enough
        fmt.resize(res);
        res = llama_chat_apply_template(m_template.c_str(), chat.data(), chat.size(),
            addAssistantPrompt, fmt.data(), int32_t(fmt.size()));
    }

    assert(res >= 0);

    fmt.resize(res);
    return fmt;
}

} // namespace ac::llama
