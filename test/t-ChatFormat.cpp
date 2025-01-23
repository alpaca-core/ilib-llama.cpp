// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/ChatFormat.hpp>
#include <astl/u8c.h>
#include <doctest/doctest.h>
#include <vector>

// All supported chat templates in
// the list is taken from llama-chat.cpp:27
constexpr std::array LLAMA_CHAT_TEMPLATES = {
    "chatml",
    "llama2",
    "llama2-sys", // missing test
    "llama2-sys-bos",
    "llama2-sys-strip",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "phi3",
    "phi4",
    "falcon3", // missing test
    "zephyr", // missing test
    "monarch",
    "gemma",
    "orion",
    "openchat",
    "vicuna",
    "vicuna-orca",
    "deepseek",
    "deepseek2",
    "deepseek3", // missing test
    "command-r",
    "llama3",
    "chatglm3",
    "chatglm4",
    "minicpm",
    "exaone3", // missing test
    "rwkv-world", // missing test
    "granite",
    "gigachat",
    "megrez",
};

int32_t calculateSingleMessageMaxSize(const std::string& tmpl) {
    const std::vector<ac::llama::ChatMsg> systemChat = {
        {"system", ""},
    };

    const std::vector<ac::llama::ChatMsg> userChat = {
        {"user", ""},
    };

    const std::vector<ac::llama::ChatMsg> asssistantChat = {
        {"assistant", ""}
    };

    ac::llama::ChatFormat fmt{tmpl};

    auto result1 = fmt.formatChat(systemChat, true);
    auto result2 = fmt.formatChat(userChat, true);
    auto result3 = fmt.formatChat(asssistantChat, true);

    auto result1Len = result1.size() > std::string("system").size() ? result1.size() - std::string("system").size() : result1.size();
    auto result2Len = result2.size() > std::string("user").size() ? result2.size() - std::string("user").size() : result2.size();
    auto result3Len = result3.size() > std::string("assistant").size() ? result3.size() - std::string("assistant").size() : result3.size();
    auto max = std::max(result1Len, std::max(result2Len, result3Len));

    return max;
}

TEST_CASE("apply") {
    const std::vector<ac::llama::ChatMsg> chat = {
        {"system", "You are a helpful assistant"},
        {"user", "Hello"},
        {"assistant", "Hi there"},
        {"user", "Who are you"},
        {"assistant", "   I am an assistant   "},
        {"user", "Another question"},
    };

    struct Test {
        std::string tplId;
        std::string tpl;
        std::string expected;
    };
    const std::vector<Test> tests = {
        // tests copied from llama.cpp/test-chat-template.cpp
        { // teknium/OpenHermes-2.5-Mistral-7B
            "chatml",
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
        },
        { // mistralai/Mistral-7B-Instruct-v0.2
            "llama2",
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            "[INST] You are a helpful assistant\nHello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
        },
        { // TheBloke/FusionNet_34Bx2_MoE-AWQ
            "llama2-sys-bos",
            "{%- for idx in range(0, messages|length) -%}\\n{%- if messages[idx]['role'] == 'user' -%}\\n{%- if idx > 1 -%}\\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\\n{%- else -%}\\n{{- messages[idx]['content'] + ' [/INST]' -}}\\n{%- endif -%}\\n{% elif messages[idx]['role'] == 'system' %}\\n{{- '[INST] <<SYS>>\\\\n' + messages[idx]['content'] + '\\\\n<</SYS>>\\\\n\\\\n' -}}\\n{%- elif messages[idx]['role'] == 'assistant' -%}\\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\\n{% endif %}\\n{% endfor %}",
            "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi there</s><s>[INST] Who are you [/INST]   I am an assistant   </s><s>[INST] Another question [/INST]",
        },
        { // bofenghuang/vigogne-2-70b-chat
            "llama2-sys-strip",
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\\\n' + system_message + '\\\\n<</SYS>>\\\\n\\\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\\\n' + content.strip() + '\\\\n<</SYS>>\\\\n\\\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
            "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi there</s>[INST] Who are you [/INST]I am an assistant</s>[INST] Another question [/INST]",
        },
        { // mlabonne/AlphaMonarch-7B
            "monarch",
            "{% for message in messages %}{{bos_token + message['role'] + '\\n' + message['content'] + eos_token + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\\n' }}{% endif %}",
            "system\nYou are a helpful assistant</s>\n<s>user\nHello</s>\n<s>assistant\nHi there</s>\n<s>user\nWho are you</s>\n<s>assistant\n   I am an assistant   </s>\n<s>user\nAnother question</s>\n<s>assistant\n",
        },
        { // google/gemma-7b-it
            "gemma",
            "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}",
            "<start_of_turn>user\nYou are a helpful assistant\n\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        },
        { // OrionStarAI/Orion-14B-Chat
            "orion",
            "{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'] + '\\n\\nAssistant: ' + eos_token }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}",
            "Human: You are a helpful assistant\n\nHello\n\nAssistant: </s>Hi there</s>Human: Who are you\n\nAssistant: </s>   I am an assistant   </s>Human: Another question\n\nAssistant: </s>",
        },
        { // openchat/openchat-3.5-0106
            // The included chat_template differs from the author's suggestions here:
            // https://huggingface.co/openchat/openchat_3.5/discussions/5#65448109b4a3f3a2f486fd9d
            // So we match against the included template but implement the suggested version.
            "openchat",
            "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
            "You are a helpful assistant<|end_of_turn|>GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi there<|end_of_turn|>GPT4 Correct User: Who are you<|end_of_turn|>GPT4 Correct Assistant:    I am an assistant   <|end_of_turn|>GPT4 Correct User: Another question<|end_of_turn|>GPT4 Correct Assistant:",
        },
        { // deepseek-ai/deepseek-coder-33b-instruct
            "deepseek",
            "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}",
            "You are a helpful assistant### Instruction:\nHello\n### Response:\nHi there\n<|EOT|>\n### Instruction:\nWho are you\n### Response:\n   I am an assistant   \n<|EOT|>\n### Instruction:\nAnother question\n### Response:\n",
        },
        { // eachadea/vicuna-13b-1.1
            // No template included in tokenizer_config.json, so this template likely needs to be manually set.
            "vicuna",
            "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '' + message['content'] + '\n\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '</s>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}",
            "You are a helpful assistant\n\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:",
        },
        { // Orca-Vicuna
            // No template included in tokenizer_config.json, so this template likely needs to be manually set.
            "vicuna-orca",
            "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{-'SYSTEM: ' + message['content'] + '\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '</s>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}",
            "SYSTEM: You are a helpful assistant\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:",
        },
        { // CohereForAI/c4ai-command-r-plus
            "command-r",
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Who are you<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>I am an assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Another question<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        },
        { // Llama-3
            "llama3",
            "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}",
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI am an assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnother question<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        { //Phi-3-mini
            "phi3",
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
            "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        },
        { //Phi-3-small
            "phi3",
            "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
            "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        },
        { //Phi-3-medium
            "phi3",
            "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
            "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        },
        { //Phi-3-vision
            "phi3",
            "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '<|assistant|>\n' -}}{% endif %}",
            "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        },
        { // ChatGLM3
            "chatglm3",
            "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
            "[gMASK]sop<|system|>\n You are a helpful assistant<|user|>\n Hello<|assistant|>\n Hi there<|user|>\n Who are you<|assistant|>\n    I am an assistant   <|user|>\n Another question<|assistant|>",
        },
        { // ChatGLM4
            "chatglm4",
            U8C("[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] %}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ tool['function'] | tojson(indent=4) }}\n......{% endif %}{% endfor %}{% endif %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"),
            "[gMASK]<sop><|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi there<|user|>\nWho are you<|assistant|>\n   I am an assistant   <|user|>\nAnother question<|assistant|>",
        },
        { // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
            "minicpm",
            U8C("{% for message in messages %}{% if message['role'] == 'user' %}{{'<用户>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}"),
            U8C("You are a helpful assistant<用户>Hello<AI>Hi there<用户>Who are you<AI>I am an assistant<用户>Another question<AI>"),
        },
        { // DeepSeek-V2
            "deepseek2",
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
            U8C("You are a helpful assistant\n\nUser: Hello\n\nAssistant: Hi there<｜end▁of▁sentence｜>User: Who are you\n\nAssistant:    I am an assistant   <｜end▁of▁sentence｜>User: Another question\n\nAssistant:"),
        },
        { // ibm-granite/granite-3.0-8b-instruct
            "granite",
            "{%- if tools %}\n    {{- '<|start_of_role|>available_tools<|end_of_role|>\n' }}\n    {%- for tool in tools %}\n    {{- tool | tojson(indent=4) }}\n    {%- if not loop.last %}\n        {{- '\n\n' }}\n    {%- endif %}\n    {%- endfor %}\n    {{- '<|end_of_text|>\n' }}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n    {{- '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'user' %}\n    {{- '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>'  + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant_tool_call' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'tool_response' %}\n    {{- '<|start_of_role|>tool_response<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- endif %}\n    {%- if loop.last and add_generation_prompt %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>' }}\n    {%- endif %}\n{%- endfor %}",
            "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>\n",
        },
        { // mistralai/Mistral-7B-Instruct-v0.2 (mistralai 'v1' template with a system prompt)
            "mistral-v1",
            "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n",
            " [INST] You are a helpful assistant\n\nHello [/INST] Hi there</s> [INST] Who are you [/INST]    I am an assistant   </s> [INST] Another question [/INST]",
        },
        { // Mistral-Large-Instruct-2407 (mistralai 'v3' template; modified to have system prompt at start)
            "mistral-v3",
            "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == \"tool\" or message.role == \"tool_results\" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message[\"role\"] == \"user\") != (ns.index % 2 == 0) %}\n            {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS] [\" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- '{\"type\": \"function\", \"function\": {' }}\n                {%- for key, val in tool.items() if key != \"return\" %}\n                    {%- if val is string %}\n                        {{- '\"' + key + '\": \"' + val + '\"' }}\n                    {%- else %}\n                        {{- '\"' + key + '\": ' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- \", \" }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST] \" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST] \" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- \"[TOOL_CALLS] [\" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- \" \" + message[\"content\"]|trim + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS] {\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n",
            "[INST] You are a helpful assistant\n\nHello[/INST] Hi there</s>[INST] Who are you[/INST] I am an assistant</s>[INST] Another question[/INST]",
        },
        { // Mistral-Nemo-Instruct-2407 (mistralai 'v3-tekken' template; modified to have system prompt at start)
            "mistral-v3-tekken",
            "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == \"tool\" or message.role == \"tool_results\" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message[\"role\"] == \"user\") != (ns.index % 2 == 0) %}\n            {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS][\" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- '{\"type\": \"function\", \"function\": {' }}\n                {%- for key, val in tool.items() if key != \"return\" %}\n                    {%- if val is string %}\n                        {{- '\"' + key + '\": \"' + val + '\"' }}\n                    {%- else %}\n                        {{- '\"' + key + '\": ' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- \", \" }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST]\" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif (message.tool_calls is defined and message.tool_calls is not none) %}\n        {{- \"[TOOL_CALLS][\" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- message[\"content\"] + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS]{\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n",
            "[INST]You are a helpful assistant\n\nHello[/INST]Hi there</s>[INST]Who are you[/INST]   I am an assistant   </s>[INST]Another question[/INST]",
        },
        { // mistralai/Mistral-Large-Instruct-2411 (mistralai 'v7' template)
            "mistral-v7",
            "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'system' %}{{ '[SYSTEM_PROMPT] ' + message['content'] + '[/SYSTEM_PROMPT]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% else %}{{ raise_exception('Only user, system and assistant roles are supported!') }}{% endif %}{% endfor %}",
            "[SYSTEM_PROMPT] You are a helpful assistant[/SYSTEM_PROMPT][INST] Hello[/INST] Hi there</s>[INST] Who are you[/INST]    I am an assistant   </s>[INST] Another question[/INST]",
        },
        { // ai-sage/GigaChat-20B-A3B-instruct
            "gigachat",
            "{% if messages[0]['role'] == 'system' -%}\n    {%- set loop_messages = messages[1:] -%}\n    {%- set system_message = bos_token + messages[0]['content'] + additional_special_tokens[1] -%}\n{%- else -%}\n    {%- set loop_messages = messages -%}\n    {%- set system_message = bos_token + '' -%}\n{%- endif -%}\n{%- for message in loop_messages %}\n    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}\n    {% endif %}\n    \n    {%- if loop.index0 == 0 -%}\n        {{ system_message -}}\n    {%- endif -%}\n    {%- if message['role'] == 'user' -%}\n        {{ message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1] -}}\n        {{ 'available functions' + additional_special_tokens[0] + additional_special_tokens[2] + additional_special_tokens[3]  + additional_special_tokens[1] -}}\n    {%- endif -%}\n    {%- if message['role'] == 'assistant' -%}\n        {{ message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1] -}}\n    {%- endif -%}\n    {%- if loop.last and add_generation_prompt -%}\n        {{ 'assistant' + additional_special_tokens[0] -}}\n    {%- endif -%}\n{%- endfor %}",
            "<s>You are a helpful assistant<|message_sep|>user<|role_sep|>Hello<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>Hi there<|message_sep|>user<|role_sep|>Who are you<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>   I am an assistant   <|message_sep|>user<|role_sep|>Another question<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>",
        },
        { // Infinigence/Megrez-3B-Instruct
            "megrez",
            /*u8*/"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|role_start|>system<|role_end|>你是Megrez-3B-Instruct，将针对用户的问题给出详细的、积极的回答。<|turn_end|>' }}{% endif %}{{ '<|role_start|>' + message['role'] + '<|role_end|>' + message['content'] + '<|turn_end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|role_start|>assistant<|role_end|>' }}{% endif %}",
            "<|role_start|>system<|role_end|>You are a helpful assistant<|turn_end|><|role_start|>user<|role_end|>Hello<|turn_end|><|role_start|>assistant<|role_end|>Hi there<|turn_end|><|role_start|>user<|role_end|>Who are you<|turn_end|><|role_start|>assistant<|role_end|>   I am an assistant   <|turn_end|><|role_start|>user<|role_end|>Another question<|turn_end|><|role_start|>assistant<|role_end|>",
        },
        { // phi-4
            "phi4",
            "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}",
            "<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|>Hello<|im_end|><|im_start|>assistant<|im_sep|>Hi there<|im_end|><|im_start|>user<|im_sep|>Who are you<|im_end|><|im_start|>assistant<|im_sep|>   I am an assistant   <|im_end|><|im_start|>user<|im_sep|>Another question<|im_end|><|im_start|>assistant<|im_sep|>",
        },
    };

    CHECK(LLAMA_CHAT_TEMPLATES.size() == ac::llama::ChatFormat::getSupportedTemplates().size());

    std::bitset<LLAMA_CHAT_TEMPLATES.size()> visited;
    auto visit = [&](const std::string& id) {
        for (size_t i = 0; i < LLAMA_CHAT_TEMPLATES.size(); i++) {
            if (LLAMA_CHAT_TEMPLATES[i] == id) {
                visited.set(i);
                break;
            }
        }
    };

    for (const auto& test : tests) {
        ac::llama::ChatFormat fmt{test.tpl};
        CHECK(fmt.tpl() == test.tpl);
        CHECK(fmt.formatChat(chat, true) == test.expected);
        CHECK(std::string(fmt.templateId()) != std::string(""));
        visit(fmt.templateId());
    }

    constexpr std::array missingTests = {
        "llama2-sys",
        "falcon3",
        "zephyr",
        "deepseek3",
        "exaone3",
        "rwkv-world"
    };
    for (auto& tpl : missingTests) {
        visit(tpl);
    }

    // Uncomment to check for missing tests if the next CHECK fails
    // for (size_t i = 0; i < LLAMA_CHAT_TEMPLATES.size(); i++) {
    //     if (!visited.test(i)) {
    //         std::cout << "Missing test for template: " << LLAMA_CHAT_TEMPLATES[i] << std::endl;
    //     }
    // }
    CHECK(visited.all());
}

TEST_CASE("formatMsg - system") {
    ac::llama::ChatMsg msg = {"system", "You are a helpful assistant"};

    auto test = [&](std::string id) {
        ac::llama::ChatFormat fmt{id};
        CHECK(fmt.tpl() == id);
        return fmt.formatMsg(msg, {}, false);
    };

    CHECK(test("chatml") == "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n");
    CHECK(test("llama2") == "[INST] You are a helpful assistant\n");
    CHECK(test("gemma") == ""); // for gemma, system message is merged with user message
    CHECK(test("llama3") == "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>");
}

TEST_CASE("formatMsg - chat") {
    SUBCASE("empty chat") {
        ac::llama::ChatMsg msg = {"user", "How are you"};

        auto test = [&](std::string id) {
            ac::llama::ChatFormat fmt{id};
            CHECK(fmt.tpl() == id);
            return fmt.formatMsg(msg, {}, true);
        };

        CHECK(test("chatml") == "<|im_start|>user\nHow are you<|im_end|>\n<|im_start|>assistant\n");
        CHECK(test("llama2") == "[INST] How are you [/INST]");
        CHECK(test("gemma") == "<start_of_turn>user\nHow are you<end_of_turn>\n<start_of_turn>model\n");
        CHECK(test("llama3") == "<|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    SUBCASE("full chat") {
        const std::vector<ac::llama::ChatMsg> chat = {
            {"system", "You are a helpful assistant"},
            {"user", "Hello"},
            {"assistant", "I am assistant"}
        };
        ac::llama::ChatMsg msg = {"user", "How are you"};

        auto test = [&](std::string id) {
            ac::llama::ChatFormat fmt{id};
            CHECK(fmt.tpl() == id);
            return fmt.formatMsg(msg, chat, true);
        };

        CHECK(test("chatml") == "\n<|im_start|>user\nHow are you<|im_end|>\n<|im_start|>assistant\n");
        CHECK(test("llama2") == "[INST] How are you [/INST]");
        CHECK(test("gemma") == "\n<start_of_turn>user\nHow are you<end_of_turn>\n<start_of_turn>model\n");
        CHECK(test("llama3") == "<|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }
}

TEST_CASE("invalid") {
    std::string badTpl = "bad template";
    CHECK_THROWS_WITH_AS(ac::llama::ChatFormat{badTpl}, "Unsupported template: bad template", std::runtime_error);
}
