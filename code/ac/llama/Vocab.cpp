// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Vocab.hpp"
#include "Model.hpp"
#include <llama-vocab.h>
#include <type_traits>
#include <cassert>

#include <llama.h>

namespace ac::llama {

static_assert(std::is_same_v<Token, llama_token>);

Vocab::Vocab(const Model& model)
    : m_model(model)
    , m_lVocab(llama_model_get_vocab(model.lmodel()))
{}
Vocab::~Vocab() = default;

Token Vocab::decoderStartToken() const noexcept {
    auto model = m_model.lmodel();
    auto ret = llama_model_decoder_start_token(model);
    if (ret != -1) return ret;
    return llama_vocab_bos(m_lVocab);
}

bool Vocab::isEog(Token token) const noexcept {
    return llama_vocab_is_eog(m_lVocab, token);
}

int32_t Vocab::nTokens() const noexcept {
    return llama_vocab_n_tokens(m_lVocab);
}

std::vector<Token> Vocab::tokenize(std::string_view text, bool addSpecial, bool parseSpecial) const {
    int32_t numTokens = int32_t(text.length()) + 2 * addSpecial; // optimistic max
    std::vector<Token> ret(numTokens);
    numTokens = llama_tokenize(m_lVocab, text.data(), int32_t(text.length()), ret.data(), numTokens, addSpecial, parseSpecial);
    if (numTokens < 0) {
        ret.resize(-numTokens);
        [[maybe_unused]] int check =
            llama_tokenize(m_lVocab, text.data(), int32_t(text.length()), ret.data(), -numTokens, addSpecial, parseSpecial);
        assert(check == -numTokens);
    }
    else {
        ret.resize(numTokens);
    }
    return ret;
}

std::string Vocab::tokenToString(Token token, bool special) const {
    std::string ret;

    auto to_piece = [&]() {
        return llama_token_to_piece(m_lVocab, token, ret.data(), int32_t(ret.size()), 0, special);
    };

    ret.resize(ret.capacity()); // make use of small string optimization
    const auto len = to_piece();
    if (len < 0) {
        ret.resize(-len);
        auto check = to_piece();
        assert(check == -len);
    }
    else {
        ret.resize(len);
    }

    return ret;
}

} // namespace ac::llama
