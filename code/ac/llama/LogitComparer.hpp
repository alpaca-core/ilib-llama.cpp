// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Token.hpp"
#include <vector>
#include <iostream>

namespace ac::llama {

class LogitComparer {
public:
    LogitComparer() = default;
    ~LogitComparer() = default;

    bool compare(const TokenDataVector& logit1, const TokenDataVector& logit2, size_t size);

private:

    float jsd(const TokenDataVector& logits1, const TokenDataVector& logits2);

    // compute cosine similarity only for the union of the two sets of tokens
    float cosine_similarity(const TokenDataVector& logits1, const TokenDataVector& logits2);
};
}
