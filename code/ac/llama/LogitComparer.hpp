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

    static bool compare(const TokenDataVector& data1, const TokenDataVector& data2);

private:
    static float jsd(const std::unordered_map<Token, float>& logits1, const std::unordered_map<Token, float>& logits2);
    static float euclidean_distance_sq(const TokenDataVector& logits1, int32_t count);

    // compute cosine similarity only for the union of the two sets of tokens
    float cosine_similarity(const TokenDataVector& logits1, const TokenDataVector& logits2);
};
}
