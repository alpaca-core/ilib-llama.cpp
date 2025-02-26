// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Token.hpp"
#include <unordered_map>
#include <span>

namespace ac::llama {

class LogitComparer {
public:
    static bool compare(const TokenDataVector& data1, const TokenDataVector& data2);

private:
    static float jsd(const std::unordered_map<Token, float>& logits1, const std::unordered_map<Token, float>& logits2);
    static float euclidean_distance_sq(std::span<const TokenData> tokens);
};
}
