// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include <cstdint>
#include <vector>

namespace ac::llama {
using Token = std::int32_t;
inline constexpr Token Token_Invalid = -1;

struct TokenData {
    Token token;
    float logit;
};

using TokenDataVector = std::vector<TokenData>;
} // namespace ac::llama
