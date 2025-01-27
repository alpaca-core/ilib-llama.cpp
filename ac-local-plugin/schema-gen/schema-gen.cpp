// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/schema/LlamaCpp.hpp>
#include <ac/local/schema/GenerateProviderSchemaDict.hpp>
#include <iostream>

int main() {
    auto d = ac::schema::generateProviderSchema<acnl::ordered_json, ac::schema::LlamaCppProvider>();
    std::cout << d.dump(2) << std::endl;
    return 0;
}
