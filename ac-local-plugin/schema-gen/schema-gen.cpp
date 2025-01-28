// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/schema/LlamaCpp.hpp>
#include <ac/schema/GenerateSchemaDict.hpp>
#include <iostream>

int main() {
    auto d = Interface_generateSchemaDict<acnl::ordered_json>(ac::schema::llama::Interface{});
    std::cout << d.dump(2) << std::endl;
    return 0;
}
