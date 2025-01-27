# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_library(ac-llama.cpp-schema INTERFACE)
add_library(ac::llama.cpp-schema ALIAS ac-llama.cpp-schema)
target_link_libraries(ac-llama.cpp-schema INTERFACE ac::frame)
target_include_directories(ac-llama.cpp-schema INTERFACE .)
