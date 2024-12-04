# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_executable(ac-llama.cpp-schema-gen)
target_sources(ac-llama.cpp-schema-gen PRIVATE
    schema-gen.cpp
)
target_link_libraries(ac-llama.cpp-schema-gen PRIVATE
    ac::llama.cpp-schema
)
