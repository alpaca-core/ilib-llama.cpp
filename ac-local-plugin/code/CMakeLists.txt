# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_ac_local_plugin(
    NAME llama
    SOURCES
        LocalLlama.cpp
    SCHEMAS
        llama-schema.yml
    LIBRARIES
        ac::llama
)
