# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
CPMAddPackage(gh:iboB/doctest-util@0.1.3)
set_target_properties(doctest PROPERTIES FOLDER test)
set_target_properties(doctest-main PROPERTIES FOLDER test)

macro(llama_test test)
    add_doctest_lib_test(${test} ac-llama t-${test}.cpp)
endmacro()

llama_test(Antiprompt)
llama_test(ChatFormat)
llama_test(Session)

add_doctest_lib_test(integration ac-llama
    SOURCES
        t-integration.cpp
    LIBRARIES
        ac-test-data::llama
)
