# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
function(add_llama_plugin_example name)
    set(tgt example-aclp-llama-${name})
    add_executable(${tgt} ep-${name}.cpp)
    target_link_libraries(${tgt} PRIVATE
        ac::local
        ac::jalog
        ac-test-data::llama
        aclp::llama-info
    )
    set_target_properties(${tgt} PROPERTIES FOLDER example)
endfunction()

add_llama_plugin_example(run)
add_llama_plugin_example(chat)
