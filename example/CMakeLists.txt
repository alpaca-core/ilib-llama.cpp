# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
function (add_example name)
    add_executable(example-ac-llama-${name} e-${name}.cpp)
    target_link_libraries(example-ac-llama-${name} PRIVATE
        ac::llama
        ac-test-data::llama
        ac::jalog
    )
    set_target_properties(example-ac-llama-${name} PROPERTIES FOLDER example)
endfunction()

add_example(basic)
add_example(embedding)
add_example(infill)
add_example(verify)

CPMAddPackage(gh:alpaca-core/helper-imgui-sdl@1.0.0)
if(TARGET ac-dev::imgui-sdl-app)
    add_executable(example-ac-llama-gui e-gui.cpp)
    target_link_libraries(example-ac-llama-gui PRIVATE
        ac::llama
        ac-test-data::llama
        ac::jalog
        ac-dev::imgui-sdl-app
    )
    set_target_properties(example-ac-llama-gui PROPERTIES FOLDER example)
endif()

CPMAddPackage(gh:nmslib/hnswlib@0.8.0)
if(TARGET hnswlib::hnswlib)
    add_executable(example-ac-llama-rag e-rag.cpp)
    target_link_libraries(example-ac-llama-rag PRIVATE
        ac::llama
        ac-test-data::llama
        ac::jalog
        hnswlib
    )
    set_target_properties(example-ac-llama-rag PROPERTIES FOLDER example)
endif()
