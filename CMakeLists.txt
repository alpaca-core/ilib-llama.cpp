# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(ilib-llama.cpp
    VERSION 1.0.0
    DESCRIPTION "Example plugin for the Alpaca Core Local SDK"
    LANGUAGES CXX
)

include(./get-ac-build.cmake)

#################
# cmake lib
CPMAddPackage(gh:iboB/icm@1.5.4)
list(APPEND CMAKE_MODULE_PATH
    "${icm_SOURCE_DIR}"
)

include(icm_add_lib)

include(init_ac_prj)

include(ac_build_prj_util)
include(ac_build_plugin_util)

#################
# options

option(AC_LLAMA_BUILD_TESTS "${PROJECT_NAME}: build tests" ${testsDefault})
option(AC_LLAMA_BUILD_EXAMPLES "${PROJECT_NAME}: build examples" ${examplesDefault})
mark_as_advanced(AC_LLAMA_BUILD_TESTS AC_LLAMA_BUILD_EXAMPLES)

init_ac_plugin_option(LLAMA)

#######################################
# packages

add_ac_local(0.1.9)

#######################################
# submodule and config

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(haveCuda YES)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS "${CMAKE_PROJECT_NAME}: system CUDA found")
    # we should add -forward-unknown-to-host-compiler but it somehow works without it
    # could it be that it depends on the CMake version?
endif()

if(haveCuda)
    set(GGML_CUDA ON)
endif()

set(GGML_CCACHE OFF)

set(LLAMA_BUILD_COMMON OFF)

add_subdirectory(llama.cpp)

#######################################
# subdirs
add_subdirectory(code)

if(AC_LLAMA_BUILD_TESTS OR AC_LLAMA_BUILD_EXAMPLES)
    CPMAddPackage(
        NAME ac-test-data-llama
        VERSION 1.0.0
        GIT_REPOSITORY https://huggingface.co/alpaca-core/ac-test-data-llama
        GIT_TAG 989ab9fe8f85a706453fe1f74a324700132f5881
    )
endif()

if(AC_LLAMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(AC_LLAMA_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_AC_LLAMA_PLUGIN)
    add_subdirectory(ac-local-plugin)
endif()
