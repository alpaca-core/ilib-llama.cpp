# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
icm_add_lib(ac-llama AC_LLAMA)
add_library(ac::llama ALIAS ac-llama)
target_include_directories(ac-llama INTERFACE .)
target_link_libraries(ac-llama
    PRIVATE
        llama
        ac::jalog
    PUBLIC
        ac::astl
        ac::local
        jinja2cpp
)
target_sources(ac-llama
    INTERFACE FILE_SET HEADERS FILES
        ac/llama/export.h
        ac/llama/Init.hpp
        ac/llama/Model.hpp
        ac/llama/ChatFormat.hpp
        ac/llama/Vocab.hpp
        ac/llama/Sampler.hpp
        ac/llama/Instance.hpp
        ac/llama/InstanceEmbedding.hpp
        ac/llama/Session.hpp
        ac/llama/AntipromptManager.hpp
        ac/llama/IncrementalStringFinder.hpp
        ac/llama/ControlVector.hpp
        ac/llama/LoraAdapter.hpp
        ac/llama/LogitComparer.hpp
        ac/llama/ResourceCache.hpp
    PRIVATE
        ac/llama/Logging.hpp
        ac/llama/Logging.cpp
        ac/llama/Init.cpp
        ac/llama/Model.cpp
        ac/llama/ChatFormat.cpp
        ac/llama/Vocab.cpp
        ac/llama/Sampler.cpp
        ac/llama/Instance.cpp
        ac/llama/InstanceEmbedding.cpp
        ac/llama/Session.cpp
        ac/llama/AntipromptManager.cpp
        ac/llama/IncrementalStringFinder.cpp
        ac/llama/ControlVector.cpp
        ac/llama/LoraAdapter.cpp
        ac/llama/LogitComparer.cpp
)
