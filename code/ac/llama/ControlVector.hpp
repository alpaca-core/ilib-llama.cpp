// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <vector>

namespace ac::llama {
class Model;

/// The `ControlVector` provides a mechanism for configuring and using
/// control vectors within a llama.cpp-based model. This class is typically used
/// to adjust model behavior by loading control vector data and specifying the
/// range of layers the control vector should apply to.
class AC_LLAMA_EXPORT ControlVector {
public:
    struct LoadInfo {
        std::string ggufPath; // Path to the GGUF file containing the control vector data.
        float strength; // The strength of the control vector's influence on the model.
    };

    ControlVector(const Model& model, const std::vector<LoadInfo>& infos, int lStart = 0, int lEnd = 0);

    std::vector<float> data;
    int nEmbd = -1;
    int controlVectorLayerStart = 0;
    int controlVectorLayerEnd = 0;
};

} // namespace ac::llama
