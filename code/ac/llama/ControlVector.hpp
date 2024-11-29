// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <vector>

namespace ac::llama {
class Model;

class AC_LLAMA_EXPORT ControlVector {
public:
    struct LoadInfo {
        std::string ggufPath;
        float strength;
    };

    ControlVector(Model* model, int lStart, int lEnd, const std::vector<LoadInfo>& infos);

    std::vector<float> data;
    int nEmbd = -1;
    int controlVectorLayerStart = 0;
    int controlVectorLayerEnd = 0;
};

} // namespace ac::llama
