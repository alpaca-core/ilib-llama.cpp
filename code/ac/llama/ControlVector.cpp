// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ControlVector.hpp"
#include "Model.hpp"

#include <llama.h>

namespace ac::llama {
namespace {
struct ControlVectorLoadResult {
    int nEmbd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / nEmbd
    std::vector<float> data;
};

ControlVectorLoadResult loadControlVector(const ControlVector::LoadInfo& loadInfo) {
    ControlVectorLoadResult result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(loadInfo.ggufPath.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        // LOG_ERR("%s: failed to load control vector file from %s\n", __func__, loadInfo.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        // LOG_WRN("%s: no direction tensors found in %s\n", __func__, loadInfo.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            // LOG_ERR("%s: invalid/unparsable direction tensor layer index in %s\n", __func__, loadInfo.fname.c_str());
            result.nEmbd = -1;
            break;
        } else if (layer_idx == 0) {
            // LOG_ERR("%s: invalid (zero) direction tensor layer index in %s\n", __func__, loadInfo.fname.c_str());
            result.nEmbd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            // LOG_ERR("%s: invalid (non-F32) direction tensor type in %s\n", __func__, loadInfo.fname.c_str());
            result.nEmbd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            // LOG_ERR("%s: invalid (non-1D) direction tensor shape in %s\n", __func__, loadInfo.fname.c_str());
            result.nEmbd = -1;
            break;
        }

        if (result.nEmbd == -1) {
            result.nEmbd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.nEmbd) {
            // LOG_ERR("%s: direction tensor in %s does not match previous dimensions\n", __func__, loadInfo.fname.c_str());
            result.nEmbd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.nEmbd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.nEmbd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.nEmbd; j++) {
            dst[j] += src[j] * loadInfo.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.nEmbd == -1) {
        // LOG_WRN("%s: skipping %s due to invalid direction tensors\n", __func__, loadInfo.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}
}

ControlVector::ControlVector(Model* model, int lStart, int lEnd, const std::vector<LoadInfo>& infos)
    : controlVectorLayerStart(lStart <= 0 ? 1 : lStart)
    , controlVectorLayerEnd(lEnd <= 0 ? llama_n_layer(model->lmodel()) : lEnd)
{
    for (const auto & info : infos) {
        auto cur = loadControlVector(info);

        if (cur.nEmbd == -1) {
            nEmbd = -1;
            break;
        }
        if (nEmbd != -1 && nEmbd != cur.nEmbd) {
            // LOG_ERR("%s: control vectors in %s does not match previous dimensions\n", __func__, info.fname.c_str());
            nEmbd = -1;
            break;
        }

        if (nEmbd == -1) {
            nEmbd = cur.nEmbd;
            data = std::move(cur.data);
        } else {
            data.resize(std::max(data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                data[i] += cur.data[i];
            }
        }
    }

    if (nEmbd == -1) {
        // LOG_ERR("%s: no valid control vector files passed\n", __func__);
        data.clear();
    }
}

} // namespace ac::llama
