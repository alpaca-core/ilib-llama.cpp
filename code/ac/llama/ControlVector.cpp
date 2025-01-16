// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ControlVector.hpp"
#include "Model.hpp"
#include "Logging.hpp"

#include <llama.h>
#include <gguf.h>

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
        LLAMA_LOG(Error, "Failed to load control vector file from ", loadInfo.ggufPath);
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        LLAMA_LOG(Warning, "No direction tensors found in ", loadInfo.ggufPath);
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
        if (layer_idx <= 0) {
            LLAMA_LOG(Error, "Invalid/Unparsable", (layer_idx == 0 ? " (zero)" : " (negative)"),
                                                " direction tensor layer index in", loadInfo.ggufPath);
            result.nEmbd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            LLAMA_LOG(Error, "Invalid (non-F32) direction tensor type in", loadInfo.ggufPath);
            result.nEmbd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            LLAMA_LOG(Error, "Invalid (non-1D) direction tensor shape in", loadInfo.ggufPath);
            result.nEmbd = -1;
            break;
        }

        if (result.nEmbd == -1) {
            result.nEmbd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.nEmbd) {
            LLAMA_LOG(Error, "Direction tensor in ", loadInfo.ggufPath, " does not match previous dimensions");
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
        LLAMA_LOG(Warning, "skipping ", loadInfo.ggufPath, " due to invalid direction tensors ");
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}
}

ControlVector::ControlVector(const Model& model, const std::vector<LoadInfo>& infos, int lStart, int lEnd)
    : controlVectorLayerStart(lStart <= 0 ? 1 : lStart)
    , controlVectorLayerEnd(lEnd <= 0 ? llama_model_n_layer(model.lmodel()) : lEnd)
{
    for (const auto & info : infos) {
        auto cur = loadControlVector(info);

        if (cur.nEmbd == -1) {
            nEmbd = -1;
            break;
        }
        if (nEmbd != -1 && nEmbd != cur.nEmbd) {
            LLAMA_LOG(Error, "Control vectors in ", info.ggufPath," does not match previous dimensions");
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
        LLAMA_LOG(Error, "No valid control vectors files passed");
        data.clear();
    }
}

} // namespace ac::llama
