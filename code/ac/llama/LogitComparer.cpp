// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "LogitComparer.hpp"

#include <unordered_map>


namespace ac::llama {

TokenDataVector softmax(const TokenDataVector& logits) {
    TokenDataVector result;
    result.resize(logits.size());

    float max_l = logits[0].logit;
    float cum_sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        float p = expf(logits[i].logit - max_l);
        result[i].token = logits[i].token;
        result[i].logit = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        result[i].logit /= cum_sum;
    }
    return result;
}

bool LogitComparer::compare(const TokenDataVector& logit1, const TokenDataVector& logit2, size_t size) {
    float jsdP = jsd(logit1, logit2);
    std::cout << "jsd probs (< 0.01 ? "<< (jsdP < 0.01 ? "YES" : "NO") << "): " << jsdP << std::endl;

    return jsdP < 0.01;
}

float LogitComparer::jsd(const TokenDataVector& logits1, const TokenDataVector& logits2) {
    std::unordered_map<int32_t, float> logit_map1, logit_map2;

    for (const auto& p : logits1) logit_map1[p.token] = p.prob;
    for (const auto& p : logits2) logit_map2[p.token] = p.prob;

    std::cout << "The probs" << std::endl;

    std::unordered_map<int32_t, float> avg_dist;
    for (const auto& [token, logit1] : logit_map1) {
        if (logit_map2.count(token)) {
            std::cout << "[" << token << "]" << logit1 << " " << logit_map2[token] << ", ";
            float logit2 = logit_map2.count(token) ? logit_map2[token] : 0.0f;
            avg_dist[token] = (logit1 + logit2) / 2.0f;
        }
    }

    std::cout << std::endl;

    auto kl_divergence = [](const std::unordered_map<int32_t, float>& P, const std::unordered_map<int32_t, float>& Q) {
        float kl = 0.0f;
        for (const auto& [token, p] : P) {
            if (p > 0.0f && Q.count(token) && Q.at(token) > 0.0f) {
                kl += p * std::log(p / Q.at(token));
            }
        }
        return kl;
    };

    auto div1 = kl_divergence(logit_map1, avg_dist);
    auto div2 = kl_divergence(logit_map2, avg_dist);

    return (div1 + div2) / 2.0f;
}


float LogitComparer::cosine_similarity(const TokenDataVector& logits1, const TokenDataVector& logits2) {
    std::unordered_map<int32_t, float> logit_map1, logit_map2;

    for (const auto& p : logits1) logit_map1[p.token] = p.logit;
    for (const auto& p : logits2) logit_map2[p.token] = p.logit;

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    // Compute norms for logits1
    for (const auto& [token, logit] : logit_map1) {
        norm1 += logit * logit;
        if (logit_map2.count(token)) {
            dot_product += logit * logit_map2[token];
        }
    }

    // Compute norm for logits2
    for (const auto& [token, logit] : logit_map2) {
        //if (logit_map1.count(token)) {
            norm2 += logit * logit;
        //}
    }

    // Prevent division by zero
    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;

    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

}
