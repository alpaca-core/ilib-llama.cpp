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


// We apply 2 step comparison
// 1. Compare the euclidean distance of the logits
//  - If the distance is less than 1% of the max distance, we consider them equal
// 2. Compare the Jensen-Shannon divergence of the probabilities
//  - If the divergence is less than the treshold, we consider them equal
bool LogitComparer::compare(const TokenDataVector& data1, const TokenDataVector& data2) {
    const int32_t minSize = std::min(data1.size(), data2.size());
    float distance1 = euclidean_distance(data1, minSize);
    float distance2 = euclidean_distance(data2, minSize);

    float relative_threshold = 0.03f; // 3% difference allowed
    float res = std::fabs(distance1 - distance2) / std::max(distance1, distance2);
    if (res > relative_threshold) {
        return false;
    }

    std::unordered_map<int32_t, float> prob_map, prob_map2;

    for (const auto& p : data1) prob_map[p.token] = p.prob;
    for (const auto& p : data2) prob_map2[p.token] = p.prob;

    float jsdP = jsd(prob_map, prob_map2);
    std::cout << "jsd probs (< 0.01 ? "<< (jsdP < 0.01 ? "YES" : "NO") << "): " << jsdP << std::endl;

    return jsdP < 0.01;
}

float LogitComparer::jsd(const std::unordered_map<Token, float>& probs1, const std::unordered_map<Token, float>& probs2) {
    std::cout << "The probs" << std::endl;

    std::unordered_map<Token, float> avg_dist;
    for (const auto& [token, p] : probs1) {
        if (probs2.count(token)) {
            std::cout << "[" << token << "]" << p << " " << probs2.at(token) << ", ";
            avg_dist[token] = (p + probs2.at(token)) / 2.0f;
        }
    }

    std::cout << std::endl;

    auto kl_divergence = [](const std::unordered_map<Token, float>& P, const std::unordered_map<Token, float>& Q) {
        float kl = 0.0f;
        for (const auto& [token, p] : P) {
            if (p > 0.0f && Q.count(token) && Q.at(token) > 0.0f) {
                kl += p * std::log(p / Q.at(token));
            }
        }
        return kl;
    };

    auto div1 = kl_divergence(probs1, avg_dist);
    auto div2 = kl_divergence(probs2, avg_dist);

    return (div1 + div2) / 2.0f;
}

float LogitComparer::euclidean_distance(const TokenDataVector& logits1, int32_t count) {
    float distance = 0.0f;
    for (int32_t i = 0; i < count; ++i) {
        distance += logits1[i].logit * logits1[i].logit;
    }

    // To achieve total result, we need to take the square root of the sum,
    // bit since we don't need it to be accurate, we can skip it
    return distance;
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
