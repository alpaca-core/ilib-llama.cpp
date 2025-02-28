// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "LogitComparer.hpp"
#include <cmath>

namespace ac::llama {

// We apply 3 step comparison
// 1. Compare the euclidean distance of the logits
//  - If the distance is less than 2% of the max distance, we consider them equal
// 2. Compare the matching tokens
//  - If at least 80% of the tokens are the same, we consider them equal
// 3. Compare the Jensen-Shannon divergence of the probabilities
//  - If the divergence is less than the treshold, we consider them equal
bool LogitComparer::compare(const TokenDataVector& data1, const TokenDataVector& data2) {
    const auto minSize = std::min(data1.size(), data2.size());
    float distance1 = euclidean_distance_sq({data1.data(), minSize});
    float distance2 = euclidean_distance_sq({data2.data(), minSize});

    float relative_threshold = 0.02f; // 2% difference allowed
    float res = std::fabs(distance1 - distance2) / std::max(distance1, distance2);
    if (res > relative_threshold) {
        return false;
    }

    std::unordered_map<int32_t, float> prob_map, prob_map2;

    for (const auto& p : data1) prob_map[p.token] = p.prob;
    for (const auto& p : data2) prob_map2[p.token] = p.prob;

    // Check if at least 80% of the tokens are the same
    float matchingTokens = 0;
    for (const auto& p : data1) {
        if (prob_map2.count(p.token)) {
            matchingTokens++;
        }
    }

    float matchingPercentage = matchingTokens / minSize;
    if (matchingPercentage < 0.8f) {
        return false;
    }

    return jsd(prob_map, prob_map2) < 0.01;
}

float LogitComparer::jsd(const std::unordered_map<Token, float>& probs1, const std::unordered_map<Token, float>& probs2) {
    std::unordered_map<Token, float> avg_dist;
    for (const auto& [token, p] : probs1) {
        if (probs2.count(token)) {
            avg_dist[token] = (p + probs2.at(token)) / 2.0f;
        }
    }

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

float LogitComparer::euclidean_distance_sq(std::span<const TokenData> tokens) {
    float distance = 0.0f;
    for (auto& t : tokens) {
        distance += t.logit * t.logit;
    }

    // To achieve total result, we need to take the square root of the sum,
    // but since we don't need it to be accurate, we can skip it
    return distance;
}

}
