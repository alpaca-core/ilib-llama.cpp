// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Token.hpp"
#include <vector>
#include <iostream>

namespace ac::llama {
using ProbVector = std::vector<std::pair<ac::llama::Token, float>>;

ProbVector softmax(const ProbVector& logits) {
    ProbVector result;
    result.resize(logits.size());

    float max_l = logits[0].second;
    float cum_sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        float p = expf(logits[i].second - max_l);
        result[i].first = logits[i].first;
        result[i].second = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        result[i].second /= cum_sum;
    }
    return result;

    // ProbVector probs;
    // float sum = 0.0f;
    // for (const auto& logit : logits) {
    //     sum += std::exp(logit.second);
    // }
    // for (const auto& logit : logits) {
    //     probs.push_back({logit.first, std::exp(logit.second) / sum});
    // }
    // return probs;

}

class LogitComparer {
public:
    LogitComparer() = default;
    ~LogitComparer() = default;

    bool compare(const ProbVector& logit1, const ProbVector& logit2, size_t size) {
        auto l1Trim = logit1;
        l1Trim.resize(size);

        auto l2Trim = logit2;
        l2Trim.resize(size);

        float similarity = cosine_similarity(logit1, logit2);
        std::cout << "cosine similarity: " << similarity << std::endl;

        float similarityTrim = cosine_similarity(l1Trim, l2Trim);
        std::cout << "cosine similarity trim: " << similarityTrim << std::endl;

        auto p1 = softmax(logit1);
        auto p2 = softmax(logit2);

        float jsdP = jsd(p1, p2);
        std::cout << "jsd probs (< 0.01 ? "<< (jsdP < 0.01 ? "YES" : "NO") << "): " << jsdP << std::endl;

        std::cout << "\n\n" << std::endl;

        for (size_t j = 0; j < size; j++) {
            auto err = std::abs(logit1[j].second - logit2[j].second);
            std::cout   << "[" << j << "]" << (err < 0.001 ? " OK" : " MISMATCH") << " (err < 0.001)."
                        <<" t: [" << logit1[j].first  << "] vs [" << logit2[j].first << "],"
                        <<" p: [" << logit1[j].second << "] vs [" << logit2[j].second << "]"
                        << std::endl;
        }

        auto printTheRest = [&size](const ProbVector& l, const std::string& name) {
            std::cout << "Print " << name << std::endl;
            for (size_t j = size; j < l.size(); j++) {
                std::cout   << "[" << j << "]"
                            <<" t: [" << l[j].first  << "]"
                            <<" p: [" << l[j].second << "]"
                            << std::endl;
            }
        };

        printTheRest(logit1, "l1");
        printTheRest(logit2, "l2");

        return jsdP < 0.01;
    }

    float jsd(const ProbVector& logits1, const ProbVector& logits2) {
        std::unordered_map<int32_t, float> logit_map1, logit_map2;

        for (const auto& p : logits1) logit_map1[p.first] = p.second;
        for (const auto& p : logits2) logit_map2[p.first] = p.second;

        std::cout << "The probs" << std::endl;

        std::unordered_map<int32_t, float> avg_dist;
        for (const auto& [token, logit1] : logit_map1) {
            std::cout << logit1 << ", ";
            float logit2 = logit_map2.count(token) ? logit_map2[token] : 0.0f;
            avg_dist[token] = (logit1 + logit2) / 2.0f;
        }

        std::cout << std::endl;

        for (const auto& [token, logit2] : logit_map2) {
            std::cout << logit2 << ", ";
            if (!logit_map1.count(token)) {
                avg_dist[token] = logit2 / 2.0f;
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

        return (kl_divergence(logit_map1, avg_dist) + kl_divergence(logit_map2, avg_dist)) / 2.0f;
    }


    // compute cosine similarity only for the union of the two sets of tokens
    float cosine_similarity(const ProbVector& logits1, const ProbVector& logits2) {
        std::unordered_map<int32_t, float> logit_map1, logit_map2;

        for (const auto& p : logits1) logit_map1[p.first] = p.second;
        for (const auto& p : logits2) logit_map2[p.first] = p.second;

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
};
}
