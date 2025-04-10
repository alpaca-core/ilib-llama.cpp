// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "AntipromptManager.hpp"

namespace ac::llama {

void AntipromptManager::addAntiprompt(std::string_view antiprompt) {
    m_antiprompts.push_back(std::string(antiprompt));
}

std::string AntipromptManager::feedGeneratedText(std::string_view text) {
    std::vector<std::pair<std::string, size_t>> matchedAntiprompts;
    for (auto& ap : m_antiprompts) {
        int found = ap.feedText(text);
        if (found > 0) {
            auto res = found == 0 ?
                ap.getString():
                ap.getString() + std::string(text.substr(found, text.length()));
            matchedAntiprompts.push_back({res, found});
        }
    }
    if (!matchedAntiprompts.empty()) {
        reset();
        std::sort(matchedAntiprompts.begin(), matchedAntiprompts.end());
        auto& [res, found] = matchedAntiprompts.front();
        return res;
    }

    return {};
}

void AntipromptManager::reset() {
    for (auto& ap : m_antiprompts) {
        ap.reset();
    }
}

void AntipromptManager::clear() {
    m_antiprompts.clear();
}

bool AntipromptManager::hasRunningAntiprompts() {
    for (auto& ap : m_antiprompts) {
        if (ap.getCurrentPos() > 0) {
            return true;
        }
    }

    return false;
}

} // namespace ac::llama
