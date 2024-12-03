// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Token.hpp"
#include <span>
#include <utility>
#include <exception>
#include <coroutine>
#include <assert.h>

namespace ac::llama {

class Session {
public:
    struct Prompt {}; // sentinel for co_await-int prompts
    struct StartGeneration {}; // sentinel for co_await-ing the start of generation

    struct SessionOpData {
        enum OpType {
            Prompt,
            GetState,

            Count
        };
        OpType type = Count;

        std::span<const Token> pendingPrompt;
        std::string cachePath;
    };

    struct SessionResult {
        enum class Type {
            Token,
            State,
            Invalid
        };

        Type type = Type::Invalid;
        Token token = Token_Invalid;
        std::vector<uint8_t> state;
    };

    class promise_type {
        SessionResult m_value;

        SessionOpData m_pendingOpData;
        // SessionOpData::OpType m_pendingOp = SessionOpData::Count;
        // std::span<const Token> m_pendingPrompt;
        // std::string m_cachePath;

        std::exception_ptr m_exception;
    public:
        Session get_return_object() noexcept {
            return Session{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        SessionResult value() const noexcept { return m_value; }

        // suspend until the initial prompt is set
        std::suspend_always initial_suspend() noexcept { return {}; }

        // keep the coroutine alive to make the last yield value available
        std::suspend_always final_suspend() noexcept { return {}; }

        std::suspend_always yield_value(Token value) noexcept {
            m_value.type = SessionResult::Type::Token;
            m_value.token = value;
            return {};
        }

        std::suspend_always yield_value(std::vector<uint8_t> stateData) noexcept {
            m_value.type = SessionResult::Type::State;
            m_value.state = stateData;
            return {};
        }

        void return_void() noexcept {
            m_value.type = SessionResult::Type::Invalid;
        }
        void unhandled_exception() noexcept {
            // why store the exception here instead of simply rethrowing?
            // because clang is stupid, that's why
            // clang's ridiculous handling of coroutines causes local coroutine variables to be destroyed twice if we
            // just rethrow here
            m_exception = std::current_exception();
        }

        struct Awaiter {
            promise_type& self;
            bool await_ready() noexcept { return true; }
            SessionOpData await_resume() noexcept {
                // clear pending after returning it
                return std::exchange(self.m_pendingOpData, SessionOpData());
            }
            void await_suspend(std::coroutine_handle<>) noexcept {}
        };

        void setOpType(SessionOpData::OpType type) {
            m_pendingOpData.type = type;
        }

        void setPrompt(std::span<const Token> prompt) {
            // assert(m_pendingOpData.type == SessionOpData::Count);
            m_pendingOpData.type = SessionOpData::Prompt;
            m_pendingOpData.pendingPrompt = prompt;
        }

        void getState() {
            m_pendingOpData.type = SessionOpData::GetState;
        }

        Awaiter await_transform(Prompt) noexcept { return Awaiter{*this}; }

        std::suspend_always await_transform(StartGeneration) noexcept { return {}; }

        void rethrowIfException() {
            if (m_exception) {
                std::rethrow_exception(m_exception);
            }
        }
    };

    using Handle = std::coroutine_handle<promise_type>;

    Session() = default;
    Session(Handle handle) : m_handle(handle) {}

    Session(Session&& other) noexcept : m_handle(std::exchange(other.m_handle, nullptr)) {}
    Session& operator=(Session&& other) noexcept {
        if (this == &other) return *this;
        if (m_handle) {
            m_handle.destroy();
        }
        m_handle = std::exchange(other.m_handle, nullptr);
        return *this;
    }

    ~Session() {
        if (m_handle) {
            m_handle.destroy();
        }
    }

    void reset() {
        if (m_handle) {
            m_handle.destroy();
        }
        m_handle = nullptr;
    }

    explicit operator bool() const noexcept { return !!m_handle; }

    // the provided span must remain valid until the next call to getToken or pushPrompt with another span
    void pushPrompt(std::span<const Token> prompt) {
        m_handle.promise().setPrompt(prompt);
    }

    void setInitialPrompt(std::span<const Token> prompt) {
        m_handle.promise().setPrompt(prompt);
        m_handle.resume();
        m_handle.promise().rethrowIfException();
    }

    Token getToken() {
        if (m_handle.done()) return Token_Invalid;
        m_handle.promise().setOpType(SessionOpData::Prompt);
        m_handle.resume();
        m_handle.promise().rethrowIfException();
        assert(m_handle.promise().value().type == SessionResult::Type::Token);
        return std::move(m_handle.promise().value().token);
    }

    std::vector<uint8_t> getState() {
        m_handle.promise().getState();
        m_handle.resume();
        assert(m_handle.promise().value().type == SessionResult::Type::State);
        return std::move(m_handle.promise().value().state);
    }

private:
    Handle m_handle;
};

} // namespace ac::llama
