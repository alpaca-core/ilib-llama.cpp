// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Session.hpp>
#include <doctest/doctest.h>
#include <vector>
#include <deque>

using namespace ac::llama;
using TokenVec = std::vector<Token>;

// the session is a coroutine wrapper
// we can test it in isolation with a fake instance (no actual AI inference going on)
// in this case we create a coroutine which:
// * receives an initial prompt
// * for every subsequent prompt, it yields the sum of prompt tokens and the initial prompt tokens (zipped, wrapped)

// Session TestSession() {
//     return Session();
// }

// TEST_CASE("session default") {
//     Session s;
//     CHECK(!s);
// }

// TEST_CASE("session empty initial") {
//     auto s = TestSession();
//     CHECK_THROWS_WITH(s.setInitialPrompt({}), "empty");
// }

// TEST_CASE("session no push") {
//     Session s = TestSession();

//     {
//         TokenVec initialPrompt = {1, 2, 3};
//         s.setInitialPrompt(initialPrompt);
//     }

//     REQUIRE(s);

//     CHECK(s.getToken() == 4);
//     CHECK(s.getToken() == Token_Invalid);
//     CHECK(s.getToken() == 3);
//     CHECK(s.getToken() == Token_Invalid);
//     CHECK(s.getToken() == 4);
//     CHECK(s.getToken() == Token_Invalid);
//     CHECK(s.getToken() == 2);
//     CHECK(s.getToken() == Token_Invalid);
// }

// TEST_CASE("session push") {
//     Session s = TestSession();

//     {
//         TokenVec initialPrompt = {1, 2, 3};
//         s.setInitialPrompt(initialPrompt);
//     }

//     REQUIRE(s);

//     TokenVec prompt = {4, 5};
//     s.pushPrompt(prompt);
//     CHECK(s.getToken() == 5);
//     CHECK(s.getToken() == 7);
//     CHECK(s.getToken() == Token_Invalid);

//     prompt = {10, 20, 30, 40};

//     s.pushPrompt(prompt);
//     CHECK(s.getToken() == 13);

//     prompt.clear(); // safe to destroy after first getToken

//     CHECK(s.getToken() == 21);

//     SUBCASE("no replace") {
//         CHECK(s.getToken() == 32);
//         CHECK(s.getToken() == 43);
//         CHECK(s.getToken() == Token_Invalid);
//     }
//     SUBCASE("replace") {
//         prompt = {100, 200};
//         s.pushPrompt(prompt);
//         CHECK(s.getToken() == 102);
//         CHECK(s.getToken() == 203);
//         CHECK(s.getToken() == Token_Invalid);
//     }

//     CHECK(s.getToken() == 2);
//     CHECK(s.getToken() == Token_Invalid);
// }

// TEST_CASE("exceptions") {
//     Session s = TestSession();

//     SUBCASE("empty initial") {
//         CHECK_THROWS_WITH(s.setInitialPrompt({}), "empty");
//     }
//     SUBCASE("bad prompt") {
//         {
//             TokenVec initialPrompt = { 1, 2, 3 };
//             s.setInitialPrompt(initialPrompt);
//         }

//         REQUIRE(s);

//         TokenVec prompt = { 4, 0xBADF00D };
//         s.pushPrompt(prompt);
//         CHECK(s.getToken() == 5);
//         CHECK_THROWS_WITH(s.getToken(), "test exception");
//     }

//     // everything is invalid from now on
//     CHECK(s.getToken() == Token_Invalid);
//     CHECK(s.getToken() == Token_Invalid);
// }

// TEST_CASE("states") {
//     Session s = TestSession();
//     SUBCASE("no init operation") {
//         CHECK_THROWS_WITH(s.getState();, "invalid initial op");
//     }

//     SUBCASE("get state") {
//         {
//             TokenVec initialPrompt = { 1, 2, 3 };
//             s.setInitialPrompt(initialPrompt);
//         }

//         std::vector<uint8_t> expectedState = {1, 2, 3};
//         auto state = s.getState();
//         CHECK(state.size() == expectedState.size());
//         for (size_t i = 0; i < state.size(); i++)
//         {
//             CHECK(state[i] == expectedState[i]);
//         }
//     }

//     SUBCASE("set empty state") {
//         {
//             TokenVec initialPrompt = { 1, 2, 3 };
//             s.setInitialPrompt(initialPrompt);
//         }
//         CHECK_THROWS_WITH(s.setState({}), "empty state");
//     }

//     SUBCASE("set state") {
//         {
//             TokenVec initialPrompt = { 1, 2, 3 };
//             s.setInitialPrompt(initialPrompt);
//         }
//         std::vector<uint8_t> state = {0, 1, 2, 3};
//         auto res = s.setState(state);
//         CHECK(res);
//     }
// }
