// Axis G: the cache suite (tests/README.md).
//
// Contract locked (cache/prefix.h): how much of the cache a run reuses, by
// its tokens and by the edits in force at each position:
//   - the longest common prefix of two token sequences;
//   - the first position where two placed edit sets differ: an edit in one
//     set alone differs at its start; a change of kind or value bits
//     (-0 apart from 0) differs where the two overlap first; ranges split
//     or ordered differently but holding the same edits do not differ; a
//     set compared with itself, or two empty sets, never differ.
//
// Not locked: that the engine rewinds to it and keeps the edits with the
// cache (runtime/engine.h), which the replay scenario runs on the model.
// Plain values, no device.

#include "core/cache/prefix.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "core/edit/edit.h"

namespace {

using frizzle::cache::first_edit_difference;
using frizzle::cache::longest_common_prefix;
using frizzle::edit::Edit;
using frizzle::edit::EditKind;
using frizzle::edit::kToTheEnd;
using frizzle::tokenizer::TokenId;

// Placed edits: from and until are positions.
Edit at(std::uint32_t layer, std::uint32_t neuron, float value,
        std::uint32_t from, std::uint32_t until = kToTheEnd,
        EditKind kind = EditKind::Set) {
  return {layer, neuron, kind, value, from, until};
}

std::vector<TokenId> ids(std::initializer_list<std::uint32_t> values) {
  std::vector<TokenId> out;
  for (const auto v : values) out.push_back(TokenId{v});
  return out;
}

}  // namespace

TEST_CASE("the longest common prefix of two token sequences") {
  const auto a = ids({2, 9, 9, 4, 7});
  CHECK(longest_common_prefix(a, ids({2, 9, 9, 5})) == 3);
  CHECK(longest_common_prefix(a, a) == 5);
  CHECK(longest_common_prefix(a, ids({2, 9})) == 2);
  CHECK(longest_common_prefix(a, {}) == 0);
}

TEST_CASE("no edits, or the same edits, never differ") {
  CHECK(first_edit_difference({}, {}) == kToTheEnd);
  const std::vector<Edit> set{at(3, 5, 8.0F, 40), at(1, 2, 0.5F, 41, 60)};
  CHECK(first_edit_difference(set, set) == kToTheEnd);
}

TEST_CASE("an edit in one set alone differs at its start") {
  const std::vector<Edit> some{at(3, 5, 8.0F, 40)};
  CHECK(first_edit_difference({}, some) == 40);
  CHECK(first_edit_difference(some, {}) == 40);
  const std::vector<Edit> two{at(3, 5, 8.0F, 40), at(7, 1, 1.0F, 52, 55)};
  CHECK(first_edit_difference(some, two) == 52);
}

TEST_CASE("a range that ends earlier or later differs where one stops") {
  const std::vector<Edit> longer{at(3, 5, 8.0F, 40, 70)};
  const std::vector<Edit> shorter{at(3, 5, 8.0F, 40, 55)};
  CHECK(first_edit_difference(longer, shorter) == 55);
  const std::vector<Edit> later{at(3, 5, 8.0F, 44, 70)};
  CHECK(first_edit_difference(longer, later) == 40);
}

TEST_CASE("a change of kind or value differs where the two overlap first") {
  const std::vector<Edit> cached{at(3, 5, 8.0F, 40, 50), at(3, 5, 2.0F, 50)};
  const std::vector<Edit> value{at(3, 5, 8.0F, 40, 50), at(3, 5, 3.0F, 50)};
  CHECK(first_edit_difference(cached, value) == 50);
  const std::vector<Edit> kind{at(3, 5, 8.0F, 40, 50),
                               at(3, 5, 2.0F, 50, kToTheEnd, EditKind::Scale)};
  CHECK(first_edit_difference(cached, kind) == 50);
  const std::vector<Edit> zero{at(0, 0, 0.0F, 10)};
  const std::vector<Edit> negative_zero{at(0, 0, -0.0F, 10)};
  CHECK(first_edit_difference(zero, negative_zero) == 10);
}

TEST_CASE(
    "ranges split or ordered otherwise, holding the same edits, do not "
    "differ") {
  const std::vector<Edit> whole{at(3, 5, 8.0F, 40, 90), at(1, 1, 1.0F, 41)};
  const std::vector<Edit> split{at(1, 1, 1.0F, 41, 60), at(3, 5, 8.0F, 61, 90),
                                at(3, 5, 8.0F, 40, 61), at(1, 1, 1.0F, 60)};
  CHECK(first_edit_difference(whole, split) == kToTheEnd);
}

TEST_CASE("the least difference over every neuron is the answer") {
  const std::vector<Edit> cached{at(2, 2, 1.0F, 30), at(9, 9, 1.0F, 45)};
  const std::vector<Edit> incoming{at(2, 2, 1.0F, 30, 80), at(9, 9, 5.0F, 45)};
  CHECK(first_edit_difference(cached, incoming) == 45);
}
