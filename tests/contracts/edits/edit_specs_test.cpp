// Axis Q: the edit specs suite (tests/README.md).
//
// Contract locked: an invalid edit set is refused by name, naming the edit
// edit/edit.h says it names: an empty range, and two ranges of one neuron
// that share a step, while ranges that only meet are one neuron's history;
// what an edit does to a neuron, with Scale × 1 an exact no-op; and the
// table's layout, which edit.wgsl reads, each range placed at the run's
// origin, saturating at kToTheEnd.
//
// Inputs: a model description built here, in Gemma 3 1B's shape (26 layers
// of 6,912 neurons). The functions read nothing else of it, so no model
// file is needed.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.io.2 Always consider ill-formed input: each refusal, including a
//            kind outside the enum, is checked by its name and index.
//     E.27   Use error codes systematically: every EditError is reached.

#include <doctest/doctest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/edit/edit.h"
#include "core/model/model_description.h"

namespace {

using frizzle::edit::apply;
using frizzle::edit::check;
using frizzle::edit::Edit;
using frizzle::edit::EditError;
using frizzle::edit::EditKind;
using frizzle::edit::kMaxEdits;
using frizzle::edit::kToTheEnd;
using frizzle::edit::placed;
using frizzle::edit::table;

constexpr std::uint32_t kLayers = 26;
constexpr std::uint32_t kWidth = 6912;
constexpr std::size_t kHead = std::size_t{2} * kLayers;

frizzle::model::ModelDescription gemma_shape() {
  frizzle::model::ModelDescription model{};
  frizzle::model::LayerDescription layer{};
  layer.feed_forward_width = kWidth;
  model.layers.assign(kLayers, layer);
  return model;
}

std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }

// An edit from step 0 to the end, unless a range is given.
Edit E(std::uint32_t layer, std::uint32_t neuron, EditKind kind, float value,
       std::uint32_t from = 0, std::uint32_t until = kToTheEnd) {
  return {layer, neuron, kind, value, from, until};
}

// A layer's [first, count] in a table.
std::uint32_t first(const std::vector<std::uint32_t>& words,
                    std::uint32_t layer) {
  return words.at(std::size_t{2} * layer);
}
std::uint32_t count(const std::vector<std::uint32_t>& words,
                    std::uint32_t layer) {
  return words.at((std::size_t{2} * layer) + 1);
}

}  // namespace

TEST_CASE("a valid set passes, empty or full") {
  const auto model = gemma_shape();
  CHECK(check({}, model).ok());
  std::vector<Edit> full;
  full.reserve(kMaxEdits);
  for (std::uint32_t i = 0; i < kMaxEdits; ++i) {
    full.push_back(E(i % kLayers, i, EditKind::Scale, 2.0F));
  }
  CHECK(check(full, model).ok());
  CHECK(
      check(std::vector<Edit>{E(kLayers - 1, kWidth - 1, EditKind::Zero, 0.0F),
                              E(0, 0, EditKind::Set, -3.5F, 4, 5)},
            model)
          .ok());
}

TEST_CASE("each fault is refused by name, with the edit's index") {
  const auto model = gemma_shape();
  const auto refused = [&](std::vector<Edit> edits, EditError error,
                           std::size_t index) {
    const auto result = check(edits, model);
    CHECK(result.error == error);
    CHECK(result.index == index);
    CHECK_FALSE(result.subject.empty());
  };
  const Edit good = E(3, 10, EditKind::Set, 1.0F);
  // A byte outside the enum, as a host's malformed data would arrive.
  const auto stray = std::bit_cast<EditKind>(std::uint8_t{7});
  refused({good, E(0, 0, stray, 1.0F)}, EditError::Kind, 1);
  refused({good, E(kLayers, 0, EditKind::Set, 1.0F)}, EditError::Layer, 1);
  refused({good, E(2, kWidth, EditKind::Set, 1.0F)}, EditError::Neuron, 1);
  refused({E(1, 1, EditKind::Scale, std::numeric_limits<float>::quiet_NaN())},
          EditError::NotFinite, 0);
  refused({E(1, 1, EditKind::Set, std::numeric_limits<float>::infinity())},
          EditError::NotFinite, 0);
  refused({E(1, 1, EditKind::Zero, 0.5F)}, EditError::ZeroValue, 0);
  refused({good, E(1, 1, EditKind::Set, 1.0F, 7, 7)}, EditError::Range, 1);
  refused({E(1, 1, EditKind::Set, 1.0F, 8, 7)}, EditError::Range, 0);
  std::vector<Edit> too_many(kMaxEdits + 1, good);
  refused(too_many, EditError::TooMany, kMaxEdits);
}

TEST_CASE("an edit without its range is refused, never applied from step 0") {
  // Initialized as an edit was before ranges: from and until are 0.
  const Edit unranged{3, 10, EditKind::Set, 1.0F, {}, {}};
  const auto result = check(std::vector<Edit>{unranged}, gemma_shape());
  CHECK(result.error == EditError::Range);
  CHECK(result.index == 0);
}

TEST_CASE(
    "ranges of one neuron that share a step are refused, the later named") {
  const auto model = gemma_shape();
  const auto result =
      check(std::vector<Edit>{E(5, 7, EditKind::Set, 1.0F, 0, 10),
                              E(9, 2, EditKind::Set, 1.0F, 3, kToTheEnd),
                              E(9, 2, EditKind::Scale, 2.0F, 0, 4),
                              E(5, 7, EditKind::Zero, 0.0F, 9, kToTheEnd)},
            model);
  CHECK(result.error == EditError::Overlap);
  CHECK(result.index == 2);
  // A range inside another, given first, still names the later index.
  const auto inside =
      check(std::vector<Edit>{E(1, 1, EditKind::Set, 1.0F, 4, 6),
                              E(1, 1, EditKind::Set, 2.0F, 0, 20)},
            model);
  CHECK(inside.error == EditError::Overlap);
  CHECK(inside.index == 1);
}

TEST_CASE("ranges of one neuron that only meet are one neuron's history") {
  const auto model = gemma_shape();
  CHECK(check(std::vector<Edit>{E(5, 7, EditKind::Set, 1.0F, 0, 6),
                                E(5, 7, EditKind::Scale, 3.0F, 6, 9),
                                E(5, 7, EditKind::Zero, 0.0F, 9, kToTheEnd),
                                E(5, 8, EditKind::Set, 1.0F, 2, 4)},
              model)
            .ok());
}

TEST_CASE("an edit invalid on its own is named before any overlap") {
  const auto model = gemma_shape();
  const auto result =
      check(std::vector<Edit>{E(5, 7, EditKind::Set, 1.0F),
                              E(5, 7, EditKind::Set, 1.0F),
                              E(kLayers, 0, EditKind::Set, 1.0F),
                              E(1, kWidth, EditKind::Set, 1.0F)},
            model);
  CHECK(result.error == EditError::Layer);
  CHECK(result.index == 2);
}

TEST_CASE("apply: Set stores, Scale multiplies once, Zero stores 0") {
  CHECK(apply(EditKind::Set, 4.25F, -1.0F) == 4.25F);
  CHECK(apply(EditKind::Scale, 3.0F, 0.5F) == 1.5F);
  CHECK(bits(apply(EditKind::Zero, 0.0F, -7.0F)) == bits(0.0F));
}

TEST_CASE("Scale × 1 leaves every value's bits as they were") {
  const std::vector<float> edges{0.0F,
                                 -0.0F,
                                 std::numeric_limits<float>::denorm_min(),
                                 -std::numeric_limits<float>::denorm_min(),
                                 std::numeric_limits<float>::min(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity()};
  for (const float f : edges) {
    CHECK(bits(apply(EditKind::Scale, 1.0F, f)) == bits(f));
  }
  // Every 65,521st bit pattern, both signs, skipping NaNs: activations are
  // finite.
  constexpr std::uint64_t kStride = 65521;
  for (std::uint64_t pattern = 0; pattern <= 0xFFFF'FFFFULL;
       pattern += kStride) {
    const auto f = std::bit_cast<float>(static_cast<std::uint32_t>(pattern));
    if (f != f) continue;
    REQUIRE(bits(apply(EditKind::Scale, 1.0F, f)) == bits(f));
  }
}

TEST_CASE(
    "the table: [first, count] a layer, then edits by layer, neuron and start, "
    "placed") {
  const auto model = gemma_shape();
  constexpr std::uint32_t kOrigin = 41;  // a rendered text of 42 tokens
  const std::vector<Edit> edits{E(12, 4031, EditKind::Set, 8.0F, 3, 9),
                                E(3, 900, EditKind::Scale, -2.0F),
                                E(12, 17, EditKind::Zero, 0.0F, 5, kToTheEnd),
                                E(25, 6911, EditKind::Scale, 1.0F, 0, 1),
                                E(12, 4031, EditKind::Scale, 2.0F, 0, 3)};
  REQUIRE(check(edits, model).ok());
  const std::vector<std::uint32_t> words = table(edits, model, kOrigin);
  REQUIRE(words.size() == kHead + 5 * edits.size());
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    const bool edited = layer == 3 || layer == 12 || layer == 25;
    CHECK(count(words, layer) == (layer == 12 ? 3U : edited ? 1U : 0U));
  }
  CHECK(first(words, 3) == kHead);
  CHECK(first(words, 12) == kHead + 5);
  CHECK(first(words, 25) == kHead + 20);
  CHECK(first(words, 0) == 0);
  // [neuron, kind, value bits, first position, end position], by layer,
  // neuron, then first position; step s at position 41 + s.
  const std::vector<std::uint32_t> expected{
      900,  1, bits(-2.0F), 41, kToTheEnd,  // layer 3
      17,   2, bits(0.0F),  46, kToTheEnd,  // layer 12
      4031, 1, bits(2.0F),  41, 44,         // layer 12, steps 0 to 3
      4031, 0, bits(8.0F),  44, 50,         // layer 12, steps 3 to 9
      6911, 1, bits(1.0F),  41, 42,         // layer 25
  };
  CHECK(std::vector<std::uint32_t>(words.begin() + kHead, words.end()) ==
        expected);
}

TEST_CASE("a range past every position is placed to the end") {
  const auto model = gemma_shape();
  const std::vector<Edit> edits{E(0, 0, EditKind::Set, 1.0F, 2, kToTheEnd - 1)};
  REQUIRE(check(edits, model).ok());
  const auto words = table(edits, model, 100);
  CHECK(words.at(kHead + 3) == 102);
  CHECK(words.at(kHead + 4) == kToTheEnd);
  const auto at = placed(edits, 100);
  REQUIRE(at.size() == 1);
  CHECK(at[0].from == 102);
  CHECK(at[0].until == kToTheEnd);
  CHECK(at[0].layer == 0);
  CHECK(at[0].value == 1.0F);
}

TEST_CASE("the table of no edits is every layer's count 0") {
  const std::vector<std::uint32_t> words = table({}, gemma_shape(), 0);
  CHECK(words == std::vector<std::uint32_t>(kHead, 0));
}
