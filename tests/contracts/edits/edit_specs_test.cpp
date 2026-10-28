// Axis Q: the edit specs suite (tests/README.md).
//
// Contract locked: an invalid edit set is refused by name, naming the edit
// edit/edit.h says it names; what an edit does to a neuron, with Scale × 1
// an exact no-op; and the table's layout, which edit.wgsl reads.
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
    full.push_back({i % kLayers, i, EditKind::Scale, 2.0F});
  }
  CHECK(check(full, model).ok());
  CHECK(check(std::vector<Edit>{{kLayers - 1, kWidth - 1, EditKind::Zero, 0.0F},
                                {0, 0, EditKind::Set, -3.5F}},
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
  const Edit good{3, 10, EditKind::Set, 1.0F};
  // A byte outside the enum, as a host's malformed data would arrive.
  const auto stray = std::bit_cast<EditKind>(std::uint8_t{7});
  refused({good, {0, 0, stray, 1.0F}}, EditError::Kind, 1);
  refused({good, {kLayers, 0, EditKind::Set, 1.0F}}, EditError::Layer, 1);
  refused({good, {2, kWidth, EditKind::Set, 1.0F}}, EditError::Neuron, 1);
  refused({{1, 1, EditKind::Scale, std::numeric_limits<float>::quiet_NaN()}},
          EditError::NotFinite, 0);
  refused({{1, 1, EditKind::Set, std::numeric_limits<float>::infinity()}},
          EditError::NotFinite, 0);
  refused({{1, 1, EditKind::Zero, 0.5F}}, EditError::ZeroValue, 0);
  std::vector<Edit> too_many(kMaxEdits + 1, good);
  refused(too_many, EditError::TooMany, kMaxEdits);
}

TEST_CASE("a repeat names the first edit that repeats an earlier one") {
  const auto model = gemma_shape();
  const auto result = check(std::vector<Edit>{{5, 7, EditKind::Set, 1.0F},
                                              {9, 2, EditKind::Set, 1.0F},
                                              {9, 2, EditKind::Scale, 2.0F},
                                              {5, 7, EditKind::Zero, 0.0F}},
                            model);
  CHECK(result.error == EditError::Duplicate);
  CHECK(result.index == 2);
}

TEST_CASE("an edit invalid on its own is named before any repeat") {
  const auto model = gemma_shape();
  const auto result = check(std::vector<Edit>{{5, 7, EditKind::Set, 1.0F},
                                              {5, 7, EditKind::Set, 1.0F},
                                              {kLayers, 0, EditKind::Set, 1.0F},
                                              {1, kWidth, EditKind::Set, 1.0F}},
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

TEST_CASE("the table: [first, count] a layer, then edits by layer and neuron") {
  const auto model = gemma_shape();
  const std::vector<Edit> edits{{12, 4031, EditKind::Set, 8.0F},
                                {3, 900, EditKind::Scale, -2.0F},
                                {12, 17, EditKind::Zero, 0.0F},
                                {25, 6911, EditKind::Scale, 1.0F}};
  REQUIRE(check(edits, model).ok());
  const std::vector<std::uint32_t> words = table(edits, model);
  REQUIRE(words.size() == kHead + 3 * edits.size());
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    const bool edited = layer == 3 || layer == 12 || layer == 25;
    CHECK(count(words, layer) == (layer == 12 ? 2U : edited ? 1U : 0U));
  }
  CHECK(first(words, 3) == kHead);
  CHECK(first(words, 12) == kHead + 3);
  CHECK(first(words, 25) == kHead + 9);
  CHECK(first(words, 0) == 0);
  // Edits as [neuron, kind, value bits], by layer, then neuron.
  const std::vector<std::uint32_t> expected{
      900,  1, bits(-2.0F),  // layer 3
      17,   2, bits(0.0F),   // layer 12
      4031, 0, bits(8.0F),   // layer 12
      6911, 1, bits(1.0F),   // layer 25
  };
  CHECK(std::vector<std::uint32_t>(words.begin() + kHead, words.end()) ==
        expected);
}

TEST_CASE("the table of no edits is every layer's count 0") {
  const std::vector<std::uint32_t> words = table({}, gemma_shape());
  CHECK(words == std::vector<std::uint32_t>(kHead, 0));
}
