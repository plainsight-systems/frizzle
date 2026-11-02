// Axis V: the kernels suite (tests/README.md), the attention weights launch.
//
// Contract locked (capture/weights.h): one layer's weights launch, alone on
// the GPU, in the step that ends the prompt (logits 1, fed 0), turns each
// query head's scores into exp2(s − m) / l in place, with the head's m and
// l from the stats slot, for the keys the step's last position read, and
// writes 0 over the rest of the head's keys; each weight within
// (8 + 3|s − m|) × 2⁻²⁴ of the f64 quotient of the same f32 inputs,
// relative to it, or 2⁻¹²⁶ absolutely below the smallest normal. It never
// writes the stats or anything outside the weights slot, and writes
// nothing in a decode step or a prefill step that does not end the prompt.
//
// What it does not check, and what does:
//   - That the scores, m and l are the attention kernel's: the attention
//     capture's test (attention_capture_test.cpp).
//   - That the graph runs it after the attention's combine: no test of its
//     own; the replay scenario's inspection compares whole stores.
//
// Inputs, all made here: two heads of 128 keys; a weights slot at word 5
// and a stats slot after it at an unaligned offset; scores in log2 units
// from a fixed sequence, from 25 below the maximum to it, with m their
// maximum and l their sum of exp2(s − m), both computed here in f32 (the
// launch takes them as given); the slot's words past the live keys left
// as a pattern, as the attention kernel leaves them. No model file.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.7    State postconditions: every word of the store is compared.

#include "core/capture/weights.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/capture/capture.h"
#include "core/kernels/interface.h"
#include "support/scratch_program.h"

namespace {

namespace frz = frizzle;
using frz::testing::ScratchProgram;

constexpr std::uint32_t kHeads = 2;
constexpr std::uint32_t kKeys = 128;
constexpr std::uint32_t kWeights = 5;
constexpr std::uint32_t kStats = kWeights + (kHeads * kKeys) + 3;
constexpr std::uint32_t kStoreWords = kStats + (2 * kHeads) + 6;

frz::capture::LayerSlots slots() {
  frz::capture::LayerSlots s{};
  s.weights = {kWeights, kHeads * kKeys};
  s.stats = {kStats, 2 * kHeads};
  s.keys = kKeys;
  s.query_heads = kHeads;
  return s;
}

ScratchProgram program() {
  const std::array<std::uint64_t, 1> sizes = {kStoreWords * 4ULL};
  return ScratchProgram(sizes,
                        [](std::span<const frz::residency::BufferRange> r) {
                          return std::vector<frz::kernels::Launch>{
                              frz::capture::weights_launch(r[0], slots())};
                        });
}

// The keys the step's last position read (capture/weights.h).
std::uint32_t count_for(std::uint32_t last) {
  return std::min(last + 1, kKeys);
}

// A store holding `count` scores a head, their m and l, and a pattern
// everywhere else.
std::vector<std::uint32_t> store_with(std::uint32_t count) {
  std::vector<std::uint32_t> s(kStoreWords);
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = 0xA5A50000U + static_cast<std::uint32_t>(i);
  }
  std::uint32_t state = 12345;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    float m = 0.0F;
    std::vector<float> scores(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      state = (state * 1664525U) + 1013904223U;
      scores[i] =
          4.0F - (25.0F * static_cast<float>(state >> 8U) / 16777216.0F);
      m = i == 0 ? scores[i] : std::max(m, scores[i]);
    }
    float l = 0.0F;
    for (const float v : scores) l += std::exp2(v - m);
    for (std::uint32_t i = 0; i < count; ++i) {
      s[kWeights + (h * kKeys) + i] = std::bit_cast<std::uint32_t>(scores[i]);
    }
    s[kStats + (2 * h)] = std::bit_cast<std::uint32_t>(m);
    s[kStats + (2 * h) + 1] = std::bit_cast<std::uint32_t>(l);
  }
  return s;
}

frz::kernels::Step step(std::uint32_t last, std::uint32_t tokens, bool logits,
                        bool fed) {
  frz::kernels::Step s{};
  s.position = last + 1 - tokens;
  s.tokens = tokens;
  s.logits = logits ? 1 : 0;
  s.fed = fed ? 1 : 0;
  return s;
}

// The weights and zeros the launch must leave, given the store before.
void weights_as_stated(std::span<const std::uint32_t> got,
                       std::span<const std::uint32_t> before,
                       std::uint32_t count) {
  REQUIRE(got.size() == before.size());
  std::size_t wrong = 0;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const double m = std::bit_cast<float>(before[kStats + (2 * h)]);
    const double l = std::bit_cast<float>(before[kStats + (2 * h) + 1]);
    for (std::uint32_t i = 0; i < kKeys; ++i) {
      const std::size_t at = kWeights + (h * kKeys) + i;
      if (i >= count) {
        if (got[at] != 0U) ++wrong;  // +0.0, exactly
        continue;
      }
      const double s = std::bit_cast<float>(before[at]);
      const double want = std::exp2(s - m) / l;
      const double w = std::bit_cast<float>(got[at]);
      const double margin =
          want < 0x1p-126 ? 0x1p-126
                          : (8.0 + (3.0 * std::fabs(s - m))) * 0x1p-24 * want;
      if (!(std::fabs(w - want) <= margin)) {  // fails on NaN
        if (wrong == 0) MESSAGE("head " << h << " key " << i << ": " << w);
        ++wrong;
      }
    }
  }
  CHECK(wrong == 0);
  // Outside the weights slot, every word as it was: the stats too.
  std::size_t touched = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const bool in_weights = i >= kWeights && i < kWeights + (kHeads * kKeys);
    if (!in_weights && got[i] != before[i]) ++touched;
  }
  CHECK(touched == 0);
}

}  // namespace

TEST_CASE("the step that ends the prompt turns scores into weights") {
  constexpr std::uint32_t kLast = 99;  // reads 100 keys of the 128
  const std::vector<std::uint32_t> before = store_with(count_for(kLast));
  ScratchProgram p = program();
  const std::array<ScratchProgram::Input, 1> in = {{{0, before}}};
  p.run(step(kLast, 3, true, false), in);
  weights_as_stated(p.words(0), before, count_for(kLast));
}

TEST_CASE("past a full window, every key has a weight") {
  constexpr std::uint32_t kLast = 300;  // reads the last 128 keys
  const std::vector<std::uint32_t> before = store_with(count_for(kLast));
  ScratchProgram p = program();
  const std::array<ScratchProgram::Input, 1> in = {{{0, before}}};
  p.run(step(kLast, 1, true, false), in);
  weights_as_stated(p.words(0), before, kKeys);
}

TEST_CASE("a decode step writes nothing") {
  const std::vector<std::uint32_t> before = store_with(count_for(99));
  ScratchProgram p = program();
  const std::array<ScratchProgram::Input, 1> in = {{{0, before}}};
  p.run(step(99, 1, true, true), in);
  CHECK(p.words(0) == before);
}

TEST_CASE("a prefill step that does not end the prompt writes nothing") {
  const std::vector<std::uint32_t> before = store_with(count_for(99));
  ScratchProgram p = program();
  const std::array<ScratchProgram::Input, 1> in = {{{0, before}}};
  p.run(step(99, 3, false, false), in);
  CHECK(p.words(0) == before);
}
