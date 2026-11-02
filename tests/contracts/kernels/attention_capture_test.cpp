// Axis V: the kernels suite (tests/README.md), the attention's capture.
//
// Contract locked (kernels/attention/attention.h, Capture): one layer's
// attention launch and its combine, alone on the GPU, in the step that
// ends the prompt (logits 1, fed 0), write into the capture store, for the
// step's last row at position p and each query head h:
//   - each live key's score, in log2 units, at weights[h × keys + (j −
//     earliest)], earliest = p + 1 − min(p + 1, keys): within the kernel's
//     stated accuracy of the f64 dot product of the same query and stored
//     key, times scale × log2(e);
//   - the head's maximum, equal to the largest of those scores, exactly,
//     and its sum, within a margin of the f64 sum of exp2(s − m) over them,
//     at stats[2h] and stats[2h + 1].
// Nothing else in the store: the slot's words past the live keys keep what
// they held. The same query at the same position gives the same bits as a
// one-token step, which splits, and as the last row of a prefill step that
// does not; and a window over a ring that has wrapped reads the right keys.
// Nothing is written in a decode step or a prefill step that does not end
// the prompt, and the attention output is the same bits whether the step
// captures or not.
//
// What it does not check, and what does:
//   - The attention output's arithmetic: the oracle, end to end, as for
//     every ported kernel (tests/README.md).
//   - That the weights launch turns these into weights: weights_test.cpp.
//   - That the graph gives the launch its layer's slots: the replay
//     scenario's inspection compares whole stores; the page's read-back
//     comes with the step that shows them (QUEUE, slice 4, step 6).
//
// The sum's margin: the fold adds at most `count` exp2 terms a head, each
// within (3 + 2|x|) units of exp2 (WGSL), rescaled by a chain of factors as
// each tile's and chunk's maximum rises; the margin taken, relative to the
// f64 sum, is (2 × count + 16 + 6 × range) × 2⁻²⁴, range the spread of the
// head's scores. Loose by design: an l from the wrong keys, or a chunk
// folded twice or not at all, is off by far more.
//
// Inputs, all made here: Gemma 3's attention shape, head dimension 256, 4
// query heads sharing 1 key-value head, an F16 cache; queries in [−1, 1)
// and keys and values multiples of 1/32 in [−2, 2], exactly representable
// in f16, from a fixed sequence; the model's scale, 1/16. A local layer of
// window 512 over a ring of 640 slots, at position 700, so its window has
// wrapped; a full-attention layer of window 4,096 at position 1,000. A
// store of the layer's two slots at unaligned offsets, filled first with a
// pattern no score shares. No model file.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.7    State postconditions: every word of the store is compared.
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level: the
//            prefilled and decoded captures are compared bit for bit.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/types.h"
#include "core/kernels/attention/attention.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "support/scratch_program.h"

namespace {

namespace frz = frizzle;
using frz::testing::ScratchProgram;

constexpr std::uint32_t kD = 256;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kRows = 64;          // the prefill step's rows
constexpr std::uint32_t kPartialRows = 512;  // as the plan's, at least
constexpr float kScale = 0.0625F;            // 1 / sqrt(256)
constexpr std::uint32_t kWeightsAt = 3;      // the slots' first words

// A layer of the test: its window, ring and the keys its slot holds.
struct Layer {
  std::uint32_t window;
  std::uint32_t slots;
  std::uint32_t keys;  // min(window, trained context)
  std::uint32_t last;  // the step's last position, p
  [[nodiscard]] std::uint32_t stats_at() const {
    return kWeightsAt + (kHeads * keys) + 5;
  }
  [[nodiscard]] std::uint32_t store_words() const { return stats_at() + 8 + 7; }
  [[nodiscard]] std::uint32_t earliest() const {
    return last + 1 - std::min(last + 1, keys);
  }
  [[nodiscard]] std::uint32_t count() const { return last - earliest() + 1; }
};

constexpr Layer kLocal{512, 640, 512, 700};
constexpr Layer kGlobal{4096, 4096, 4096, 1000};

constexpr std::size_t kQuery = 0;
constexpr std::size_t kKeys = 1;
constexpr std::size_t kValues = 2;
constexpr std::size_t kAttended = 3;
constexpr std::size_t kPartials = 4;
constexpr std::size_t kPartialStats = 5;
constexpr std::size_t kStore = 6;

// An f16's bits for a value f16 holds exactly (a normal or zero).
std::uint16_t f16(float v) {
  if (v == 0.0F) return 0;
  const std::uint32_t b = std::bit_cast<std::uint32_t>(v);
  const std::uint32_t sign = (b >> 16U) & 0x8000U;
  const int exponent = static_cast<int>((b >> 23U) & 0xFFU) - 127 + 15;
  REQUIRE((exponent > 0 && exponent < 31));
  REQUIRE((b & 0x1FFFU) == 0);  // no mantissa bits f16 drops
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10U) |
      ((b >> 13U) & 0x3FFU));
}

class Sequence {
 public:
  explicit Sequence(std::uint32_t seed) : state_(seed) {}
  std::uint32_t next() {
    state_ = (state_ * 1664525U) + 1013904223U;
    return state_ >> 8U;
  }
  float unit() { return static_cast<float>(next()) / 16777216.0F; }

 private:
  std::uint32_t state_;
};

// The inputs a case writes: queries for `rows` rows, the layer's cache, and
// a store holding the pattern.
struct Inputs {
  std::vector<std::uint32_t> query;
  std::vector<std::uint32_t> keys;
  std::vector<std::uint32_t> values;
  std::vector<std::uint32_t> store;
  std::vector<float> key_floats;  // by slot, then element, for the reference
};

// The last row's queries are the same for every case of a layer, so a
// prefill step's last row and a one-token step ask the same thing.
Inputs inputs(const Layer& layer, std::uint32_t rows) {
  Inputs in;
  Sequence early(7);
  Sequence last(11);
  in.query.resize(std::size_t{rows} * kHeads * kD);
  for (std::uint32_t r = 0; r < rows; ++r) {
    Sequence& s = r + 1 == rows ? last : early;
    for (std::uint32_t i = 0; i < kHeads * kD; ++i) {
      in.query[(std::size_t{r} * kHeads * kD) + i] =
          std::bit_cast<std::uint32_t>((2.0F * s.unit()) - 1.0F);
    }
  }
  Sequence cache(13);
  const std::size_t elements = std::size_t{layer.slots} * kD;
  in.key_floats.resize(elements);
  in.keys.resize(elements / 2);
  in.values.resize(elements / 2);
  for (std::size_t i = 0; i < elements; i += 2) {
    const auto pick = [&] {
      return static_cast<float>(static_cast<int>(cache.next() % 129) - 64) /
             32.0F;
    };
    const float k0 = pick();
    const float k1 = pick();
    in.key_floats[i] = k0;
    in.key_floats[i + 1] = k1;
    in.keys[i / 2] = f16(k0) | (std::uint32_t{f16(k1)} << 16U);
    const float v0 = pick();  // in order: an operator's operands are not
    const float v1 = pick();
    in.values[i / 2] = f16(v0) | (std::uint32_t{f16(v1)} << 16U);
  }
  in.store.resize(layer.store_words());
  for (std::size_t i = 0; i < in.store.size(); ++i) {
    in.store[i] = 0xA5A50000U + static_cast<std::uint32_t>(i);
  }
  return in;
}

ScratchProgram program(const Layer& layer) {
  const std::array<std::uint64_t, 7> sizes = {
      std::uint64_t{kRows} * kHeads * kD * 4,
      std::uint64_t{layer.slots} * kD * 2,
      std::uint64_t{layer.slots} * kD * 2,
      std::uint64_t{kRows} * kHeads * kD * 4,
      std::uint64_t{kPartialRows} * kHeads * kD * 4,
      std::uint64_t{kPartialRows} * kHeads * 2 * 4,
      std::uint64_t{layer.store_words()} * 4};
  return ScratchProgram(
      sizes, [&](std::span<const frz::residency::BufferRange> r) {
        frz::model::LayerDescription l{};
        l.query_heads = kHeads;
        l.key_value_heads = 1;
        l.head_dimension = kD;
        l.attention_window = layer.window;
        const std::array<frz::kernels::Launch, 2> launches =
            frz::kernels::attention_launches(
                {l,
                 kScale,
                 r[kQuery],
                 {r[kKeys], r[kValues], layer.slots},
                 frz::capability::find_format(frz::gguf::TensorType::F16),
                 r[kAttended],
                 r[kPartials],
                 r[kPartialStats],
                 kPartialRows,
                 {r[kStore], kWeightsAt, layer.stats_at(), layer.keys}});
        return std::vector<frz::kernels::Launch>(launches.begin(),
                                                 launches.end());
      });
}

frz::kernels::Step step(const Layer& layer, std::uint32_t tokens, bool logits,
                        bool fed) {
  frz::kernels::Step s{};
  s.position = layer.last + 1 - tokens;
  s.tokens = tokens;
  s.logits = logits ? 1 : 0;
  s.fed = fed ? 1 : 0;
  return s;
}

void run(ScratchProgram& p, const frz::kernels::Step& s, const Inputs& in) {
  const std::array<ScratchProgram::Input, 4> writes = {{{kQuery, in.query},
                                                        {kKeys, in.keys},
                                                        {kValues, in.values},
                                                        {kStore, in.store}}};
  p.run(s, writes);
}

// The case's capture: run one step and return the store.
std::vector<std::uint32_t> captured(const Layer& layer, std::uint32_t tokens) {
  const Inputs in = inputs(layer, tokens);
  ScratchProgram p = program(layer);
  run(p, step(layer, tokens, true, false), in);
  return p.words(kStore);
}

// Every word as the contract says: scores within the dot product's margin,
// m their maximum exactly, l within the fold's margin, everything else as
// it was.
void scores_and_stats_as_stated(const Layer& layer,
                                std::span<const std::uint32_t> got,
                                std::uint32_t rows) {
  const Inputs in = inputs(layer, rows);
  const double log2e = std::numbers::log2e;
  const std::size_t last_row = std::size_t{rows - 1} * kHeads * kD;
  std::size_t wrong = 0;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    double top = 0.0;
    double low = 0.0;
    for (std::uint32_t i = 0; i < layer.count(); ++i) {
      const std::uint32_t slot = (layer.earliest() + i) % layer.slots;
      double dot = 0.0;
      double size = 0.0;
      for (std::uint32_t k = 0; k < kD; ++k) {
        const double q = std::bit_cast<float>(
            in.query[last_row + (std::size_t{h} * kD) + k]);
        const double key = in.key_floats[(std::size_t{slot} * kD) + k];
        dot += q * key;
        size += std::fabs(q * key);
      }
      const double want = dot * kScale * log2e;
      const double margin = (kD + 2) * 0x1p-24 * size * kScale * log2e;
      const double s =
          std::bit_cast<float>(got[kWeightsAt + (h * layer.keys) + i]);
      if (!(std::fabs(s - want) <= margin)) {  // fails on NaN
        if (wrong == 0) MESSAGE("head " << h << " key " << i << ": " << s);
        ++wrong;
      }
      top = i == 0 ? s : std::max(top, s);
      low = i == 0 ? s : std::min(low, s);
    }
    const float m = std::bit_cast<float>(got[layer.stats_at() + (2 * h)]);
    CHECK(static_cast<double>(m) == top);
    double sum = 0.0;
    for (std::uint32_t i = 0; i < layer.count(); ++i) {
      sum += std::exp2(
          std::bit_cast<float>(got[kWeightsAt + (h * layer.keys) + i]) - top);
    }
    const double l = std::bit_cast<float>(got[layer.stats_at() + (2 * h) + 1]);
    const double margin =
        ((2.0 * layer.count()) + 16.0 + (6.0 * (top - low))) * 0x1p-24 * sum;
    CHECK(std::fabs(l - sum) <= margin);
  }
  CHECK(wrong == 0);
  // Words the contract does not write keep the pattern: past the live keys
  // of each head, and outside both slots.
  std::size_t touched = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    bool written = false;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t first = kWeightsAt + (std::size_t{h} * layer.keys);
      written = written || (i >= first && i < first + layer.count());
    }
    written = written || (i >= layer.stats_at() && i < layer.stats_at() + 8);
    if (!written && got[i] != in.store[i]) ++touched;
  }
  CHECK(touched == 0);
}

}  // namespace

TEST_CASE("a one-token step that ends the prompt writes scores and stats") {
  scores_and_stats_as_stated(kLocal, captured(kLocal, 1), 1);
}

TEST_CASE("the last row of an unsplit prefill step writes the same bits") {
  // 64 rows × 10 chunks is over the partial buffers' 512 rows: unsplit.
  // The one-token step at 700 reads chunks 2 to 10, 1 × 9 rows: split.
  CHECK(captured(kLocal, kRows) == captured(kLocal, 1));
}

TEST_CASE("a full-attention layer's scores reach back to position 0") {
  scores_and_stats_as_stated(kGlobal, captured(kGlobal, 1), 1);
}

TEST_CASE(
    "a decode step and a prefill step that does not end it write nothing") {
  const Inputs in = inputs(kLocal, kRows);
  ScratchProgram p = program(kLocal);
  run(p, step(kLocal, 1, true, true), in);
  CHECK(p.words(kStore) == in.store);
  run(p, step(kLocal, kRows, false, false), in);
  CHECK(p.words(kStore) == in.store);
}

TEST_CASE("the output is the same bits whether the step captures or not") {
  const Inputs in = inputs(kLocal, kRows);
  ScratchProgram p = program(kLocal);
  // Unsplit: a prefill step that does not end the prompt, then one that does.
  run(p, step(kLocal, kRows, false, false), in);
  const std::vector<std::uint32_t> unsplit = p.words(kAttended);
  run(p, step(kLocal, kRows, true, false), in);
  CHECK(p.words(kAttended) == unsplit);
  // Split: a decode step, then the one-token step that ends the prompt.
  const Inputs one = inputs(kLocal, 1);
  run(p, step(kLocal, 1, true, true), one);
  const std::vector<std::uint32_t> split = p.words(kAttended);
  run(p, step(kLocal, 1, true, false), one);
  CHECK(p.words(kAttended) == split);
}
