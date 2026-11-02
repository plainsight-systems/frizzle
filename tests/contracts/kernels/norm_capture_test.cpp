// Axis V: the kernels suite (tests/README.md), the norm's capture.
//
// Contract locked (kernels/norm/norm.h, Capture): one norm launch, alone on
// the GPU, in the step that ends the prompt (logits 1, fed 0), writes into
// the capture store, from the step's last row only:
//   - into the residual slot, X as the launch leaves it: after the add where
//     it adds, and as it was read where it does not;
//   - into the added slot, y exactly as it was added: so X before plus the
//     captured y, one f32 add an element, is X written, bit for bit.
// It writes nothing else into the store, and nothing at all in a decode
// step or a prefill step that does not end the prompt. X and normed are
// the same bits whether the step captures or not. The final norm, a launch
// over the last token, captures from that row too.
//
// What it does not check, and what does:
//   - That the norm's arithmetic is right: the oracle, end to end, as for
//     every ported kernel (tests/README.md).
//   - That the graph gives each norm the right layer's slots, so the store
//     holds layer l's residual where layout() says: no test names it on
//     its own; the replay scenario's inspection compares whole stores, and
//     the page's read-back of these captures comes with the step that shows
//     them (QUEUE, slice 4, step 6).
//   - The identity "X before plus y is X written" holds where the target's
//     compiler does not fuse the post-norm's last multiply into the add;
//     WGSL does not say it may not. Checked here on the target, as the
//     attention's equal bits are, not promised.
//
// Inputs, all made here, from a fixed linear congruential sequence: Gemma
// 3's width of 1,152 (two vec4s an invocation, the second guarded); 3 rows;
// gains near 1, X of order 10, y of order 100, both signs; a store of two
// slots at unaligned word offsets, 7 and 7 + 1,152 + 5, filled first with a
// pattern no value shares. No model file.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.7    State postconditions: every word of the store is compared, and
//            X and normed whole.

#include <doctest/doctest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

#include "core/kernels/interface.h"
#include "core/kernels/norm/norm.h"
#include "core/residency/plan.h"
#include "support/scratch_program.h"

namespace {

namespace frz = frizzle;
using frz::testing::ScratchProgram;

constexpr std::uint32_t kWidth = 1152;
constexpr std::uint32_t kRows = 3;
constexpr float kEpsilon = 1e-6F;
constexpr std::uint32_t kResidualSlot = 7;
constexpr std::uint32_t kAddedSlot = kResidualSlot + kWidth + 5;
constexpr std::uint32_t kStoreWords = kAddedSlot + kWidth + 9;

// The buffers, in the plan's order.
constexpr std::size_t kGain = 0;
constexpr std::size_t kPostGain = 1;
constexpr std::size_t kOutput = 2;
constexpr std::size_t kHidden = 3;
constexpr std::size_t kNormed = 4;
constexpr std::size_t kStore = 5;

// A fixed sequence, so every run sees the same inputs.
class Sequence {
 public:
  explicit Sequence(std::uint32_t seed) : state_(seed) {}
  // A float in [low, high).
  float next(float low, float high) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float unit = static_cast<float>(state_ >> 8U) / 16777216.0F;
    return low + ((high - low) * unit);
  }

 private:
  std::uint32_t state_;
};

// How a vector of inputs is drawn: from the sequence at `seed`, in
// [low, high), every third value negated where `signs`.
struct Draw {
  std::uint32_t seed;
  float low;
  float high;
  bool signs;
};

std::vector<std::uint32_t> floats(std::size_t count, const Draw& d) {
  Sequence s(d.seed);
  std::vector<std::uint32_t> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    float v = s.next(d.low, d.high);
    if (d.signs && (i % 3 == 1)) v = -v;
    out[i] = std::bit_cast<std::uint32_t>(v);
  }
  return out;
}

std::vector<std::uint32_t> pattern() {
  std::vector<std::uint32_t> p(kStoreWords);
  for (std::size_t i = 0; i < p.size(); ++i) {
    p[i] = 0xA5A50000U + static_cast<std::uint32_t>(i);
  }
  return p;
}

struct Inputs {
  std::vector<std::uint32_t> gain = floats(kWidth, {1, 0.5F, 1.5F, false});
  std::vector<std::uint32_t> post_gain = floats(kWidth, {2, 0.5F, 1.5F, false});
  std::vector<std::uint32_t> output =
      floats(std::size_t{kRows} * kWidth, {3, 1.0F, 100.0F, true});
  std::vector<std::uint32_t> hidden =
      floats(std::size_t{kRows} * kWidth, {4, 0.1F, 10.0F, true});
  std::vector<std::uint32_t> store = pattern();
};

// What a case's launch is.
struct Shape {
  bool add = true;
  bool post_norm = true;
  frz::kernels::Rows rows = frz::kernels::Rows::EveryToken;
  bool residual = true;
  bool added = true;
};

ScratchProgram program(const Shape& shape) {
  const std::array<std::uint64_t, 6> sizes = {
      4ULL * kWidth,         4ULL * kWidth,         4ULL * kRows * kWidth,
      4ULL * kRows * kWidth, 4ULL * kRows * kWidth, 4ULL * kStoreWords};
  return ScratchProgram(
      sizes, [&](std::span<const frz::residency::BufferRange> r) {
        const frz::residency::WeightView gain =
            ScratchProgram::f32_view(kGain, kWidth);
        const frz::residency::WeightView post =
            ScratchProgram::f32_view(kPostGain, kWidth);
        frz::kernels::NormLaunch n{
            &gain,      shape.post_norm ? &post : nullptr,
            shape.add,  r[kOutput],
            r[kHidden], r[kNormed],
            kEpsilon,   shape.rows};
        n.capture.store = r[kStore];
        if (shape.residual) n.capture.residual = kResidualSlot;
        if (shape.added) n.capture.added = kAddedSlot;
        return std::vector<frz::kernels::Launch>{frz::kernels::norm_launch(n)};
      });
}

frz::kernels::Step step(std::uint32_t tokens, bool logits, bool fed) {
  frz::kernels::Step s{};
  s.position = 40;
  s.tokens = tokens;
  s.logits = logits ? 1 : 0;
  s.fed = fed ? 1 : 0;
  return s;
}

void run(ScratchProgram& p, const frz::kernels::Step& s, const Inputs& in) {
  const std::array<ScratchProgram::Input, 5> inputs = {
      {{kGain, in.gain},
       {kPostGain, in.post_gain},
       {kOutput, in.output},
       {kHidden, in.hidden},
       {kStore, in.store}}};
  p.run(s, inputs);
}

std::span<const std::uint32_t> row(std::span<const std::uint32_t> rows,
                                   std::uint32_t r) {
  return rows.subspan(std::size_t{r} * kWidth, kWidth);
}

// The slot of the width at `first`.
std::vector<std::uint32_t> slot(std::span<const std::uint32_t> store,
                                std::uint32_t first) {
  return {store.begin() + first, store.begin() + first + kWidth};
}

// Every word outside the slots at `firsts` is still `before`'s.
void untouched_but(std::span<const std::uint32_t> store,
                   std::span<const std::uint32_t> before,
                   std::initializer_list<std::uint32_t> firsts) {
  REQUIRE(store.size() == before.size());
  std::size_t differing = 0;
  for (std::size_t i = 0; i < store.size(); ++i) {
    bool in_slot = false;
    for (const std::uint32_t f : firsts) {
      in_slot = in_slot || (i >= f && i < std::size_t{f} + kWidth);
    }
    if (!in_slot && store[i] != before[i]) ++differing;
  }
  CHECK(differing == 0);
}

// X before plus each captured y, one f32 add, is X written, bit for bit.
struct Identity {
  std::span<const std::uint32_t> before;  // X as written into the step
  std::span<const std::uint32_t> added;   // the captured y
  std::span<const std::uint32_t> after;   // X as the launch wrote it
};

void added_is_what_was_added(const Identity& rows) {
  std::size_t differing = 0;
  for (std::size_t i = 0; i < kWidth; ++i) {
    const float sum = std::bit_cast<float>(rows.before[i]) +
                      std::bit_cast<float>(rows.added[i]);
    if (std::bit_cast<std::uint32_t>(sum) != rows.after[i]) {
      if (differing == 0) {
        MESSAGE("first difference at element " << i);
      }
      ++differing;
    }
  }
  CHECK(differing == 0);
}

}  // namespace

TEST_CASE("an adding norm captures the residual and what it added") {
  const Inputs in;
  ScratchProgram p = program({});
  run(p, step(kRows, true, false), in);
  const std::vector<std::uint32_t> hidden = p.words(kHidden);
  const std::vector<std::uint32_t> store = p.words(kStore);
  const std::span<const std::uint32_t> last = row(hidden, kRows - 1);
  const std::vector<std::uint32_t> residual = slot(store, kResidualSlot);
  CHECK(std::vector<std::uint32_t>(last.begin(), last.end()) == residual);
  added_is_what_was_added(
      {row(in.hidden, kRows - 1), slot(store, kAddedSlot), last});
  untouched_but(store, in.store, {kResidualSlot, kAddedSlot});
}

TEST_CASE("a norm that does not add captures X as it was read") {
  const Inputs in;
  ScratchProgram p =
      program({.add = false, .post_norm = false, .added = false});
  run(p, step(kRows, true, false), in);
  std::vector<std::uint32_t> want = in.store;
  const std::span<const std::uint32_t> last = row(in.hidden, kRows - 1);
  for (std::uint32_t i = 0; i < kWidth; ++i) want[kResidualSlot + i] = last[i];
  CHECK(p.words(kStore) == want);
}

TEST_CASE("the final norm captures what it added, from the last row") {
  const Inputs in;
  ScratchProgram p =
      program({.rows = frz::kernels::Rows::LastToken, .residual = false});
  run(p, step(kRows, true, false), in);
  const std::vector<std::uint32_t> hidden = p.words(kHidden);
  const std::vector<std::uint32_t> store = p.words(kStore);
  added_is_what_was_added({row(in.hidden, kRows - 1), slot(store, kAddedSlot),
                           row(hidden, kRows - 1)});
  untouched_but(store, in.store, {kAddedSlot});
}

TEST_CASE("X and normed are the same bits whether the step captures or not") {
  const Inputs in;
  ScratchProgram p = program({});
  run(p, step(kRows, false, false), in);  // does not end the prompt
  const std::vector<std::uint32_t> hidden = p.words(kHidden);
  const std::vector<std::uint32_t> normed = p.words(kNormed);
  run(p, step(kRows, true, false), in);  // ends it, and captures
  CHECK(p.words(kHidden) == hidden);
  CHECK(p.words(kNormed) == normed);
}

TEST_CASE("a decode step writes nothing to the store") {
  const Inputs in;
  ScratchProgram p = program({});
  run(p, step(1, true, true), in);
  CHECK(p.words(kStore) == in.store);
}

TEST_CASE("a prefill step that does not end the prompt writes nothing") {
  const Inputs in;
  ScratchProgram p = program({});
  run(p, step(kRows, false, false), in);
  CHECK(p.words(kStore) == in.store);
}
