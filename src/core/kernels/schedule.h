#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/kernels/interface.h"

namespace frizzle::kernels {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The launches a step can dispatch, planned once at build from every
// launch's geometry, so a step walks those alone rather than every launch
// of the graph. The program builds it and asks it once a step
// (kernels/program.h); workgroups_for still decides each listed launch's
// workgroups (kernels/interface.h).
//
// Contract
//   - Schedules(launches): plans from every launch's geometry, in the
//     graph's order. Pure: no GPU object, no failure.
//   - of(tokens, logits): the indices, ascending, of every launch whose
//     token range holds `tokens` and, where `logits` is false, that does
//     not cover the last token alone. The span is valid while the
//     Schedules lives.
//   - Invariant: for every step, the list holds every launch
//     workgroups_for gives workgroups, in order, and only launches whose
//     token range and rows allow the step. A listed launch may still get
//     none: a combine in a step that does not split its key chunks, the one
//     thing that depends on the step's position.
//   - Precondition of of(): tokens >= 1. Not checked: below the first
//     bound there is no interval.
//
// Algorithm
//   1. Bounds: 1, every launch's least token count above 1, and every
//      launch's most + 1 that fits in 32 bits; sorted, duplicates removed.
//      Between two bounds, every launch runs at all of the interval or at
//      none of it.
//   2. For each interval, at its least token count: the list without
//      logits, of the launches whose range holds it and that do not cover
//      the last token alone; and the list with logits, of every launch
//      whose range holds it. Interval i's lists sit at 2i and 2i + 1.
//   3. of(): the interval is the last bound not above `tokens`, found by
//      binary search; return its list for `logits`.
//
// Design and cost
//   - Counted: at most 2 × launches + 1 bounds, two lists an interval, each
//     at most the launches; built once, at build. A step searches the
//     bounds and walks its list, where it would walk every launch; the
//     calls out of the module are unchanged. The bounds are the matrix
//     products' tile widths and the token counts where each product stops
//     splitting along K (kernels/matmul/matmul.h); how many that makes for
//     Gemma 3 1B is not counted here.
//   Optimization (practice): the launches a step can run planned once, by
//   token count and logits, and reused by every step (GDSA.17).
//   - Nothing tests Schedules alone. Every step the kernels suite, the
//     oracle and replay run goes through it.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant: the invariant above,
//            established by the constructor.
//     F.24   Use a span to designate a half-open sequence: a list.
//   C++ performance guidelines
//     GDSA.17 Plan once per key, reuse the plan: the key is the step's
//            interval and whether it asks for logits; the plan is rebuilt
//            only with the program.

class Schedules {
 public:
  // Plans from every launch's geometry, in the graph's order.
  explicit Schedules(std::span<const Geometry> launches);

  // The launches a step of `tokens` can dispatch, in order. Precondition:
  // tokens >= 1.
  [[nodiscard]] std::span<const std::uint32_t> of(std::uint32_t tokens,
                                                  bool logits) const noexcept;

 private:
  std::vector<std::uint32_t>
      bounds_;  // each interval's least token count, ascending
  std::vector<std::vector<std::uint32_t>>
      lists_;  // interval i without logits at 2i, with at 2i + 1
};

}  // namespace frizzle::kernels
