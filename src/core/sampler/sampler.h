#pragma once

#include <cstdint>
#include <string>

#include "core/kernels/interface.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace frizzle::sampler {

// Axis F: changes with a new sampling method. The settings a model samples
// with are policy, and are passed in.
//
// The draw: the next token, drawn on the GPU from the kCandidates (64)
// candidates top-k selection kept (kernels/topk/topk.h), sorted largest
// first. The token stays on the GPU: the next decode step embeds it from the
// draw's record (kernels/gather/gather.h, a `fed` step), and the CPU reads
// the record back a step behind, while that step runs (kernels/program.h,
// runtime/runtime.h). Also the settings' check, and their place in the step.
//
// Contract
//   - check(settings): refuses, never clamps, naming the setting and its
//     range: top_k outside 1 to kCandidates; temperature not finite, or
//     below 0; top_p outside (0, 1]; min_p outside [0, 1). The runtime
//     calls it before a turn begins and refuses the turn as Settings
//     (runtime.h). A top_k of 0 is refused: it does not mean every token.
//   - apply(settings, seed, step): writes the draw's fields of the step
//     (kernels/interface.h, Step): the seed, low word first; top_k,
//     temperature, top_p and min_p as given; and log_min_p, ln min_p in f32
//     on the CPU, or 0 for a min_p of 0. Precondition: check(settings)
//     passed. The runtime calls it once a turn; the fields then ride in the
//     head every step writes.
//   - draw_launch(candidates, sampled): the draw's one launch, a pure
//     function of its arguments: over `candidates`, kCandidates (logit bits,
//     token) pairs sorted largest first, into `sampled`, one SampledRecord.
//     The settings, the seed and the position are the step's. It covers the
//     last token alone (Rows::LastToken), so it runs only in a step that
//     asks for logits. Preconditions, stated, not checked: `candidates`
//     holds kCandidates pairs as top-k writes them, from a vocabulary of at
//     least kCandidates tokens; `sampled` holds 16 bytes.
//   - SampledRecord: the token drawn; `failed`; u, the uniform the draw
//     used; a word of padding. 16 bytes, what the step reads back.
//   - Determinism: the draw is a function of the candidates, the settings,
//     the seed and the drawn token's position alone. The same bits on the
//     same GPU and compiler; WGSL's exp may round differently on another,
//     which moves a draw only where a cut or u's comparison falls within
//     that rounding. Across GPUs, not verified.
//   - Failure is visible: when the first candidate's logit is not finite
//     (top-k ranks every NaN and +∞ first, topk.h), the draw writes token 0
//     and sets `failed`, so the next step embeds a real token, and the
//     runtime, reading the record a step later, ends the turn NonFinite
//     (runtime.h). A drawn token is otherwise one of the candidates, below
//     the vocabulary.
//
// Algorithm, the draw (draw.wgsl)
//   One workgroup of 64 invocations. k = min(top_k, 64); x_i is candidate
//   i's logit, x_0 the first's.
//   1. Side by side, each invocation i below k works candidate i, in f32:
//      for a logit of −∞, every weight is 0 and the min-p test fails;
//      otherwise its top-p weight exp(x_i − x_0), only when top_p < 1; its
//      tempered weight exp((x_i − x_0) / temperature), only when
//      temperature > 0; and its min-p test, min_p = 0 or x_i ≥ x_0 +
//      log_min_p. Each goes to workgroup memory. Invocations at or past k
//      read nothing. The last invocation computes u (below). A barrier.
//   2. Only when top_p < 1: invocation 0 sums the k top-p weights in
//      candidate order; a barrier; each invocation below k divides its
//      weight by that total, the softmax at temperature 1; a barrier. At a
//      top_p of 1 none of it runs: top-p keeps every candidate, and an f32
//      softmax would round a far tail to zero and drop it.
//   3. Invocation 0 alone, in candidate order: the tempered weights' running
//      sums and, when top_p < 1, beside them the cumulative probability,
//      stopping at the first candidate where it reaches top_p. The kept
//      candidates are that prefix, at least one; otherwise all k.
//   4. min-p: the survivors are the longest prefix of the kept, at least the
//      first, whose candidates pass the test. Sorted largest first, those
//      that pass are a prefix.
//   5. Temperature 0 draws the first candidate. Otherwise, by inverse
//      transform: the first survivor whose running sum exceeds u × the
//      survivors' total, or, should rounding leave none, the last survivor.
//   6. The record: the token, or 0 when failed; failed, 1 or 0; u's bits; 0.
//   u: Philox4x32-10, keyed by the turn's 64-bit seed as two words, with
//   the counter (position, 0, 0, 0), position being the drawn token's: the
//   step's first position plus its tokens. So a token's draw is the same
//   whether its prompt was prefilled in one step or resumed from the cache;
//   a run replays from its seed, and any step replays alone. WGSL has no
//   64-bit integer, so Philox's high products are formed from 16-bit
//   halves. u is the first output word's top 23 bits × 2⁻²³, plus 2⁻²⁴:
//   in [2⁻²⁴, 1 − 2⁻²⁴], every value exact in f32, never 0 or 1.
//   Rounding points: each exponential, quotient, product and sum in f32 by
//   WGSL's rules; x_0 + log_min_p once an invocation; the sums in candidate
//   order, at most 64 terms, on one invocation; ln min_p once a turn, on the
//   CPU. No bound on a survivor's share is derived here.
//
// Design and cost
//   Counted at policy.h's defaults (top_k 40, top_p 0.95, min_p 0.05,
//   temperature 0.8), the settings every host passes, and at the largest
//   top_k, 64.
//   - Launches: one in a step that asks for logits, after selection's three
//     (topk.h), about 1.5 µs (interface.h's estimate, not measured in
//     Frizzle). No buffer write of its own: its constants, one word, are
//     written once at load (kernels/program.h); its settings and seed are 28
//     bytes of the 64-byte head every step writes (kernels/interface.h).
//   - Device memory: each invocation below k loads its candidate's logit
//     and the first's, and invocation 0 the chosen token: 8 × top_k + 4
//     bytes logically, 4 × top_k + 4 distinct, which the GPU's caches serve
//     — 324 and 164 at 40, 516 and 260 at 64 — and it writes 16.
//   - Barriers and serial work: three barriers when top_p < 1, one at a
//     top_p of 1. On invocation 0, one after another, four loops of at most
//     top_k iterations — 160 at 40, 256 at 64 — or three at a top_p of 1.
//   - An estimate of its latency at the worst, top_k 64 and top_p below 1,
//     on cores near 1.4 GHz from about 400 cycles for the loads, 50 a
//     barrier and 8 an iteration with its read, add or compare and loop
//     control: 2,598 cycles, 1.9 µs, leaving out the exponentials' and the
//     divisions' latency and the issue slots Philox's 10 rounds take. An
//     estimate, partial, not a bound, not measured in Frizzle.
//   - The split at k is by invocation, so a SIMD group straddling k idles
//     its lanes past it; the branches on top_p and temperature are uniform,
//     the step's settings, so every barrier is reached by all (GPU.4).
//   - Readback: the record's 16 bytes are copied into one of two mappable
//     slots and mapped while the next step runs (kernels/program.h); on the
//     CPU a token costs one mapped read of 16 bytes and the runtime's work
//     on it, a step behind (runtime.h). Read back before the next step could
//     start, each token would wait a map's round trip, about 0.5 ms, an
//     estimate, not measured in Frizzle, against a decode step's estimated
//     1.8 ms of weight reads and 0.45 ms of dispatch overhead (interface.h).
//   - Working buffers for Gemma 3 1B's vocabulary of 262,144, counted:
//     `partials_a` 131,072 bytes and `partials_b` 8,192, selection's;
//     `candidates`, 512; `sampled`, 16 (graph/graph.h).
//   - Decode steps are submitted back to back, each fed. When a token is a
//     stop, or the turn is cancelled, the step queued behind it still runs,
//     and its draw is discarded; the cache keeps each position whose input
//     token is accepted (runtime.h, turn.h). The discarded step's entries
//     lie past the cache's length, counted by its high-water mark and
//     overwritten in place by the next tokens there (cache/kv.h); no step
//     is queued at the context offered (turn.h, Context).
//   - Optimization (practice): only the first top_k candidates are read and
//     worked, and none of top-p's work is done at a top_p of 1, where it
//     would keep every candidate.
//   - Optimization (practice): the token drawn and kept on the GPU, the
//     readback a step behind (GPU.1, GPU.7).
//   - Inverse transform spends one uniform a draw, where a random-key race
//     spends one a candidate, up to 64, and gives no cumulative probability,
//     which top-p needs (GDSA.21's caveats).
//   - Lever not taken: the draw folded into selection's last pass, whose
//     one workgroup already holds the 64 candidates: one launch fewer, about
//     1.5 µs, and its candidate loads, at the cost of the sampling method
//     living inside the selection kernel, so that a new method would change
//     the selection (docs/architecture/change-axes.md: axis F apart from E).
//
// Verification
//   No test checks the draw on its own: no CPU reference, no known-answer
//   test of Philox, no test of draw frequencies, though GDSA.3 and GDSA.21
//   ask for them; and none of check's refusals. The oracle compares the
//   logits, which come before selection and the draw. Every reply the
//   replay scenario compares is drawn here, and those replies are compared
//   between runs of the engine from the same seed, never with a reference.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.8    Prefer pure functions: draw_launch computes from its arguments,
//            and the draw from the candidates, the settings, the seed and
//            the position.
//     I.5    State preconditions: apply's, draw_launch's, and check, which
//            the runtime calls before a turn.
//     ES.45  Avoid magic constants: the seed's word width (sampler.cpp)
//            and the record's four words (SampledRecord) are spelled from
//            their types.
//   C++ performance guidelines
//     GDSA.21 Draw categorical samples by a one-pass random-key race: not
//            taken, by its own caveats; one uniform a draw, and top-p needs
//            the cumulative probability the race does not give.
//     GDSA.3 Derive random numbers from (seed, stream, counter): Philox,
//            keyed by the seed, counted by the drawn token's position; the
//            transform from bits to u is the draw's own, exact in f32.
//     GDSA.2 Declare each reduction's determinism level: the sums run on one
//            invocation in candidate order, so the same bits run to run on
//            one GPU; across GPUs, not verified.
//     GPU.4  Keep SIMD-group control flow coherent: the settings' branches
//            are uniform; the split at top_k is by invocation.
//     GPU.1  Keep data on the device: the token stays on the GPU, and only
//            its 16-byte record is read back.
//     GPU.7  Pipeline CPU and GPU work: the readback a step behind, in two
//            slots.

// What the draw writes, and the readback carries.
struct SampledRecord {
  std::uint32_t token;
  std::uint32_t failed;  // 1 when the top candidate's logit was not finite
  float u;  // the uniform the draw used: a step replayed shows the same
  std::uint32_t padding;
};
static_assert(sizeof(SampledRecord) == 4 * sizeof(std::uint32_t));

// What check refuses, naming the setting and its range.
struct SettingsResult {
  bool ok = true;
  std::string subject;
};

[[nodiscard]] SettingsResult check(const policy::SamplingSettings& settings);

// Writes the draw's fields of `step`: the seed, the settings, and ln min_p.
// Precondition: check(settings) passed.
void apply(const policy::SamplingSettings& settings, policy::Seed seed,
           kernels::Step& step);

// The draw's launch: over `candidates`, kernels::kCandidates (logit, token)
// pairs sorted largest first, into `sampled`, one SampledRecord; the
// settings and the seed are the step's (kernels/interface.h).
[[nodiscard]] kernels::Launch draw_launch(
    const residency::BufferRange& candidates,
    const residency::BufferRange& sampled);

}  // namespace frizzle::sampler
