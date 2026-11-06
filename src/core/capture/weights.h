#pragma once

// Axis O: changes with a new captured value, or what is sent.
//
// The attention weights' launch: in the step that ends the prompt, it turns
// each query head's scores at the last position, which the attention kernel
// wrote into the layer's weights slot, into that head's attention weights,
// in place, using the maximum and sum the attention kernel's fold made
// (kernels/attention/attention.h; capture/capture.h, the store).
//
// What the weights are, and what they are not: for the step's last query of
// head h, at position p, the attention kernel's contract is
//   out = Σ_j softmax(s)_j × v_j,  softmax(s)_j = exp2(s_j − m) / l,
// in log2 units, m the largest live score and l = Σ_j exp2(s_j − m). The
// weights are those softmax factors, from the kernel's own scores s_j, its
// own maximum m and its own sum l. The kernel never forms them: its fourth
// phase applies exp2(s_j − m_tile), rescaled afterwards by each later tile's
// α and each chunk's fold factor. So a weight differs from the factor the
// model applied to v_j by the rounding of those rescalings, a few units in
// the last place; it is not bit for bit what the model used. m is the same
// number either way (a maximum rounds nothing), and l is the fold's own.
//
// Contract
//   - weights_launch(store, slots): one layer's launch, binding the store
//     (binding 2) and nothing else; its constants are the layer's weights
//     and stats slots, its keys a head and its query heads.
//   - In the step that ends the prompt (capture/capture.h), for each query
//     head h, with
//       p        = position + tokens − 1, the step's last position,
//       earliest = p + 1 − min(p + 1, keys),
//       count    = p − earliest + 1, the keys p's query read,
//     it writes, for i in 0 .. keys − 1,
//       weights[h × keys + i] = exp2(s − m_h) / l_h   for i < count,
//                               0                      for i >= count,
//     where s is the score already at weights[h × keys + i], the key at
//     position earliest + i, and m_h, l_h are stats[2h] and stats[2h + 1].
//     In any other step it writes nothing. It writes nothing outside the
//     layer's weights slot, and never the stats.
//   - Preconditions: the layer's attention launches ran earlier in the same
//     step (graph/graph.h orders this launch after the combine), so every
//     score and both statistics are this step's; keys is layout()'s for
//     the layer, so count <= keys for any position the plan can offer.
//     l_h >= 1 always: the key holding the maximum adds exp2(0) = 1 to it,
//     and p's own key is always live for p's query, so no head is empty.
//   - Numerical: each weight within (8 + 3|s − m_h|) × 2⁻²⁴ of the same
//     quotient computed in f64 from the same f32 s, m_h and l_h, relative
//     to it, or within 2⁻¹²⁶ absolutely where the quotient is below the
//     smallest normal f32: the subtraction's rounding through exp2, exp2's
//     own 3 + 2|x| units (WGSL), the division's 2.5. A weight whose score
//     lies far below the maximum is tiny, and its relative error grows with
//     the distance; one 100 below is 2⁻¹⁰⁰ and invisible. A margin the GPU
//     test checks on the target, not a proof.
//   - Determinism (GDSA.2): no reduction; each weight is a function of its
//     own score and its head's statistics, so the same inputs give the same
//     bits run to run. The statistics carry the attention kernel's level,
//     batch-invariant.
//
// Algorithm (weights.wgsl), one workgroup of 256 a query head
//   1. If the step does not end the prompt (fed is 1), stop.
//   2. p, earliest and count from the step and keys, as above.
//   3. Invocation j of head h's workgroup takes i = j, j + 256, ... below
//      keys: the weight for i < count, else 0, written over the score.
//
// Design and cost
//   - Weights, not raw scores (the user, 2026-11-02): scores in log2 units
//     cannot be read, and forming the weights on the page would be a
//     computation beside the model; this launch is the one exception to
//     "nothing is computed beside the model" (AGENTS.md, Recurring
//     mistakes).
//   - In place: the scores' words become the weights. The scores are not
//     kept; what is shown is the weights.
//   - A launch of its own, after the combine, not inside the attention
//     kernel: the final (m, l) exist only once the query's last chunk is
//     folded, by main when a step is unsplit and by the combine when it is
//     split. Normalizing inside would mean both entry points writing
//     weights, or a workgroup waiting on others (GDSA.4).
//   - A launch over the last token (Rows::LastToken), so the program runs
//     it only in steps that ask for logits; `fed` tells the step that ends
//     the prompt from a decode step, as the copy launch does
//     (capture/capture.h). Its invocations are the query heads × 256, one
//     workgroup a head.
//   - Zeros past count, so every word of the slot is written in the step
//     that ends the prompt and the store holds no earlier run's keys
//     (capture.h).
//   - Cost, counted for Gemma 3 1B (not measured): in the step that ends
//     the prompt, each local layer reads and writes 8 KiB, each global one
//     512 KiB, 2.2 MiB in all, once a run; in a decode step, 26 dispatches
//     each reading the step and stopping, about 40 µs a token at about
//     1.5 µs a dispatch, an estimate, not measured in Frizzle.
//
// Corpus guidance
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level: it
//            makes none; the sum it divides by is the attention fold's.
//     GDSA.16 Never materialize the large intermediate: the attention
//            kernel's score matrix is still never formed; this launch sees
//            one row of keys a head.
//     GPU.6  Batch tiny GPU work: its decode cost is counted above.
//     GPU.2  Shape data for coalesced lane access: adjacent invocations
//            read and write adjacent words of a head's keys.
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed: it takes the
//            layer's slots, not words and counts.
//     I.5    State preconditions: the attention launches ran first in the
//            step.
//     I.7    State postconditions: every word of the slot written, in the
//            step that ends the prompt only.

#include "core/capture/capture.h"
#include "core/kernels/interface.h"
#include "core/residency/plan.h"

namespace frizzle::capture {

// Preconditions: `slots` is one of layout()'s; `store` is the plan's
// capture store, layout()'s total words long.
[[nodiscard]] kernels::Launch weights_launch(
    const residency::BufferRange& store, const LayerSlots& slots);

}  // namespace frizzle::capture
