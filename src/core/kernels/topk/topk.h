#pragma once

#include <cstdint>
#include <vector>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"

namespace frizzle::kernels {

// Axis E: changes with a new or optimized kernel.
//
// Top-k selection: the step's logits, one row of Gemma 3 1B's 262,144-token
// vocabulary, reduced to its kCandidates (64) most likely tokens, sorted, for
// the sampler's draw (sampler/sampler.h). A selection, not a sort of the
// vocabulary: each workgroup keeps the top 64 of its tile in workgroup memory
// and passes them on.
//
// Contract
//   - topk_launches(logits, vocabulary, partials_a, partials_b, candidates):
//     the launches (kernels/interface.h, Launch) that reduce `logits`, one
//     row of `vocabulary` f32, to the top kCandidates (logit, token) pairs,
//     sorted, in `candidates`, through `partials_a` and `partials_b`. A pure
//     function of its arguments: no GPU object, no I/O.
//   - The order: by logit, largest first, and between equal logits the lower
//     token first. It is a total order, so the 64 kept are exactly the first
//     64 of the whole row under it, whatever the tiling. −0 and +0 are equal
//     as logits and tie. Every NaN, whatever its sign, ranks above +∞, so any
//     NaN in the row reaches the top, where the draw refuses a candidate that
//     is not finite (sampler.h): reported, never sampled, and never summed
//     into a softmax. −∞ keeps its place below every finite logit, a weight
//     of zero.
//   - Output: candidate i, for i from 0 to 63, is two u32, the logit's f32
//     bits and its token. The selection compares and never computes, so the
//     bits are the head's, except that −0 comes back as +0 and every NaN as
//     the quiet NaN 0x7FC00000.
//   - Determinism: only integer keys and token ids are compared, so the
//     output is the same bits for any tiling and run to run, given the same
//     logits. The same holds across GPUs by that argument; it is not
//     verified.
//   - Every pass covers the last token alone (Rows::LastToken), so it runs
//     only in a step that asks for logits (kernels/interface.h).
//   - Preconditions, stated, not checked: vocabulary >= kCandidates, which
//     the graph refuses otherwise (graph/graph.cpp, output); `partials_a`
//     holds n₁ = 64 × ceil(vocabulary / 1,024) pairs and `partials_b`
//     64 × ceil(n₁ / 1,024), as the graph sizes them; each later pass writes
//     fewer than the one two before; `candidates` holds kCandidates pairs. A
//     broken one gives launches that read or write the wrong words, which
//     WebGPU keeps within each bound range; nothing reports it.
//
// Algorithm, one pass
//   1. Keys. A logit's f32 bits b become a u32 key whose unsigned order is
//      the logits' order: −0 (0x80000000) is first made +0; a NaN (exponent
//      all ones, mantissa not zero) becomes 0xFFFFFFFF, the largest key;
//      otherwise a negative number has every bit flipped and a positive one
//      its sign bit set. Keys are compared as integers, so no float
//      comparison decides the order, and WGSL's freedom to assume no NaN or
//      infinity cannot reorder it.
//   2. Load. A workgroup of 256 invocations takes a tile of 1,024 entries,
//      4 each, as keys and token ids in workgroup memory, 8 KiB. An entry
//      past the row gets key 0 and token 0xFFFFFFFF, last under the order.
//      `first` reads logits, a token's id its index. `merge` reads the last
//      pass's (bits, token) pairs; a pair padded with token 0xFFFFFFFF, from
//      a tile that held fewer than 64, loads as past the row. Then one
//      barrier.
//   3. Sort the runs. Each of the tile's 16 runs of 64 is sorted by a
//      bitonic network, run r descending when r is even and ascending when
//      odd, so each pair of runs joined is bitonic: 21 stages (1 + 2 + … +
//      6), each 512 compare-exchanges, two an invocation, a barrier after
//      each.
//   4. Merge the runs in pairs, 16 to 8 to 4 to 2 to 1. In each pair, the
//      elementwise larger entries of the descending run and the ascending
//      one after it form a bitonic sequence holding the pair's top 64 (one
//      stage and a barrier), sorted in 6 more stages into the first run's
//      slot, descending for an even output run and ascending for an odd. The
//      active invocations halve with the runs. 4 rounds, 28 stages.
//   5. Write. The tile's first 64 entries, now descending, are written as
//      (bits, token) pairs to the workgroup's 64 slots of the output, a key
//      turned back into bits by inverting step 1.
//
// Algorithm, the passes (topk.cpp)
//   1. Pass 1 is `first`, over the logits; each later pass is `merge`, over
//      the last pass's output, whose count is 64 × the last pass's tiles.
//      A pass runs ceil(count / 1,024) workgroups of 256 invocations.
//   2. The pass that runs one workgroup is the last, and writes
//      `candidates`. Before it, `partials_a` and `partials_b` alternate,
//      pass 1 writing a and pass 2 b, so no pass binds one buffer both
//      read-only and writable.
//   3. Each pass's constants, binding 1, are struct TopK { count: u32 }, the
//      entries the pass reads. Bindings: 2 its input, 3 its output.
//   For Gemma 3 1B: 262,144 → 16,384 → 1,024 → 64, three passes of 256, 16
//   and 1 workgroups.
//
// Design and cost
//   - Optimization: select, not sort. 64 are kept from each tile of 1,024,
//     never a sorted vocabulary (GDSA.7). k is fixed at 64, well inside
//     what an on-chip selection holds, and a tile fits workgroup memory,
//     where every stage reorders it (GPU.5).
//   - Global memory, counted for Gemma 3 1B (GDSA.6): pass 1 reads 1,048,576
//     bytes of logits and writes 131,072, 16,384 pairs of 8 bytes; pass 2
//     reads those and writes 8,192; pass 3 reads those and writes 512.
//     1,327,616 bytes in all, about 3.3 µs at the M3 Max's published
//     400 GB/s (counted bytes over a published figure, not measured in
//     Frizzle).
//   - Launches, counted: 3, and the draw's, 4 in a step that asks for
//     logits; at about 1.5 µs each (interface.h's estimate), about 6 µs.
//     No buffer write in a step: each pass's constants are written once at
//     load. No readback: the candidates stay on the device for the draw
//     (GPU.1).
//   - Compare-exchanges, counted: a tile's 21 sorting stages are 16 runs ×
//     32 × 21 = 10,752, its merges 15 output runs × (64 + 6 × 32) = 3,840:
//     14,592 a tile, 3,983,616 over the 273 tiles of the three passes, run
//     side by side within a pass.
//   - Barriers, counted: 50 a pass, one after the load, 21 in the sort and
//     7 in each of the 4 merge rounds, one after another. At about 50
//     cycles a stage near 1.4 GHz, about 1.8 µs a pass and 5.4 µs for the
//     three (an estimate, not measured in Frizzle).
//   - The draw adds its own work, of which sampler.h estimates only a
//     partial subtotal, not a bound, so no total for the stage follows from
//     these figures. The selection's time is not measured in Frizzle.
//   - Levers not taken:
//       - Selection fused into the head's matrix product, so logits are
//         never written (GDSA.7). A decode workgroup of the head produces 8
//         logits (matmul.h, Algorithm 2), so its top 64 keeps all 8, and
//         carrying them on as (logit, token) pairs writes 8 bytes a token
//         where the logits write 4: 16V bytes written and read against the
//         8V of writing the logits and selecting from them, 2,097,152 bytes
//         more for Gemma 3 1B, counted, and no launch saved. Fusion pays only
//         with the head retiled so a workgroup produces hundreds of logits;
//         its decode form is shaped for reading the weights once (matmul.h),
//         and is not retiled for this.
//       - Radix select, which has no ceiling on k (GDSA.7): k is fixed at
//         64, where bitonic select is the simpler of the two, and radix
//         select's histogram passes need atomics or a launch per digit.
//   - Verification: no test checks the selection on its own; the ported
//     kernel families get no tests of their arithmetic (tests/README.md,
//     Kernels). The oracle compares the logits, which the selection reads,
//     not its candidates. Every reply the replay scenario compares is drawn
//     from its candidates, and those replies are compared between runs of
//     the engine, never with a reference.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.5    State preconditions: topk_launches', stated in comments and not
//            checked.
//     F.8    Prefer pure functions: topk_launches computes from its
//            arguments alone.
//     F.20   Prefer return values to out parameters: the launches are
//            returned.
//     ES.103 Don't overflow: a pass's tiles are a quotient and a remainder
//            test, never count + 1,023, which could wrap near 2^32
//            (topk.cpp).
//     ES.45  Avoid magic constants: the workgroup, the tile and k are named
//            (kWorkgroupSize, residency::kSelectionTile, kCandidates).
//   C++ performance guidelines
//     GDSA.7 Select top-k with a selection primitive, not a full sort: an
//            on-chip bitonic select, k well inside its reach; fusion into
//            the head and radix select weighed above.
//     GDSA.2 Declare each reduction's determinism level: a total order on
//            integer keys, the same 64 for any tiling, run to run; across
//            GPUs by the same argument, not verified.
//     GDSA.6 Count passes over global memory: the bytes above.
//     GPU.5  Use workgroup memory only when reuse or reordering pays: the
//            tile is reordered there by every stage; its barriers are
//            counted above.
//     GPU.6  Batch tiny GPU work: 3 launches, counted.
//     GPU.1  Keep data on the device: the candidates are read by the draw on
//            the GPU; nothing here is read back.

// The candidates a selection keeps, and the draw reads; the graph sizes the
// buffers by it (graph/graph.cpp).
inline constexpr std::uint32_t kCandidates = residency::kCandidates;

// The launches that reduce `logits`, one row of `vocabulary` f32, to the
// top kCandidates in `candidates`, (logit, token) pairs, sorted, through
// `partials_a` and `partials_b` alternately. Preconditions (Contract):
// vocabulary >= kCandidates; `partials_a` holds n₁ = 64 × ceil(vocabulary /
// 1,024) pairs and `partials_b` 64 × ceil(n₁ / 1,024), as the graph sizes
// them; each later pass writes fewer than the one two before; `candidates`
// holds kCandidates.
[[nodiscard]] std::vector<Launch> topk_launches(
    const residency::BufferRange& logits, std::uint32_t vocabulary,
    const residency::BufferRange& partials_a,
    const residency::BufferRange& partials_b,
    const residency::BufferRange& candidates);

}  // namespace frizzle::kernels
