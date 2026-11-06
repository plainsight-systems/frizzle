#pragma once

#include "core/formats/format.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace frizzle::kernels {

// Axis E: changes with a new or optimized kernel.
//
// QK-norm, rotary position embedding and the KV cache append, in one launch
// a layer: between the QKV projection, which wrote the step's queries, keys
// and values, and attention, which reads the queries from the query buffer
// and every key and value from the KV cache. "KV cache" is always the stored
// keys and values (cache/kv.h); the GPU's own caches are "the GPU's caches".
//
// Contract
//   - rope_launch(rope): the layer's one launch. Pure: no GPU object; the
//     program carries it out (kernels/interface.h).
//   - Inputs, for a Gemma 3 1B layer: 4 query heads and 1 key-value head of
//     d = 256 values; the rotary base, 10,000 for the 22 sliding-window
//     layers and 1,000,000 for the 4 full-attention layers, every sixth
//     (arch/gemma3/gemma3.cpp); the query and key gains, F32, d wide and
//     shared by every head, which the file stores as 1 + w and upload writes
//     as stored (arch/architecture.h); the file's norm ε, 10⁻⁶; the plan's
//     query (1,024 f32 a row), key and value (256 f32 a row) working
//     buffers; the layer's keys and values in the KV cache, each `slots`
//     entries of 256 values (residency/plan.h: 5,120 slots for a window
//     layer at the default reserve, the context offered for a
//     full-attention layer); and the cache's format, F16, whose pack the
//     launch composes.
//   - Effect, in a step of `tokens` rows from `position`, for row r at
//     p = position + r: the row's query heads normalized and rotated by p,
//     written back in place, f32; its key head normalized and rotated by p,
//     and its value head unchanged, packed into slot p mod slots of the
//     layer's keys and values. The key and value buffers are only read;
//     nothing else is written.
//   - Preconditions the graph checks before it builds the launch, refusing
//     the model otherwise (graph/graph.h): d is 64, 128 or 256, so d / 2
//     pairs fit the constants (kMaxPairs, 128) and a head's d / 8
//     invocations fit a workgroup; (query heads + 2 × key-value heads) ×
//     d / 8 is a multiple of 64 — 6 × 32 = 192 for Gemma 3 — though the
//     workgroup rule below needs only that d / 8 divides it, which always
//     holds; the query and key gains both present; no frequency factors,
//     which rope does not apply. Preconditions no code here checks:
//     each gain is F32, d values, in one piece; `cache_format` has a pack
//     (the program refuses a launch whose pack format has none) and is the
//     format the plan sized `cache` in; position + tokens <= 2^24, which the
//     program enforces (kernels/interface.h), so p is exact as an f32;
//     tokens <= kPrefillBlock <= slots, so a step's tokens take distinct
//     slots.
//   - Invariant: attention reads every key and value from the KV cache, the
//     step's own included, so a key is the same rounded value whether
//     attention meets it in the step that wrote it or a later one.
//     Attention's 1 / sqrt(d), 1 / 16, is attention's.
//   - Determinism (GDSA.2): run to run, and independent of the step's other
//     rows and its token count: a head is reduced by its own invocations in
//     one fixed order, and everything else is per value. A token's query,
//     key and value are the same bits whether it was prefilled or decoded.
//   - Ownership: the launch owns none of what it names. It names the plan's
//     buffers by index, offset and size (Binding), each gain by its view's
//     one piece; the joined WGSL by a view of static storage; the cache
//     format by a pointer into the capability table's static data. Nothing
//     in `rope` is kept past the call.
//   - Failure: none at run time. Packing saturates each value to F16's
//     largest finite, ±65,504, rather than storing an infinity
//     (formats/f16/f16_pack.wgsl).
//   - Accuracy, against the same computation in f64 from the same turns: each
//     rotated value within (|x| + |y|) × (2⁻¹¹ + 2π × 2⁻²⁴ × |t|), t the
//     angle in turns (Algorithm, step 1): WGSL bounds cos and sin to 2⁻¹¹
//     absolute on [−π, π], and t carries one f32 rounding, half a unit:
//     at position 32,767 and the first pair, t ≈ 5,215, about 0.0015
//     radians. The norm's sum of 256 squares adds in a fixed tree. Keys and
//     values are then rounded once to F16, to nearest on the target
//     (formats/format.h). A derived bound, not a tested one (Tests, below).
//
// Algorithm. Dimensions: d = 256; a head is d / 8 = 32 invocations, its
// lanes; a row's heads are its 4 query heads, its key head, then its value
// head, 6 × 32 = 192 invocations, of which the first 160 rotate.
//   - The workgroup (rope.cpp, workgroup_for): the largest power of two
//     from 64 down to a head's lanes that divides both a row's invocations
//     and its rotating ones, so that every invocation of a workgroup has one
//     token, and its heads all rotate or are all value heads. For Gemma 3,
//     64 does not divide 160, so the workgroup is 32: one head, six
//     workgroups a row, 6 × tokens a step.
//   - The token: each workgroup finds its row, (workgroup × 32) / 192, its
//     position p, and whether its head rotates, from its index alone, so
//     these are uniform across it, as the barriers after them require. A
//     workgroup past the step's rows returns.
//   1. A rotating workgroup computes its token's angles: for each pair k <
//      d / 2 = 128, shared out among its invocations, four each,
//      t = f32(p) × turns_k, rounded once; r = t − round(t), exact, in
//      [−0.5, 0.5]; angle = 2π × r, in [−π, π]; cos and sin of it into the
//      workgroup's tables. turns_k = θ_k / 2π, θ_k = base^(−2k / d), is the
//      launch's constant.
//   2. Invocation i of a head loads vec4s i and i + 32 of it: dimensions
//      4i .. 4i + 3 and 4i + 128 .. 4i + 131, four whole pairs (k, k + d / 2).
//   3. QK-norm, in a rotating workgroup: each invocation's 8 squares summed,
//      then a tree across the head's 32 invocations in workgroup memory,
//      halving at each level — log2(32) = 5 levels, a barrier before the tree
//      and after each level, 6 in all; the first also publishes step 1's
//      tables. Then r = inverseSqrt(sum / d + ε) and x = (x × r) × g, g the
//      query or key gain.
//   4. Query and key heads: each pair turned by its angle,
//        x' = x × cos − y × sin,  y' = x × sin + y × cos,
//      (x, y) dimensions k and k + d / 2: halves pairing, the only one rope
//      has.
//   5. Queries written back in place, f32. Keys and values packed by the
//      cache format's pack, four values to two words: invocation i of the
//      key or value head h, at slot s = p mod slots, writes words w + 2i,
//      w + 2i + 1, w + 2i + d / 4 and w + 2i + d / 4 + 1,
//      w = (s × H_kv + h) × d / 2.
//   Launch constants (binding 1), as WGSL lays them out:
//       struct Rope { slots: u32, epsilon: f32,
//                     turns: array<vec4<f32>, 32> }   // offset 16
//     16 + 128 × 4 = 528 bytes, three 256-byte slots of the constants
//     buffer; for d = 256 every turn is used.
//   Overrides: `head_dimension`, `query_heads`, `key_value_heads`, and
//   `pairs_per_head`, d / 2, which sizes the angle tables and is set on every
//   launch, as WebKit requires of an override that sizes a workgroup array;
//   the program sets `workgroup_size`.
//   Bindings: 2 the query gain and 3 the key gain, read-only; 4 the query
//   buffer, read and written; 5 the key and 6 the value buffers, read-only;
//   7 the layer's keys and 8 its values in the KV cache, written. Seven
//   storage buffers, within WebGPU's default eight, which the device asks
//   for (gpu/device_requirements.h).
//   The one form is assembled from two sources: rope.wgsl — the constants,
//   the gains, the queries, the angle tables, the norm's tree and the
//   rotation — then rope_cache.wgsl — the key, value and cache bindings and
//   the entry point — joined into one text at compile time (rope.cpp,
//   join). The program composes step.wgsl before it and the cache format's
//   pack after it (kernels/program.h).
//
// Design and cost
//   - Pipelines: the program keys a pipeline by the kernel's text, its
//     formats and its overrides. Every Gemma 3 1B layer has the same shape
//     and QK-norm, so one pipeline serves all 26 launches; the two rotary
//     bases differ only in each launch's constants.
//   - The workgroup of 32 follows from uniformity: a workgroup's barriers
//     sit under the branch on whether its heads rotate, and WGSL requires
//     every invocation of a workgroup to reach a barrier, so no workgroup
//     may hold rotating and value heads together (GPU.4).
//   - The angle tables are shared by the heads of a workgroup. At Gemma 3's
//     workgroup of 32, one head, they are shared by none: each rotating
//     workgroup computes its head's 128 pairs, 5 × 128 = 640 a token and a
//     layer, what computing them once a head costs. The tables add their
//     store and load in workgroup memory and no barrier beyond the norm's
//     first. A workgroup holds 1,152 bytes: 32 partial sums and the two
//     tables, 128 × 4 bytes each. A value-head workgroup computes no angles
//     and reaches no barrier.
//   - Optimization: the norms, both rotations and the cache append ride in
//     one launch, not six (GPU.6, GDSA.6). Unfused — two norms, two
//     rotations and two cache writes, each reading and writing its own pass
//     — a token and a layer would move 23 KiB, against 11 KiB here, in six
//     launches a layer, 130 more a pass, about 195 µs a step at the 1.5 µs a
//     launch measured in Frizzle (kernels/interface.h).
//   - Optimization: every invocation holds both values of each pair it
//     rotates, so the rotation exchanges nothing between invocations; and
//     adjacent invocations load and store adjacent vec4s, 16 bytes apart
//     (GPU.2).
//   - Optimization: the turns, fixed by the file, are computed once at load
//     in f64 into the form the kernel reads, turns a position, and ride in
//     constants written once (CDSA.32).
//   - Launches, counted: one a layer, 26 a pass, about 39 µs a step at
//     1.5 µs each, measured in Frizzle (kernels/interface.h).
//   - Bytes, counted, a token and a layer, f32 working buffers and an F16
//     cache: queries read and written, 8 d H_q = 8 KiB; keys and values
//     read, 8 d H_kv = 2 KiB; both written to the cache, 4 d H_kv = 1 KiB:
//     11 KiB. The gains, 2 × 1 KiB a layer, are read by every head and
//     served from the GPU's caches. A 512-row prefill step moves 26 × 512 ×
//     11 KiB = 143 MiB, about 0.37 ms at the target's published 400 GB/s
//     (an estimate, not measured in Frizzle); a decode step 286 KiB, about
//     0.7 µs, far below its launches.
//   - Arithmetic, counted: 640 sin and cos pairs a token and a layer; a
//     512-row prefill step, 8.5 million pairs, 17 million sin and cos. If
//     each costs about 40 instructions — a range check and a polynomial — at
//     roughly 5,120 lanes issuing one a cycle at about 1.4 GHz, that is about
//     0.1 ms: an estimate, not measured in Frizzle, and below the step's
//     0.37 ms of bytes, which it overlaps.
//   - Where WebGPU limits it:
//       - WGSL bounds pow only to 3 + 2 |y log2 x| units in the last place —
//         about 43 for the last pair at a base of 10⁶ — and cos and sin only
//         on [−π, π]. So the turns are computed once in f64 rather than with
//         pow per value, and the angle is reduced to [−π, π] before cos and
//         sin. Neither costs anything a step: the turns ride in constants
//         written at load, and the reduction is a round and a subtract.
//       - Subgroup operations are optional in WebGPU, and the device asks
//         for none (gpu/device_requirements.h). So a head's sum is a tree in
//         workgroup memory, 6 barriers, where a subgroup sum would need
//         none; WGSL's zero-filled workgroup memory adds a store and a
//         barrier a workgroup.
//   - Levers not taken:
//       - A prefill step's angles computed once, by a launch of their own,
//         into a working buffer the rope launches read: 2 bases × 512 rows ×
//         128 pairs = 131,072 pairs a step, 65 times fewer than the
//         workgroups compute. It saves the estimate above, about 0.1 ms a
//         512-row step, and costs a launch and a second form of the rope
//         launch, one for each regime.
//       - A cos and sin table by position: no sin or cos in the kernel, for
//         32,768 × 128 pairs × 8 bytes a base, 64 MiB for the two, taken
//         from the cache's budget — 16,384 tokens of context where the
//         budget binds, at 4 KiB a token for the four full-attention layers
//         once the rings are full — to save arithmetic in a kernel its bytes
//         bound.
//       - Attention's 1 / sqrt(d) folded into the queries here: one multiply
//         a query value saved against one a score; attention, which owns the
//         scale, takes it.
//   - Tests: none of this kernel alone. The ported kernel families get no
//     unit tests of their arithmetic by decision (tests/README.md); the
//     oracle, frizzle_oracle, checks rope end to end, as part of Gemma 3 1B's
//     logits against the frozen reference (tests/fixtures/reference/). So
//     the accuracy bound above, the ring's wrap, prefill and decode writing
//     the same bits, and the turns' f64 computation are locked by no test of
//     their own.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed: the gains are
//            views, the cache a planned layer, the format a Format.
//     C.12   Don't make data members const or references in a copyable
//            type: RopeLaunch holds its small parts by value and the rest by
//            pointer.
//     F.20   Prefer return values to out parameters: the launch is
//            returned.
//     I.5    State preconditions: the declaration's, and which of them the
//            graph checks.
//     C.48   Prefer default member initializers: every RopeLaunch member has
//            one, so a launch left partly filled holds nulls and zeros.
//     ES.42  Keep use of pointers simple: the constants' turns and the
//            joined text are written with at(), and the padding is a
//            std::array, at load and at compile time, never in a step.
//     I.24   Avoid adjacent parameters of one type: join takes its parts as
//            one ordered array.
//     ES.45  Avoid magic constants: a lane's 8 values, kValuesPerLane
//            (rope.cpp).
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level:
//            batch-invariant, as above.
//     GDSA.6 Count passes over global memory: the bytes above, and the
//            unfused count against them.
//     GPU.2  Shape data for coalesced lane access: adjacent invocations,
//            adjacent vec4s.
//     GPU.4  Keep control flow coherent: every branch that a barrier follows
//            is uniform across its workgroup, hence the workgroup of 32.
//     GPU.5  Use workgroup memory where reuse or reordering pays: the
//            norm's partial sums, which a tree needs; the angle tables, whose
//            reuse across heads does not happen at a workgroup of one head.
//     GPU.6  Batch tiny GPU work: six steps, one launch.
//     CDSA.32 Transform static data once into the layout its consumer reads:
//            the turns, at load.

// One layer's rope launch.
struct RopeLaunch {
  model::LayerDescription layer = {};  // heads, head dimension, rotary base
  const residency::WeightView* query_norm =
      nullptr;  // F32, head dimension; never null
  const residency::WeightView* key_norm =
      nullptr;  // F32, head dimension; never null
  float epsilon = 0.0F;
  residency::BufferRange query = {};  // the plan's working buffers
  residency::BufferRange key = {};
  residency::BufferRange value = {};
  residency::PlannedCacheLayer cache = {};  // the layer's
  const formats::Format* cache_format =
      nullptr;  // never null; its pack writes the cache
};

// The launch for one layer. Preconditions: the head dimension is 64, 128 or
// 256, and (query heads + 2 × key-value heads) × head dimension / 8 a
// multiple of 64; the views are F32 of the widths stated; `cache_format` has a
// pack and is the format the plan sized `cache` in.
[[nodiscard]] Launch rope_launch(const RopeLaunch& rope);

}  // namespace frizzle::kernels
