#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace frizzle::kernels {

// Axis E: changes with a new or optimized kernel.
//
// The matrix products of Gemma 3 1B's graph: in each of the 26 layers, Q, K
// and V as one product, the output projection, gate and up as one product
// with GELU applied, and the down projection; and the output head, the token
// embedding the file ties to it, over the step's last row (graph/graph.h).
// For a weight W of N rows of K weights (a GGUF matrix lists K first, so row
// o is output o's K weights, contiguous) and input rows x of K floats:
//
//   y[t][o] = Σ_k W[o][k] × x[t][k]
//
// The products, counted from the pinned file (formats/device_layout.h,
// residency/plan.h):
//
//   product      rows N               K      format             epilogue
//   Q, K, V      1,024 + 256 + 256    1,152  Q4_0               QKV
//   output       1,152                1,024  Q4_0               Write
//   gate, up     6,912 + 6,912        1,152  Q4_0               GatedActivation
//   down         1,152                6,912  Q4_0; Q4_1 in      Write
//                                            layers 0 to 2
//   head         262,144, 3 pieces    1,152  Q8_0               Write
//
// Contract
//   - matmul_launches(matmul, gate, range_sums): the launches that compute
//     one product (kernels/interface.h, Launch), and nothing else: no GPU
//     object, no I/O. A pure function of its arguments; the weight views
//     are read during the call and not kept.
//   - What the launches write, once each step has run: Write, row t of the
//     step's output o at out[t × out_width + o], out_width the weight's N
//     (a piece writes its own rows, from its first row); the decode form
//     writes row 0 alone, which for the head is the last row's logits. QKV,
//     output row j of Q, K or V at its own buffer's t × width + j, width its
//     rows: the rope kernel's query, key and value. GatedActivation,
//     activation[t × 6,912 + j] = gelu_tanh(gate_j) × up_j, never gate or
//     up themselves. Each output is written once a step, by one invocation.
//   - Launches, in order: for each piece of a one-member weight, or once for
//     a fused group, a decode launch in one-token steps, then for each tile
//     width of 8, 16 and 32 tokens a tiled launch, a split launch, or both,
//     and, when the product splits, the split's combine. For each width the
//     tiled and split launches' token ranges are disjoint and cover the
//     width's steps, so every step runs exactly one form of each product.
//     With rows LastToken (the head) the decode launch alone, in every step
//     that asks for logits, over the step's last row.
//   - Preconditions, stated, not checked: weights[0] is a planned view
//     whose format the capability table lists; K = dimensions[0] is a whole
//     number of 32-weight groups (formats/format.h). A fused group, members
//     3 (Q, K, V) or 2 (gate, up), is a PlannedGroup's (residency/plan.h):
//     each member one piece, all in one buffer within one binding's span,
//     sharing the first member's format and K; gate and up have equal rows.
//     The outputs hold kPrefillBlock rows of their widths; the input rows of
//     K floats; gate.denominator > 0. A broken precondition gives launches
//     that read or write the wrong words, which WebGPU keeps within each
//     bound range; nothing reports it.
//   - split_tokens and range_sum_floats: pure, noexcept; the graph sizes
//     `range_sums` with range_sum_floats under the same gate the launches
//     take, so the two cannot disagree (graph/graph.cpp, layer_range_sums).
//     A step splits only while its range sums fit the buffer passed in;
//     past that it runs tiled, the same bits.
//   - Numerical: each output is its K products added in one order fixed by
//     K alone (Algorithm, 1), in f32 with fma; the weights decode exactly as
//     format.h states. Within γ(2K) × Σ_k |W[o][k] x[k]| of the same sum in
//     f64 over the decoded weights, γ(n) = nu / (1 − nu), u = 2⁻²⁴: the bound
//     for a K-term dot product in which every multiply-add rounds at most
//     twice, since WGSL lets fma round twice. K is at most 6,912, the down
//     projection's, so γ(2K) is about 8.2 × 10⁻⁴. The GELU's e^x truncates
//     its series within about 2⁻²² (Algorithm, 5).
//   - Determinism (GDSA.2): run to run, and batch-invariant by design: a
//     token's outputs are the same bits decoded alone, in any tile width,
//     or split, since every form adds in the order of Algorithm 1, never one
//     the step chooses. Equal bits across forms rest on the target's
//     compiler fusing the explicit fmas alike in every pipeline, which WGSL
//     does not promise. Not bit for bit across GPUs or with other
//     implementations, whose orders differ.
//
// Algorithm
//   1. The order of addition. A row's G = K / 32 groups are cut into 32
//      ranges at fixed boundaries: range r is groups floor(r × G / 32) up to
//      floor((r + 1) × G / 32). Gemma 3 1B's: K 1,024, one group a range;
//      K 1,152, 28 ranges of one group and 4 of two; K 6,912, 24 of seven
//      and 8 of six. Then
//        s_r = fma(w, x, … fma(w_a, x_a, 0))   over range r's weights, k in
//                                              order, from zero
//        y   = ((0 + s_0) + s_1) + … + s_31    32 adds, in order, from zero
//      Every multiply-add is an explicit fma and every add one f32 add; those
//      are the rounding points.
//   2. Decode form, one row: the step's only row, or its last for the head.
//      A workgroup of 64 invocations is two sets of 32; each set takes
//      set_rows rows, and invocation r of a set takes range r of each. For
//      each group of its range it loads the input's 32 matching floats into
//      registers once, unpacks the group of each of its rows (format's
//      unpack, 32 weights at a time) and adds the products into that row's
//      range sum. The workgroup's range sums meet in workgroup memory, 8
//      rows × 32 floats; after one barrier invocations 0 to 2 × set_rows − 1
//      each add one row's 32 in order and write it. set_rows: 4 for the
//      gated product (2 gate rows and the same 2 up rows, so invocations 0
//      to 3 each hold a gate row and its up row), 1,728 workgroups; and for
//      a product of more than 4,088 rows, as each of the head's pieces is,
//      32,769 workgroups in all. Otherwise 1: for Q, K, V (768
//      workgroups), the output and the down projection (576 each).
//   3. Prefill tiled form, a step of 2 to 512 tokens. A workgroup takes a
//      tile of τ tokens × 64 outputs, τ the narrowest of 8, 16 and 32 that
//      holds the step (8 for 2 to 8 tokens, 16 for 9 to 16), else 32. Its 4τ
//      invocations each own 4 consecutive tokens × 4 outputs: invocation t
//      takes token quad t / 16 and row slots l, l + 16, l + 32, l + 48 for
//      l = t mod 16. For each range r, and each group g of it:
//        a. barrier; stage group g of the tile's 64 rows, decoded, into
//           workgroup memory, zeros for rows past the product, and the τ
//           tokens' 32 inputs for g, zeros past the step; each invocation
//           writes whole vec4s; barrier;
//        b. for each of the group's 8 k-quads, read 4 vec4s of weights and 4
//           of inputs, transpose the weights in registers, and add 16 vec4
//           fmas, k in order, into the 4 tokens' range sums (vec4s over the
//           4 outputs).
//      At the range's end each of the 4 totals adds its range sum,
//      componentwise, and the range sums restart at zero: the order of 1.
//      An invocation whose token quad lies wholly past the step stages and
//      passes both barriers but skips b. Workgroups are numbered token tile
//      first, then output tile. A gated tile's 64 slots are 32 gate rows and
//      the same 32 up rows: slot 16j + c holds gate row 2c + j for j 0 and
//      1, and up row 2c + j − 2 for j 2 and 3, so the invocation that
//      finishes a gate output holds its up output too.
//   4. Prefill split form, a step up to split_tokens tokens. Each tiled
//      workgroup becomes 32, one a range: workgroup (token tile, output
//      tile, r), numbered in that order, runs 3a and 3b for range r alone
//      and writes each output's range sum to `range_sums` at
//      (r × tokens + t) × rows + o, o the product's row. A combine launch
//      then gives each output of each token one invocation, which adds its
//      32 range sums in order to a total from zero and writes it as the
//      epilogue says: the same range sums and the same 32 adds as 1, so the
//      same bits.
//   5. GELU, gated product only: z = √(2/π) × fma(0.044715 g², g, g);
//      e = e^x for x = −2|z|, clamped at −100, computed as k = round(x log2
//      e), r = x − k ln 2 with ln 2 in two parts so r is nearly exact, e^r
//      by its Taylor series to r⁶ in Horner form (within about 2⁻²² for |r|
//      <= ln 2 / 2), times 2^k by ldexp; tanh(z) = sign(z) × (1 − e) × (1 /
//      (1 + e)); gelu = fma(g / 2, tanh, g / 2), times up. No intermediate
//      overflows, which matters because WGSL may assume no infinities.
//   6. Launches (matmul.cpp). Binding 1, 48 bytes: for each member, its
//      first word in the weight binding, its blocks (for unpack), its first
//      output row (for a gated member, 0: its first row within its own
//      weight) and its rows. Overrides: `columns` (K) and `out_width` on
//      every launch; `set_rows` on decode; `tile_tokens` on tiled and split;
//      `x_tile_vec4s`, 8 × the tile's tokens or the widest tile's, on every
//      launch (matmul.wgsl). Bindings: 2 the weights, a piece or the span of
//      a fused group, members read from their own first words, the file's
//      tensors between them bound and never read; 3 the input; then the
//      outputs, one or query, key and value. The split launch binds 2 the
//      weights, 3 the input, 4 `range_sums`; the combine binds 2
//      `range_sums`, read-only, then the outputs from 3. The split's steps
//      are min(split_tokens, the floats `range_sums` holds ÷ (rows × 32)).
//
// Design and cost
//   - One order of addition for every form, fixed by K, so a token's
//     products do not depend on the step that holds them (GDSA.2). Explicit
//     fma: written as acc + w × x, the target's compiler contracted it in
//     one pipeline and not the other, and decode and prefill bits differed.
//     Explicit e^x for the same reason: the built-in exp gave different
//     bits in the two pipelines.
//   - Weights stay in their file's format and are decoded in the load path
//     (GDSA.18); none is written out expanded.
//   - Decode reads every weight once from device memory, counted: Q, K, V
//     995,328 bytes, output 663,552, gate and up 8,957,952, down 4,478,976
//     (Q4_1, 4,976,640): 393,984,000 across 26 layers; the head 320,864,256,
//     45% of the step's 714,848,256 bytes (682 MiB), about 1.8 ms at the
//     M3 Max's published 400 GB/s, the floor for a decode step's products
//     (counted bytes over a published figure, not measured in Frizzle).
//     Arithmetic, 2 operations a weight, is far below it.
//   - Decode input reads, counted: each set reads its whole input row once,
//     at most 27 KiB, small enough for the GPU's caches to hold: a layer's
//     Q, K, V 6.75 MiB, output 4.5, gate and up 15.2, down 30.4;
//     the head 288 MiB. Whether they cost time is not measured in Frizzle.
//     Input values are reused across rows only where set_rows exceeds 1,
//     the gated product and the head.
//   - Optimization: a decode product that would make fewer than 512
//     workgroups at 4 rows a set takes 1 row a set (Algorithm, 2):
//     Q, K and V at 4 would be 192 workgroups, the output and down 144.
//     The reason is an estimate, not measured in Frizzle: products of about
//     128 workgroups read their weights at about 107 to 180 GB/s on the
//     target, those of 512 or more at about 260 to 383 (GPU.3). The cost is
//     the input reads above.
//   - Coalescing (GPU.2): decode's 32 invocations of a set read the 32
//     ranges of a row; where a range is one group (K 1,024) they read 32
//     adjacent blocks of each stream, for Q4_0's codes 512 bytes in 512.
//     Where ranges are 6 or 7 groups (K 6,912) invocation r reads about
//     block 6.75r + i at its i-th step, the rest of each line by its own
//     later steps, from the GPU's caches.
//   - Prefill (GPU.5, GDSA.16): each decoded weight in workgroup memory
//     serves the tile's τ tokens and each input 64 outputs; a k-quad is 64
//     fmas from 8 vec4 reads of workgroup memory, whole vec4s along K
//     (SIMD.2). Workgroup memory: 9 KiB of weights (64 slots of 8 vec4s and
//     a vec4 of padding, so the 16 slots one read takes spread across the
//     banks) and τ / 8 KiB of inputs: 10, 11 or 13 KiB, under WebGPU's
//     default 16. Decode's range sums take 1 KiB.
//   - Optimization: an invocation's 32 accumulators, 4 range sums and 4
//     totals of 4 outputs each, are named vec4s and its loops' bounds
//     constants, never arrays indexed by loop counters, which a compiler may
//     keep in memory rather than registers (GPU.3). A form with 64
//     accumulators in such arrays ran each launch of a 32-token step about
//     14 to 17 times as long as a 16-token step's (an estimate, not measured
//     in Frizzle). The totals beside the range sums are what batch invariance
//     costs: a kernel free to add in its own order would keep the 16 totals
//     alone.
//   - Tile width (GDSA.6): each token tile reads and decodes the whole
//     weight, so the narrowest tile that holds the step keeps it to one
//     weight pass up to 32 tokens and does the fewest multiply-adds among
//     those. A 32-lane SIMD-group on the target issues two token quads
//     together, so a step's products run to the next 8 tokens, at most 7
//     padded a step (GPU.4): guarding each token's chain instead would add a
//     branch a token on every k-quad of every tile. Neither changes an
//     output's order of addition.
//   - Optimization (browser): a tiled workgroup works out each invocation's
//     place in its tile (lane, live token quad, row slots' members and rows)
//     once, not once a range: recomputed each range, the tiled form ran 2 to
//     3% slower in Chrome (an estimate, not measured in Frizzle).
//   - Optimization: the split along K, for the three products whose tiles
//     are few beside their length along K. A tiled workgroup walks all G groups
//     in turn, so at a step of up to 32 tokens, one token tile, Q, K and V run
//     24 workgroups of 36 groups each, the output 18 of 32, and the down
//     projection 18 of 216 (432 barriers each): fewer workgroups than the
//     target has GPU cores (30 or 40 on an M3 Max). Split, they run 768, 576
//     and 576 workgroups of 1 or 2, 1, and 6 or 7 groups. At 21 tokens, Q, K
//     and V took about 95 µs tiled and 40 split, the down projection about 504
//     and 96: estimates, not measured in Frizzle, each launch timed alone in
//     Chrome 151 on an M3 Max. Because the split falls on the 32 ranges the
//     order already fixes, it changes no output's bits (GDSA.2). Its cost is
//     writing each range sum out and back, 128 bytes each way an output a
//     token (GDSA.16's caveat: the combine must see all 32), laid out range
//     first so both passes are coalesced (GPU.2): at 21 tokens 4.1 MB each
//     way for Q, K and V and 3.1 MB for the output or down; at the split's
//     top, 12 MiB at 64 tokens and 13.5 MiB at 96. And one combine launch a
//     product, 3 a layer.
//   - The gate, SplitGate (CDSA.21): the tiled form's workgroups W = output
//     tiles × 32-token tiles split when W ≤ 2G and W ≤ 64. So a product's
//     split steps are a token range, a whole number of 32-token tiles:
//     Q, K and V split up to 64 tokens, the output and down projection up
//     to 96; gate and up never (split_tokens is 1 for them by rule: their
//     combine would need a gate total and its up total). The ratio and the
//     cap were fitted to timings taken outside Frizzle, over product shapes
//     not all Gemma 3 1B's, the cap to bound the range sums at 16 MiB (64
//     tiles of 64 outputs × 32 tokens × 32 floats); whether 2 and 64 are
//     the best values for Gemma 3 1B's three products is not measured in
//     Frizzle. The gate is passed in, not read from ambient state (I.1).
//   - The working buffer `range_sums` is the most range_sum_floats over the
//     products a layer runs (graph/graph.cpp): 13.5 MiB, set by the output
//     and down projections at 96 tokens.
//   - Optimization: Q, K and V as one product and gate and up as one, one
//     binding spanning the members, so upload writes the file as it lies: 2
//     products a layer where there would be 5, 78 launches fewer a step,
//     about 117 µs at interface.h's measured 1.5 µs a launch (GPU.6).
//   - Optimization: GELU is the gate-and-up product's epilogue, so gate and
//     up are never written: per token and layer, 27 KiB written where
//     separate kernels would move 135 KiB (gate and up written, read back,
//     the activation written) (GDSA.6), and no up working buffer, 13.5 MiB
//     at kPrefillBlock rows.
//   - Prefill at 512 tokens, counted: 698 × 10⁶ layer weights × 512 tokens,
//     715 × 10⁹ operations, about 51 ms at the M3 Max's roughly 14 f32
//     TFLOPS (an estimate, not measured in Frizzle) if the tiles ran at
//     peak; the head adds one token's. Each weight is decoded once a token
//     tile, 16 times: 6.3 GB of weight loads, from device memory between
//     394 MB, if a weight tile stays in the GPU's caches while its 16 token
//     tiles run, and 6.3 GB, about 16 ms at 400 GB/s, if none does; WebGPU
//     promises neither the order workgroups run in nor what the caches keep,
//     which is why they are numbered token tile first. Each input float is
//     loaded once an output tile: 22.3 GB from the GPU's caches. Workgroups:
//     Q, K, V 384; output 288; gate and up 3,456; down 288.
//   - Launches a step, counted (interface.h: about 1.5 µs each, measured):
//     a decode step 4 a layer and 3 for the head, 107, about 160 µs; a
//     prefill step of 2 to 64 tokens 7 a layer, 65 to 96 tokens 6, longer
//     4; and 3 for the head when the step asks for logits. At load, 22
//     launches a layer and 3 for the head are built, each its own bind group.
//   - WebGPU's limits: no f32 matrix-multiply units across browsers, so
//     every multiply-add runs on the shader cores; subgroup operations are
//     optional, so decode's range sums meet in workgroup memory; workgroup
//     memory is 16 KiB at the defaults the harness keeps
//     (gpu/device_requirements.h), so a tile is 32 × 64.
//   - Levers not taken: an order each form chooses for itself (fewer
//     accumulators, at the cost of a token's bits depending on its step);
//     inputs staged at f16 (the optional shader-f16 feature, and inputs
//     rounded to f16 would no longer be the f32 that decode reads); a fixed
//     order in fewer levels, 8 range sums rather than 32 (a quarter of the
//     split's bytes, every token's bits changing once).
//   - Verification. No test checks this kernel alone: the ported kernel
//     families' arithmetic is left to the oracle (tests/README.md). The
//     oracle decodes one token a step, so it covers the decode form and
//     the head, within its tolerance, and no prefill form. The replay
//     scenario runs the tiled and split forms through the engine's prefill
//     and compares replies, not values: a prefill error too small to change
//     a sampled token passes it. Untested: the tiled, split and combine
//     forms' values; batch invariance; the split's bits against the tiled
//     form's; the gate's boundaries (64 and 65, 96 and 97 tokens); the
//     launches' geometry and token ranges.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.8    Prefer pure functions: matmul_launches, split_tokens and
//            range_sum_floats compute from their arguments alone.
//     F.20   Prefer return values to out parameters: the launches are
//            returned.
//     I.1    Make interfaces explicit: the gate is a parameter the graph
//            passes, never a global.
//     I.5    State preconditions: matmul_launches' and split_tokens',
//            stated in comments and not checked.
//     I.24   Avoid adjacent parameters that can be invoked by the same
//            arguments in either order: split_tokens and range_sum_floats
//            take a ProductShape, its rows and columns named.
//     R.3    A raw pointer is non-owning: MatmulLaunch's weight views, read
//            during the call.
//     Enum.2 Use enumerations for sets of related named constants: Epilogue.
//     Enum.7 Specify the underlying type only when necessary: Epilogue is one
//            byte; it chooses entry points on the CPU and is never written
//            to a GPU buffer, a file or a message.
//     ES.45  Avoid magic constants: the tiles, the micro-tile, the ranges,
//            the decode workgroup target and x_tile's vec4s a token are
//            named (matmul.cpp).
//     ES.42  Keep use of pointers simple: a launch's members, weights and
//            outputs are read with at(), at load, never in a step.
//     SL.con.3 Avoid bounds errors: those reads are bounds-checked.
//     Per.6  Don't claim performance without measurements: every time not
//            measured in Frizzle is labelled an estimate.
//   C++ performance guidelines
//     GDSA.2  Declare each reduction's determinism level: run to run and
//             batch-invariant, the order fixed by K, never by the step; the
//             test it asks for, a token alone against the same token in
//             steps of several sizes, bit for bit, is not run.
//     GDSA.18 Block-scaled codes decoded in the load path: unpack in
//             registers, no weight expanded.
//     GDSA.6  Count passes over global memory: the bytes above; the
//             activation epilogue; the narrowest tile holding a step.
//     GDSA.16 Stream through on-chip tiles: prefill walks K through
//             workgroup memory, each output written once; its caveat is the
//             split's range sums, written because the combine needs all 32.
//     GPU.2   Coalesced lane access: decode's ranges, the range sums laid
//             out range first.
//     GPU.3   Occupancy as a latency budget: named accumulators; enough
//             decode workgroups.
//     GPU.4   Coherent SIMD-group control flow: the guard is per token quad,
//             uniform across a quad's 16 lanes.
//     GPU.5   Workgroup memory where reuse pays: each staged weight and
//             input is read by many invocations; padded slots.
//     GPU.6   Batch tiny GPU work: fused weights and the fused activation.
//     SIMD.2  Linear vector loads: whole vec4s along K.
//     CDSA.21 A measured, named, injectable threshold: SplitGate is named
//             and passed in; its values were measured outside Frizzle, and
//             neither side is tested here.
//     GDSA.4  No wait on another workgroup without a forward-progress
//             guarantee: the split's combine is a launch of its own.

// What a product writes (Contract). One byte: it chooses entry points on the
// CPU and is never written to a GPU buffer, a file or a message.
enum class Epilogue : std::uint8_t {
  Write,
  QKV,
  GatedActivation,
};

// One matrix product of the graph.
struct MatmulLaunch {
  // The weight, or a fused group's members in output order: Q, K, V, or
  // gate, up. The first `members` are never null; a fused group's members
  // are one piece each, in one buffer. Read during the call, not kept.
  std::array<const residency::WeightView*, 3> weights;
  std::uint32_t members;         // 1, 2 or 3
  residency::BufferRange input;  // rows of K floats
  // Write: outputs[0]. QKV: query, key, value. GatedActivation: the
  // activation, in outputs[0].
  std::array<residency::BufferRange, 3> outputs;
  Epilogue epilogue;
  Rows rows;  // LastToken for the head
};

// When a prefill product splits along K: the tiled form's workgroups W at
// most `numerator` / `denominator` of its groups along K, and at most
// `max_workgroups`. The graph passes it; kSplitGate is the value it passes.
struct SplitGate {
  std::uint32_t numerator;
  std::uint32_t denominator;
  std::uint32_t max_workgroups;
};

// Fitted to timings taken in Chrome 151 on an M3 Max, outside Frizzle: an
// estimate, not measured in Frizzle (Design and cost, the gate). For Gemma 3
// 1B, Q, K and V split up to 64 tokens, the output and down projections up
// to 96.
inline constexpr SplitGate kSplitGate{2, 1, 64};

// A product's weight as the gate sees it: `rows` outputs of `columns`
// weights each (N and K above), named so the two cannot be swapped.
struct ProductShape {
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
};

// The most tokens a step may hold for the product to split — a whole number
// of 32-token tiles, at most kPrefillBlock — or 1 when it never splits, as a
// gated activation never does. Preconditions: shape.columns is a whole number
// of 32-weight groups; shape.rows > 0; gate.denominator > 0.
[[nodiscard]] std::uint32_t split_tokens(Epilogue epilogue, ProductShape shape,
                                         SplitGate gate) noexcept;

// The floats `range_sums` must hold for the product to split every step the
// gate takes: split_tokens × rows × 32, or 0 when it never splits. A graph
// plans the most of its products'. Preconditions: split_tokens'.
[[nodiscard]] std::uint64_t range_sum_floats(Epilogue epilogue,
                                             ProductShape shape,
                                             SplitGate gate) noexcept;

// The product's launches: for each piece of the weight, a decode launch,
// and for each tile width a tiled launch and, where `gate` splits the
// product, a split launch, each in its part of the width's token range, and
// the split's combine; or for the head, the decode form alone, in every
// step that asks for logits. `range_sums` is the working buffer of that
// name, bound when the product splits: a step splits only while its range
// sums fit it, so a weight in pieces, whose pieces' fewer rows the gate may
// split further than the graph planned for, runs those steps tiled, the same
// bits. Preconditions: the header's Contract.
[[nodiscard]] std::vector<Launch> matmul_launches(
    const MatmulLaunch& matmul, SplitGate gate,
    residency::BufferRange range_sums);

}  // namespace frizzle::kernels
