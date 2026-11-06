#pragma once

#include <cstdint>
#include <optional>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace frizzle::kernels {

// Axis E: changes with a new or optimized kernel.
//
// RMSNorm, with the residual add before it fused in (docs/architecture/
// logical-overview.md, "Inside one layer"). One kernel for Gemma 3 1B's
// attention, feed-forward and final norms, and for both regimes: a row's norm
// needs only that row.
//
// For each row it covers:
//   1. Where the launch adds: y, the last block's output, is added into X,
//      the hidden row, and X written back. Where the architecture normalizes
//      a block's output before adding it — Gemma 3's post-attention and
//      post-feed-forward norms — y is first normalized with its own gain.
//   2. normed = (X × r) × g, where r = 1 / sqrt(mean(X²) + ε) and g is the
//      norm's gain: normalize, then multiply by the gain, in that order. ε is
//      the file's (model description). Gemma 3's gains need no special case:
//      the GGUF file stores them as 1 + w (arch/architecture.h).
// The first layer's attention norm does not add — the embedding wrote X —
// and the final norm covers only the step's last token, row tokens - 1,
// which is all the output head reads (interface.h).
//
//   - One workgroup of 256 invocations per row. Invocation i takes the row's
//     vec4s i, i + 256, …, holding them in registers, so X is read from
//     memory once; rows up to 4,096 wide, at most 4 vec4s an invocation;
//     Gemma 3 1B's are 1,152, 288 vec4s. The width is an override constant,
//     so each invocation's loops run ceil(W / 1,024) times, fixed when the
//     pipeline is built — twice for Gemma 3 1B — and the bounds check can
//     fold away only where 1,024 divides the width, which 1,152 is not. A
//     model has one width, so this adds no pipelines.
//   - The sum of squares is reduced in workgroup memory, in a fixed tree:
//     each invocation's own vec4s in order, then halving across the
//     workgroup, 8 levels. No atomics (GDSA.5). A barrier where one
//     invocation reads what another wrote, and nowhere else: one before the
//     tree and one after each level, 9 a reduction; one more only between
//     the post-norm's reduction and the next, which rewrites the partial sum
//     the first was read from (GPU.8).
//   - Determinism (GDSA.2): run to run, and independent of the step's other
//     rows and its token count, since one workgroup reduces one row in one
//     fixed order — so a token's normed row, and the key and value computed
//     from it, are the same whether the token was prefilled or decoded. Not
//     bit for bit with other implementations or across GPUs, which sum in
//     other orders.
//   - Accuracy, for finite rows whose squares stay finite in f32: each
//     output within 2⁻¹⁸ of the same computation in f64, measured against
//     the largest output of its row. That is the margin a 4,096-wide row's sum
//     leaves (16 sequential adds an invocation, then 8 tree levels, each a
//     rounding of at most 2⁻²⁴), with the inverse square root's 2 units and the
//     multiplies; WGSL lets the compiler reassociate and does not fix the
//     rounding direction, so it is a design margin, not a proof, and no test
//     checks it. Against the row, not the element: adding y into X can cancel,
//     and an element near zero carries its row's error. A row whose squares
//     overflow f32 — elements of about 10¹⁹ — is outside it; how near
//     Gemma 3 1B's activations come is not measured here.
//   - Variants are override constants, so one source serves all of them:
//     `add`, and `post_norm` (which requires `add`); `width` is one too.
//   - post_scale is not a variant but a constant (binding 1): wherever the
//     launch has a post-norm, y through its post-norm is multiplied by it at
//     run time before the add, X + c × postnorm(y). Gemma 3's graph always
//     passes 1, NormLaunch's default, so the multiply changes no value. It
//     is kept for the bits. Measured (Dawn native, this machine, the
//     oracle's fixed tokens): without it, the oracle's worst deviation moved
//     from 0.044009 to 0.0402013 nats, the tokens unchanged. Inferred: the
//     multiply keeps the shader compiler from fusing the post-norm's gain
//     multiply into the residual add, so the rounding differs without it.
//     It is kept so the output stays bit for bit with the recorded
//     baseline. No test locks it.
//   - Every binding is declared in every variant, so a launch that does not
//     add still binds the output buffer and, without a post-norm, binds its
//     gain in the post-norm's place: both read-only, so neither aliases a
//     buffer the kernel writes.
//   - Capture (capture/capture.h). A launch may write two vectors into the
//     capture store, in the step that ends the prompt (step.logits 1 and
//     step.fed 0) and for the step's last row only:
//       - with `capture_residual`, X as the layer's later steps read it:
//         after the add where the launch makes it, before it normalizes;
//         where the launch does not add, X as it was read, so layer 0's
//         attention norm captures the embedding.
//         Into the residual slot.
//       - with `capture_added` (requires `add`), y exactly as it was added
//         into X: after the post-norm and its multiply by post_scale where
//         the launch has a post-norm. Into the added slot. So X before plus
//         the captured y, one f32 add an element (correctly rounded in
//         WGSL), is X written, bit for bit. WGSL does not say a compiler may
//         not fuse the last multiply before the add into the add, which would
//         break it: the identity is what the target gives, which the capture
//         test checks, as attention's equal bits are, not a promise.
//     Writing changes nothing the launch computes: X and normed are the same
//     bits whether the step captures or not. No other step and no other row
//     writes to the store. Which launch writes which layer's slots is the
//     graph's (graph/graph.h).
//     The step's `logits`, which no kernel read before, tells the step that
//     ends the prompt from a prefill step that does not: neither is fed,
//     and a norm runs over every row of every step (kernels/step.wgsl).
//     The two are override constants, so a launch that captures neither
//     pays nothing but a binding; each slot's first word is a constant.
//     Cost: a test of the step a row; in the step that ends the prompt, one
//     row's width written once or twice, 4.5 KiB each at Gemma 3's 1,152.
//     Why here and not a copy launch: y after the post-norm exists only in
//     this kernel's registers, never in a buffer, and X after minus X
//     before is not y in f32; X is in the same registers, so writing it
//     costs no dispatch, where a copy would cost one a layer a token.
//   - Constants (binding 1), as WGSL lays them out:
//       struct Norm { epsilon: f32, post_scale: f32, residual_slot: u32,
//                     added_slot: u32 }
//     Bindings: 2 the gain, 3 the post-norm's gain, 4 the block's output y,
//     read-only; 5 the hidden buffer X; 6 the normed buffer; 7 the capture
//     store, read-write. Six storage bindings, under WebGPU's default limit
//     of eight a stage (gpu/device_requirements.h).
//   - Gains are F32 and one piece: a norm's weight is one row of the hidden
//     width, far under a binding (format.h's F32).
//   - It reads the plan's `output` working buffer: a block's last matmul
//     writes its result there, kPrefillBlock rows of the hidden width, and
//     the norm after it adds it into X (residency/plan.h).
//
// What it costs, counted, for Gemma 3 1B, width W = 1,152, 4.5 KiB a row.
// Launches: two a layer and the final norm — 53 a pass, about 80 µs of
// dispatch at about 1.5 µs each, an estimate, not measured in Frizzle
// (interface.h). Bytes, a row: a launch that adds, with Gemma 3's post-norm,
// reads X, y, the gain and the post-norm's gain and writes X and normed —
// 6 × 4.5 KiB, 27 KiB; the first layer's attention norm, which does not add,
// reads X and the gain and writes normed, 13.5 KiB. One norm launch over a
// 512-row prefill step moves 13.5 MiB. A pass's 53 — one without the add, 51
// with it over every row, the final norm's one row — move 4W × (309 T + 6)
// bytes for T rows: about 695 MiB at T = 512, of which the gains, read again
// by every row but 4.5 KiB each, are about 232 MiB; at decode, 1.4 MiB. At
// 400 GB/s, the M3 Max's memory bandwidth, those are about 35 µs, 1.8 ms
// and 3.6 µs, estimates, not measured; a decode step's bytes are far below
// its launches' 80 µs. A row's latency is its reductions', not its bytes:
// with Gemma 3's post-norm, two dependent trees of 8 levels — y's scale is
// needed before X can be added to and reduced — 16 levels and two inverse
// square roots; one tree for the first layer's attention norm.
// Optimization (practice): the residual add and the gain ride in the norm's
// launch, rather than in launches of their own (GPU.6).
// Optimization (practice): the width fixed at pipeline creation, not read
// from a uniform, so the loops run as many times as the row needs, not 4
// (GDSA.6's whole-tile accounting: an inactive iteration still costs its
// instructions). Gemma 3's 288 vec4s still leave 224 invocations idle in
// the second.
//
// Where WebGPU limits it, and what each limit costs here:
//   - Subgroup operations (subgroupAdd) are the `subgroups` feature, which
//     WebGPU leaves optional and not every browser offers. For cross-browser
//     compatibility the harness requires only WebGPU's defaults and no
//     optional feature (gpu/device_requirements.h), though the target
//     itself offers subgroups. So the reduction is the tree above, 9 barriers,
//     where a subgroup reduction — a subgroup sum, the partials through
//     workgroup memory, a subgroup sum of those — needs 1.
//   - WGSL zero-fills workgroup memory before a workgroup runs, so each one
//     also stores its 1 KiB of partial sums and waits at one more barrier
//     before Step 1, though the kernel writes every entry before reading it.
// Within those limits the reduction is a tree without subgroups: sequential
// addressing, so active invocations stay contiguous; each invocation's own
// vec4s summed before the tree; several vec4s an invocation for rows wider
// than 1,024. The last levels, within one subgroup, still take a barrier
// each, where a subgroup sum would take none. And one workgroup a row, two
// vec4s an invocation for Gemma 3 1B: a 4.5 KiB row split across workgroups
// would need a second launch or atomics to combine its sums, slower for a
// decode step's one row, not faster.
//
// Tests: the norm's arithmetic has no test of its own; the oracle covers it
// end to end on the pinned model (tests/README.md). The capture alone has a
// test, in the kernels suite: the vectors it writes and where, and that X
// and normed are unchanged by it. No unit test checks the accuracy margin
// above, a row's independence of its step (replay's exact-replay checks
// cover that end to end), or the final norm touching only the last row
// apart from its capture; no test at all locks the multiply by post_scale.
//
// Guidelines, by corpus:
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level — as
//            above.
//     GDSA.5 Aggregate within the workgroup before touching global memory —
//            no atomics; one write a row.
//     GDSA.6 Count passes over global memory — the bytes above; and no
//            loop iterations beyond the row's.
//     GPU.5  Use workgroup memory where reuse or reordering pays — the
//            reduction's partial sums, 1 KiB.
//     GPU.6  Batch tiny GPU work — the add and the gain fused in, and the
//            capture's writes ride in the same launch.
//     GPU.2  Shape data for coalesced lane access — the capture's stores
//            are not: an invocation writes its vec4's four words, a slot
//            being word-aligned only, so adjacent lanes are 16 bytes
//            apart. One row once a run, 4.5 KiB; not worth a shuffle.
//     GPU.8  Make barriers describe real hazards — none after the last
//            reduction, where nothing writes the partial sums again.
//   C++ Core Guidelines
//     C.12   Don't make data members const or references in a copyable
//            type — the gains are pointers.
//     I.7    State postconditions — the capture's: X before plus the
//            captured y is X written; X and normed unchanged by it.

// The capture store and the slots one norm writes (the header's Capture):
// each slot's first word, or none. A slot is the norm's width long.
struct NormCapture {
  residency::BufferRange store;           // bound always: binding 7
  std::optional<std::uint32_t> residual;  // X, after the add
  std::optional<std::uint32_t> added;     // y as added; requires `add`
};

// One norm of the graph.
struct NormLaunch {
  const residency::WeightView* gain = nullptr;  // F32, one row's width
  const residency::WeightView* post_gain =
      nullptr;       // Gemma 3's post-norm of y, or null
  bool add = false;  // add y into X first; required by a post-norm
  residency::BufferRange output = {};  // y: the block's output
  residency::BufferRange hidden = {};  // X
  residency::BufferRange normed = {};
  float epsilon = 0.0f;
  Rows rows = Rows::EveryToken;  // LastToken for the final norm
  // What the post-normed y is multiplied by before the add, at run time,
  // wherever the launch has a post-norm. Gemma 3's graph leaves it 1: the
  // multiply is kept for the bits (above).
  float post_scale = 1.0f;
  // What the launch writes into the capture store, if anything (above).
  NormCapture capture = {};
};

// The launch for one norm. Preconditions: `gain` is not null; the gains are
// F32 views of one row's width, at most 4,096; a post-norm only where the
// launch adds; `capture.store` the plan's capture store, and each
// slot in it one of capture::layout()'s, of the norm's width; `added` only
// where the launch adds.
[[nodiscard]] Launch norm_launch(const NormLaunch& norm);

}  // namespace frizzle::kernels
