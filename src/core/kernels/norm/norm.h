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
// logical-overview.md, "Inside one layer"). One kernel for the attention,
// feed-forward and final norms of every architecture here, and for both
// regimes: a row's norm needs only that row.
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
//     memory once; rows up to 4,096 wide, at most 4 vec4s an invocation,
//     cover every listed model (Llama 3.2 1B's 2,048 is the widest). The
//     width is an override constant, so each invocation's loops run
//     ceil(W / 1,024) times, fixed when the pipeline is built — once for
//     Qwen3, twice for Llama 3.2 and Gemma 3 — and the bounds check folds
//     away where 1,024 divides the width. A model has one width, so this
//     adds no pipelines.
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
//   - Accuracy, for finite rows whose squares stay finite in f32 — every
//     activation the listed models produce: each output within 2⁻¹⁸ of the
//     same computation in f64, measured against the largest output of its
//     row. That is the margin a 4,096-wide row's sum leaves (16 sequential
//     adds an invocation, then 8 tree levels, each a rounding of at most
//     2⁻²⁴), with the inverse square root's 2 units and the multiplies; WGSL
//     lets the compiler reassociate and does not fix the rounding direction,
//     so it is a margin the GPU test checks on the target, not a proof.
//     Against the row, not the element: adding y into X can cancel, and an
//     element near zero carries its row's error. A row whose squares
//     overflow f32 — about 10¹⁹ — is outside it; no listed model's
//     activations come near.
//   - Variants are override constants, so one source serves all of them:
//     `add`, and `post_norm` (which requires `add`); `width` is one too.
//     An encoder's (graph/encoder.h) take three more, each changing one
//     launch's form and none of the decoder's:
//       `scaled` (requires `add`): X multiplied by a scalar after the add
//         and before the reduction — Gemma 4's layer output scale, applied
//         to the whole residual stream once a layer's last block is added;
//       `normalize` (default true): false writes X after the add and
//         nothing else — the add before the per-layer gate, which reads X;
//       `rows_per_token` (default 1): a launch covers tokens ×
//         rows_per_token rows of `width`, workgroup w the row w — the
//         per-layer inputs, 24 rows of 512 a token, each normalized on its
//         own. Not with a launch over the last token.
//     An audio encoder's (graph/audio.h) take three more, each changing one
//     launch's form and none of the others':
//       `post_scale` (requires `post_norm`): y, through its post-norm, times
//         the constants' post_scale before the add — the half-step
//         feed-forward's residual weight, 0.5: X + c × postnorm(y). 1
//         elsewhere, and not applied.
//       `renorm` (requires `add`): after the add, X replaced by its own
//         norm with the layer's output gain, written back as X, and then,
//         where the launch normalizes, normalized again with its gain into
//         `normed` — a Conformer layer's closing norm and the next layer's
//         first in one launch. A third reduction where the post-norm makes
//         two; the barrier between each and the next, as between the
//         post-norm's and X's.
//       `unit_gain`: g = 1, its binding bound to the input in its place —
//         the multimodal embedder's RMS norm without a gain, at a width of
//         1,536, which no other norm here has; the width's override covers
//         it, two vec4s an invocation, the second guarded.
//     Every binding is declared in every variant, so a launch that does not
//     add still binds the output buffer and, without a post-norm, a scale or
//     a re-norm, binds its gain in their places — with a unit gain, the
//     output buffer in the gain's: all read-only, so none aliases a buffer
//     the kernel writes.
//   - Capture (capture/capture.h; slice 4, step 5). A launch may write two
//     vectors into the capture store, in the step that ends the prompt
//     (step.logits 1 and step.fed 0) and for the step's last row only:
//       - with `capture_residual`, X as the layer's later steps read it:
//         after the add, the scale and the re-norm where the launch makes
//         them, before it normalizes; where the launch does not add, X as
//         it was read, so layer 0's attention norm captures the embedding.
//         Into the residual slot.
//       - with `capture_added` (requires `add`), y exactly as it was added
//         into X: after the post-norm and post_scale where the launch makes
//         them. Into the added slot. So, where the launch neither scales nor
//         re-norms, X before plus the captured y, one f32 add an element
//         (correctly rounded in WGSL), is X written, bit for bit. WGSL does
//         not say a compiler may not fuse the post-norm's last multiply
//         into the add, which would break it: the identity is what the
//         target gives, which the GPU test checks, as attention's equal
//         bits are, not a promise.
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
//     read-only; 5 the hidden buffer X; 6 the normed buffer; 7 the scale,
//     one f32, read-only; 8 the re-norm's gain, read-only; 9 the capture
//     store, read-write. Eight storage bindings: WebGPU's default limit a
//     stage, reached exactly; a ninth would need the limit raised, which
//     the harness asks for no adapter to do (gpu/device_requirements.h).
//   - Gains are F32 and one piece: a norm's weight is one row of the hidden
//     width, far under a binding (format.h's F32).
//   - It reads the plan's `output` working buffer: a block's last matmul
//     writes its result there, kPrefillBlock rows of the hidden width, and
//     the norm after it adds it into X (residency/plan.h).
//
// What it costs, counted. Launches: two a layer and the final norm — 57 a
// pass for Qwen3 0.6B, about 86 µs of dispatch at 1.5 µs (interface.h).
// Bytes, a row, for Qwen3's 1,024 width: X read and written, y read, the
// gain read, normed written — 4 KiB each, 20 KiB; Gemma 3's post-norm adds
// its gain, 4.5 KiB at its 1,152, 27 KiB a row in all; the first layer's
// attention norm, which does not add, 12 KiB. One norm launch over a 512-row
// prefill step moves about 10 MiB, about 25 µs at 400 GB/s. A pass's 57 —
// one without the add, 55 with it over every row, the final norm's one row —
// move W × (1,112 T + 20) bytes for T rows and width W: about 556 MiB, 1.46
// ms, at T = 512, of which the gains, read again by every row but 4 KiB
// each, are 112 MiB; at decode, 1.1 MiB, under 3 µs, far below the
// launches' 86 µs. A row's latency is its reductions', not its
// bytes: one tree of 8 levels, or for Gemma 3's post-norm two dependent ones
// — y's scale is needed before X can be added to and reduced — 16 levels and
// two inverse square roots.
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
// than 1,024. The last levels, run without barriers within a subgroup, are
// the subgroup sum above. And one
// workgroup a row, one vec4 an invocation for Qwen3: a 4 KiB row split
// across workgroups would need a second launch or atomics to combine its
// sums, slower for a decode step's one row, not faster.
//
// Verification the implementation is held to, on the GPU against an f64
// reference: every variant, at the widths of the listed models; rows of
// large and of tiny magnitude; a step of many rows and of one, each row's
// result identical in both; and the final norm touching only the last row.
// The capture alone has a test of its own, in the kernels suite
// (tests/README.md): the vectors it writes and where, and that X and normed
// are unchanged by it; the norm's arithmetic stays the oracle's to check.

// The encoder's variants, likewise, at EmbeddingGemma 2's width of 512: a
// scaled add, X then normalized; an add alone, normed untouched; and 2 rows a
// token over 2 tokens, each row normalized on its own, with an ε of its own.
// The audio encoder's, at its width of 1,024 and the unit gain's 1,536: a
// post-norm at a scale of 0.5; a re-norm then a norm, and a re-norm alone,
// X written as the re-norm left it; a unit gain, normed equal to the f64
// RMS norm without one.
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
//   C++ Core Guidelines
//     C.12   Don't make data members const or references in a copyable
//            type — the gains are pointers.
//     GPU.8  Make barriers describe real hazards — none after the last
//            reduction, where nothing writes the partial sums again.
//     I.7    State postconditions — the capture's: X before plus the
//            captured y is X written; X and normed unchanged by it.

// The capture store and the slots one norm writes (the header's Capture):
// each slot's first word, or none. A slot is the norm's width long.
struct NormCapture {
  residency::BufferRange store;           // bound always: binding 9
  std::optional<std::uint32_t> residual;  // X, after the add
  std::optional<std::uint32_t> added;     // y as added; requires `add`
};

// One norm of the graph.
struct NormLaunch {
  const residency::WeightView* gain =
      nullptr;  // F32, one row's width; null for a unit gain
  const residency::WeightView* post_gain =
      nullptr;       // Gemma 3's post-norm of y, or null
  bool add = false;  // add y into X first; required by a post-norm
  residency::BufferRange output = {};  // y: the block's output
  residency::BufferRange hidden = {};  // X
  residency::BufferRange normed = {};
  float epsilon = 0.0f;
  Rows rows = Rows::EveryToken;  // LastToken for the final norm
  // Gemma 4's layer output scale: an F32 view of one float that multiplies
  // X after the add, or null.
  const residency::WeightView* scale = nullptr;
  // False: the launch adds and writes X, and nothing else.
  bool normalize = true;
  // Rows a token, each normalized on its own: the per-layer inputs' 24.
  std::uint32_t rows_per_token = 1;
  // What the post-normed y is multiplied by before the add: an audio
  // encoder's half-step 0.5, else 1.
  float post_scale = 1.0f;
  // A Conformer layer's closing norm's gain: X replaced by its norm with it
  // after the add, or null. Requires `add`.
  const residency::WeightView* renorm_gain = nullptr;
  // A unit gain's row width, where `gain` is null; the block's output is
  // then bound in the gain's place, so it must not be X or normed.
  std::uint32_t width = 0;
  // What the launch writes into the capture store, if anything (above).
  NormCapture capture = {};
};

// The launch for one norm. Preconditions: the gains are F32 views of one
// row's width, at most 4,096; a post-norm or a scale only where the launch
// adds; a launch that does not normalize adds; rows_per_token 1 for a launch
// over the last token; `capture.store` the plan's capture store, and each
// slot in it one of capture::layout()'s, of the norm's width; `added` only
// where the launch adds.
[[nodiscard]] Launch norm_launch(const NormLaunch& norm);

}  // namespace frizzle::kernels
