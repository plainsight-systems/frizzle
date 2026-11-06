#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "core/capture/capture.h"
#include "core/formats/format.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace frizzle::graph {

// Axis A: changes with the order of work in a layer, or a new ◉ or ✎ point.
//
// The blocks Gemma 3's graph composes, and the working buffers they bind.
// The graph itself (arch/gemma3/gemma3.cpp, reached through
// arch/architecture.h's GraphFn) is the order Gemma 3 runs them in, and what
// it does that the model description does not carry, such as its embedding
// scale.
// The edit point (✎, docs/architecture/file-mapping.md) is here, in the
// feed-forward block, and the capture points (◉): the neurons just after
// the edit, and the residual, what each block adds and the attention
// weights at the norms and the attention that already hold them.
//
// Contract
//   - Builder appends a step's launches to `out`, block by block, in the
//     order Gemma 3's graph calls them; the program builds from them
//     at load (kernels/program.h). Pure, as the kernel launchers are: no GPU
//     object, no fetch.
//   - Preconditions, as architecture.h's GraphFn states: `model` is a
//     describe's; `plan` was made from it, as upload carried it out;
//     `cache_format` has a pack and is the format the plan sized the cache
//     in. embed is called once, first, and output once, last; `out` outlives
//     the builder. A broken precondition (a plan without a buffer the blocks
//     bind, a tensor the plan did not place, a describe's weight missing)
//     aborts.
//   - Refused, named, with the layer: a layer whose gate and up the plan
//     could not group (UngroupedFeedForward) — formats or widths that
//     differ, a member in pieces, a span wider than a binding; their
//     epilogue reads both in one product, and the plan holds no buffer for
//     either alone. A layer whose Q, K and V the plan could not group
//     (UngroupedQueryKeyValue), for the same reasons; the graph runs them
//     as one product only. A layer outside the kernels' shapes
//     (UnsupportedShape): a head dimension of 64, 128 or 256; (query heads +
//     2 × key-value heads) × head dimension / 8 a multiple of 64; the query
//     heads a key-value head serves dividing 1,024 / head dimension; QK-norm
//     on both queries and keys; a hidden width of at most 4,096; no
//     frequency factors (rope_freqs.weight), which rope does not apply. A
//     vocabulary under the 64 candidates selection keeps.
//   - Residual: the norm that begins a block adds the last block's result
//     into `hidden`, so `hidden` is never written by a matrix product. The
//     builder carries what `output` holds — whether a block has written it,
//     and the post-norm that block's layer applies to it before the add,
//     where the layer has the weight. Post-norms are detected, never assumed.
//   - Buffers: every launch reads working buffers other than the ones it
//     writes, each its own GPU buffer (residency/plan.h), so no dispatch
//     binds one buffer both read-only and writable. WebGPU orders a pass's
//     dispatches by their buffer use, so each launch sees what the one
//     before it wrote.
//   - The edit launch runs over every row of every step and changes only the
//     activation's values that an edit's range holds at the row's position
//     (edit/edit_kernel.h); with a table of no edits it writes nothing, so
//     the graph computes what it computed without it.
//   - The capture launch follows it in every layer: in the step that ends
//     the prompt it copies the activation's last row, as edited, into the
//     layer's neurons slot of `captures` (capture/capture.h).
//   - The other captures, written where their values already are
//     (capture/capture.h), in the step that ends the prompt only, at its
//     last row:
//       - layer l's attention norm: the residual coming into layer l, and,
//         for l > 0, what layer l − 1's feed-forward block added, which
//         this norm adds (kernels/norm/norm.h);
//       - layer l's feed-forward norm: what layer l's attention block
//         added;
//       - the final norm: what the last layer's feed-forward block added;
//       - layer l's attention and its combine: the last row's scores and
//         its maximum and sum (kernels/attention/attention.h), then the
//         weights launch, after the combine, turning them into weights
//         (capture/weights.h).
//     So every word of `captures` is written in that step: each layer's
//     neurons, residual, two additions, weights and stats, the last
//     layer's feed-forward addition by the final norm.
//   - No capture writes a buffer the model reads, and none changes what a
//     launch computes, so the graph computes what it computed without
//     them.
//   - Names: every launch carries its role and layer (kernels/interface.h),
//     for a profile: "embed"; "attention.norm", "attention.qkv",
//     "attention.rope", "attention.scores", "attention.combine",
//     "attention.weights", "attention.output"; "ffn.norm", "ffn.gate_up",
//     "ffn.edit", "ffn.capture", "ffn.down"; and "output.norm", "output.head",
//     "output.select", "output.draw", which belong to no layer.
//   - Determinism (GDSA.2): every kernel the blocks launch is
//     batch-invariant — the gather, norms and rope work a row alone,
//     attention folds chunks fixed by position, each matrix product adds in
//     an order fixed by K, the edit and copy kernels add nothing, the
//     weights launch divides by the attention fold's sum — and
//     no block adds a reduction of its own. So a token's logits are the same
//     bits whether its prompt was prefilled in one step, in several, or decoded
//     a token at a time, as each kernel states.
//
// Algorithm, the launches each block appends, in order
//   - embed(scale): the token embedding's gather, one launch a piece, into
//     `hidden`, each weight times `scale` (kernels/gather/gather.h).
//   - attention(layer):
//       1. Attention norm: `output` added into `hidden` and `hidden`
//          normalized into `normed` with the layer's gain — without the add
//          before any block has written `output` (kernels/norm/norm.h).
//          Captures the residual into the layer's slot, and what it added
//          into the last layer's mlp_added slot.
//       2. Q, K and V from `normed`: the plan's group as one product into
//          `query`, `key` and `value` (kernels/matmul/matmul.h).
//       3. QK-norm, RoPE, and the key and value
//          appended to the layer's cache (kernels/rope/rope.h).
//       4. Attention over the layer's cache into `attention`, and its
//          combine (kernels/attention/attention.h); both capture into the
//          layer's weights and stats slots.
//       5. The weights launch, over the layer's weights and stats slots
//          (capture/weights.h).
//       6. Output projection: `attention` into `output`.
//   - gated_feed_forward(layer):
//       7. Feed-forward norm: `output` added into `hidden`, normalized into
//          `normed`. Captures what it added into the layer's
//          attention_added slot.
//       8. Gate and up from `normed`, the plan's group as one product writing
//          activation(gate) × up into `activation`.
//       9. The edit: the run's edits for the layer, from `edit_table`,
//          applied to each row of `activation` whose position their range
//          holds (edit/edit_kernel.h).
//      10. The capture: `activation`'s last row into the layer's neurons
//          slot of `captures`, in the step that ends the prompt
//          (capture/capture.h).
//      11. Down: `activation` into `output`.
//   - output(): the final norm, `output` added into `hidden` and normalized
//     with the output norm's gain, capturing what it added into the last
//     layer's mlp_added slot; the head, `normed` into `logits` — the
//     output head's weight, or the token embedding's where the file ties
//     them; top-k selection, `logits` to `candidates` (kernels/topk/topk.h);
//     and the draw, `candidates` to the record in `sampled`
//     (sampler/sampler.h). All cover the step's last token only, and run
//     only in a step that asks for logits.
//   A matrix product is all its launcher's launches — a decode launch and
//   one a prefill tile width, for each piece — and attention its main launch
//   and its combine; the program dispatches those a step's token count,
//   position and logits select (kernels/interface.h).
//
// Design and cost
//   - Composition, not a base class holding the order: Gemma 3's graph is a
//     function that calls these blocks, reached through architecture.h's
//     function pointer (C.129).
//   - The edit launch is the graph's own, always present: one a layer, over
//     every row, in every step. Its cost, accepted (logical-overview.md,
//     Edits): 26 more dispatches a step for Gemma 3 1B, about 40 µs at
//     about 1.5 µs a dispatch, an estimate, not measured in Frizzle.
//     Rebuilding the graph for each edit set, or folding the edit into
//     gate·up's epilogue, were the alternatives (edit/edit_kernel.h).
//   - The capture launch is the graph's own too, always present: one a
//     layer, over the last token, after the edit launch. It adds 26 more
//     dispatches in each scoring step for Gemma 3 1B, which in decode read
//     one word and stop: about 40 µs a token at the same estimate, not
//     measured. The weights launch adds 26 more a scoring step, the same
//     again. `captures` is one working buffer of every slot, 3.2 MiB,
//     planned in decoder_working_set. The alternatives are in
//     capture/capture.h.
//   - Counted from the launchers' code for Gemma 3 1B at Q4_0 with an F16
//     cache, not from a run: a decode step dispatches 11 launches a layer —
//     the attention norm, the one QKV product, rope, attention, the weights
//     launch, the output projection, the feed-forward norm, gate and up,
//     the edit, the capture, down — and 12 where the layer's key chunks
//     split, its combine; with the embedding's 3, one a piece of the table,
//     and the output block's 8 — the final norm, the head's 3 pieces,
//     selection's 3 passes, the draw — 297 a step, 323 when every layer's
//     combine runs, as it does past position 63 (GPU.6). At about 1.5 µs a
//     dispatch (an estimate, not measured in Frizzle), 0.45 to 0.48 ms of
//     GPU time. A prefill step runs each product's tile and split launches
//     for its token count instead (kernels/matmul/matmul.h), not counted
//     here. Pipelines, bind groups and launch constants are made once, at
//     load (kernels/program.h).
//   - Optimization (practice): logits only for a step whose last token is
//     sampled, the only logits the sampler reads.
//     A prefill step that does not end the prompt skips every launch over
//     the last token alone — the final norm, the head, selection, the draw,
//     and the capture and weights launches (GDSA.6): for Gemma 3 1B the head
//     is the Q8_0 token embedding, about 321 MB not read.
//   - The graph is built at preflight and again at load, never per step;
//     a step makes no call through it (WASM.4).
//   - Verified in Frizzle by the llama.cpp comparison (tests/README.md,
//     integration): Gemma 3 1B decoded a token at a time over a pinned
//     600-token sequence, every position's top 20 within llama.cpp's own
//     CPU-to-Metal spread, through this graph with an empty edit table.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.129  Distinguish implementation inheritance from interface
//            inheritance: no inheritance at all; the graph composes the
//            blocks, and its interface is architecture.h's function
//            pointer.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work: 11 dispatches a layer in the count
//            above, the edit, capture and weights launches' tiny ones
//            among them, the edit's fused alternative named.
//     GDSA.6 Count passes over global memory: the head's read, and the
//            other launches over the last token alone, skipped where no
//            token is scored.
//     GDSA.2 Declare each floating-point reduction's determinism level:
//            batch-invariant end to end, as above.
//     WASM.4 Reduce indirect dispatch in hot paths: the graph's function
//            pointer is called at preflight and load, never per step.

enum class GraphError : std::uint8_t {
  Ok,
  // A layer's gate and up are not one group in the plan. The subject names
  // the layer.
  UngroupedFeedForward,
  // A layer's Q, K and V are not one group in the plan. The subject names
  // the layer.
  UngroupedQueryKeyValue,
  // A layer's shape is outside a kernel's preconditions. The subject names
  // the layer and the value.
  UnsupportedShape,
};

struct GraphResult {
  GraphError error = GraphError::Ok;
  std::string subject;

  [[nodiscard]] bool ok() const noexcept { return error == GraphError::Ok; }
};

// The working buffers the blocks above bind, for the residency plan to place
// (residency/plan.h), in this order, each kPrefillBlock rows at f32 unless
// stated: `hidden`, `normed` and `output`, the hidden width; `query` and
// `attention`, the widest layer's query heads × head dimension; `key` and
// `value`, its key-value heads × head dimension; the split attention's
// `partials` and `partial_stats`, partial_rows_for the trained context
// (kernels/attention/attention.h); `activation`, the widest feed-forward;
// one row of `logits`, the vocabulary; selection's `partials_a` and
// `partials_b` and the `candidates` it keeps, (logit, token) pairs of two
// words (kernels/topk/topk.h); the draw's 16-byte record, `sampled`
// (sampler/sampler.h); `range_sums`, one row of the most
// kernels::range_sum_floats any of its products needs under
// kernels::kSplitGate — Q, K and V fused, the output and down projections —
// at least one float: 13.5 MiB for Gemma 3; and
// `edit_table`, one row of 2 × layers + 3 × edit::kMaxEdits u32 words, the
// run's edits (edit/edit.h), 48 KiB for Gemma 3 1B; and `captures`, one row
// of capture::layout's words, every layer's captures at the last prompt
// position (capture/capture.h), 3.2 MiB for Gemma 3 1B. Every product's
// launches take that gate and that buffer (kernels/matmul/matmul.h). A decoder
// keeps a KV cache. Which buffers a graph binds is the graph's to say, and
// where they lie the planner's (docs/architecture/change-axes.md).
//
// Selection and the draw have four working buffers of their own
// (sampler/sampler.h): `partials_a` and `partials_b`, which its passes
// alternate between — `partials_a` the first pass's output, kCandidates
// (logit, token) pairs of 8 bytes for every kSelectionTile of the
// vocabulary, 131 KB for Gemma 3; `partials_b` the second's, kCandidates
// for every kSelectionTile of those, 8 KB; each later pass writes fewer than
// the one two before it, into the same buffer — `candidates`, the kCandidates
// kept, 512 bytes; and `sampled`, the draw's 16-byte record. Selection's last
// pass writes `candidates` in place of a partials buffer, so for a vocabulary
// of 16,384 or fewer — Gemma 3's is 262,144 — no pass writes `partials_b`, and
// for one of 1,024 or fewer none writes `partials_a`; each is still planned,
// 512 bytes at least, so every plan holds the same working buffers.
[[nodiscard]] residency::WorkingSet decoder_working_set(
    const model::ModelDescription& model);

// The most range sums any of a decoder layer's products needs split along K,
// in floats, under kernels::kSplitGate.
[[nodiscard]] std::uint64_t layer_range_sums(
    const model::LayerDescription& layer, std::uint64_t width);

// Appends a step's launches to `out`, block by block, in the order Gemma 3's
// graph calls them. It holds what `output` holds between
// blocks (above). Preconditions, as architecture.h's GraphFn states: `model`
// is a describe's; `plan` was made from it, as upload carried it out;
// `cache_format` has a pack and is the format the plan sized the cache in.
// embed is called once, first, and output once, last; `out` outlives the
// builder.
class Builder {
 public:
  Builder(const model::ModelDescription& model,
          const residency::ResidencyPlan& plan,
          const formats::Format& cache_format,
          std::vector<kernels::Launch>& out);

  void embed(float scale);
  [[nodiscard]] GraphResult attention(std::uint32_t layer);
  [[nodiscard]] GraphResult gated_feed_forward(std::uint32_t layer);
  [[nodiscard]] GraphResult output();

 private:
  // What `output` holds when the next norm adds it.
  struct Pending {
    bool written = false;  // no block yet: the first norm does not add
    const residency::WeightView* post_gain =
        nullptr;  // the writer's post-norm, or null
    // The added slot of the block that wrote it, which the next norm
    // captures what it adds into: a layer's attention_added or mlp_added.
    std::optional<std::uint32_t> added_slot;
  };

  // The plan's working buffers, by what they hold.
  struct Buffers {
    residency::BufferRange hidden, normed, query, key, value, attention,
        partials, partial_stats, output, activation, logits, partials_a,
        partials_b, candidates, sampled, range_sums, edit_table, captures;
  };

  [[nodiscard]] const residency::WeightView& view(gguf::TensorId tensor) const;
  // The layer's weight in `role`, or null where it has none.
  [[nodiscard]] const residency::WeightView* role(std::uint32_t layer,
                                                  model::Role role) const;
  // The plan's group whose members, in order, are `members`, or null.
  [[nodiscard]] const residency::PlannedGroup* group(
      std::initializer_list<std::optional<gguf::TensorId>> members) const;
  void product(const residency::WeightView& weight,
               const residency::BufferRange& input,
               const residency::BufferRange& output, kernels::Rows rows);

  const model::ModelDescription* model_;
  const residency::ResidencyPlan* plan_;
  const formats::Format* cache_format_;
  std::vector<kernels::Launch>* out_;
  std::vector<const residency::WeightView*>
      views_;  // by tensor id; null for a tensor the plan did not place
  Buffers buffers_;
  std::vector<capture::LayerSlots> slots_;  // each layer's in `captures`
  Pending pending_;
};

}  // namespace frizzle::graph
