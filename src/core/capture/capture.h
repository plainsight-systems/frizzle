#pragma once

// Axis O: changes with a new captured value, or what is sent.
//
// The capture store and its copy launch: at the last position of the step
// that ends the prompt, for every layer, what the model used, written
// inside the pass into slots of their own, so the page is shown what the
// run computed and the buffers they came from may be reused by the next
// layer (docs/architecture/logical-overview.md, "Inside one layer").
//
// What a layer's slots hold, all at that one position (slice 4, step 5):
//   - neurons: the gated activation after the layer's edits, as the down
//     projection reads it (this file's copy launch);
//   - residual: the residual stream coming into the layer, X after the
//     add the layer's first norm makes, before it normalizes (the norm
//     kernel's capture, kernels/norm/norm.h); layer 0's is the embedding;
//   - attention_added and mlp_added: what each block adds into the
//     residual, its output after its post-norm, exactly the vector the
//     norm kernel adds (kernels/norm/norm.h);
//   - weights: each query head's attention weights over the keys it read,
//     head-major, `keys` a head (capture/weights.h);
//   - stats: each query head's softmax maximum and sum, the fold's
//     (kernels/attention/attention.h), which the weights were divided by.
// Q, in the plan with the pass map, is not captured.
//
// Contract
//   - layout(model): where every slot lies in the store, in 32-bit words.
//     Pure. The store is two regions:
//       1. Every layer's neurons, in layer order, layer l's slot after the
//          earlier layers', its feed-forward width long: `neuron_words` in
//          all. Kept first and contiguous, so a host that sends the
//          neurons alone sends one range (wasm/bindings.cpp).
//       2. Then, for each layer in order, its residual, attention_added and
//          mlp_added (the embedding width each), its weights (query heads ×
//          keys) and its stats (query heads × 2: m then l a head), in that
//          order, each after the last.
//     keys = min(attention_window, trained_context): the most keys a query
//     of the layer reads in any context the plan can offer, so the store is
//     sized from the model alone, before the plan picks its context, as the
//     attention's partial buffers are (graph/graph.h).
//     A store past 2³² words, 16 GiB, which no plan could place, aborts:
//     a slot's first word is a u32 constant of the launches that write it.
//     For Gemma 3 1B (26 layers, width 1,152, 4 query heads, windows 512
//     and, in 4 global layers, the trained 32,768):
//       neurons  26 × 6,912                         179,712 words
//       residual and the two additions 26 × 3 × 1,152 89,856
//       weights  22 × 4 × 512 + 4 × 4 × 32,768      569,344
//       stats    26 × 8                                 208
//       the store                                   839,120 words, 3.2 MiB
//   - capture_launch(source, store, slot): a launch that copies one row of
//     `slot.width` words, binding the source (binding 2) and the store
//     (binding 3). In the step that ends the prompt, and only there, it
//     copies the source's last row into the slot, word for word: the bits
//     of every value, NaN, infinity, negative zero and subnormals included.
//     It writes nothing outside the slot, and in any other step nothing at
//     all. The graph uses it for the neurons alone.
//   - "The step that ends the prompt": the step that asks for logits with
//     its tokens given, not fed from the last draw (kernels/interface.h,
//     Step: logits 1, fed 0). The turn loop has exactly one a turn, the last
//     prefill step, and recomputes the last prompt position every turn,
//     even when the cache holds the whole prompt (runtime.h, cache/prefix.h).
//     Every decode step is fed; a prefill step that does not end the prompt
//     asks for no logits. Every kernel that writes the store writes it in
//     that step only, at its last row.
//   - Every word of the store is written in the step that ends the prompt,
//     by the launches the graph places (graph/graph.h): so a store read
//     after that step holds that step's values and nothing of an earlier
//     run's.
//   - The graph places the neurons' copy after the layer's edit launch and
//     before its down projection, so an edited neuron is captured with the
//     value it was set to: what the model used.
//   - Preconditions of capture_launch: `slot` is one of layout()'s for the
//     model the graph is built for; the source's rows are the slot's width;
//     the store is layout()'s total words.
//   - The store keeps its words until a later step that ends a prompt
//     writes them. It is not cleared between runs: which run it holds is
//     the engine's to say (runtime/engine.h), never the store's.
//
// Algorithm, the copy kernel (capture.wgsl), one workgroup of 256
//   1. If the step does not end the prompt (fed is 1), stop.
//   2. Invocation j copies words j, j + 256, ... below the slot's width:
//      store[first + i] = source[(tokens − 1) × width + i], as u32, so no
//      value passes through a float register.
//
// Design and cost
//   - The copy is a launch over the last token (Rows::LastToken), so the
//     program runs it only in steps that ask for logits (kernels/
//     schedule.h); the kernel itself tells the step that ends the prompt
//     from a decode step by `fed`, which the step already carries.
//   - The residual and the additions are written by the norm kernel, not
//     copied: what a block adds exists only in the norm's registers, never
//     in a buffer, and X after the add minus X before is not it in f32. The
//     norm holds the residual in registers at the same point, so writing it
//     there costs no dispatch, where a copy would cost 26 a token.
//   - The weights are the attention kernel's scores and statistics,
//     normalized by their own launch (capture/weights.h): the attention
//     kernel never forms a normalized weight (GDSA.16).
//   - Bound as arrays of u32 in the copy: the activation is f32 to every
//     other kernel, and a buffer's binding type is each pipeline's own. A
//     copy by words keeps every bit, which a float load and store need not
//     do for a NaN's payload.
//   - The store is one working buffer the plan sizes at load, a scratch need
//     of layout()'s total words (graph decoder_working_set), its own GPU
//     buffer (residency/plan.h), so no dispatch binds one buffer both
//     read-only and writable. A slot is a word offset in a launch's
//     constants, not a binding offset: a storage binding's offset must be a
//     multiple of 256 bytes, which a slot need not be.
//   - Cost, counted for Gemma 3 1B (not measured):
//       - The step that ends the prompt: the neurons' 26 copies, 702 KiB;
//         the norms' writes, 351 KiB; the scores' and weights' 2.2 MiB,
//         most of it the global layers' zeros past the prompt. Once a run.
//       - Every decode step: the copy's 26 dispatches and the weights' 26
//         (capture/weights.h), each reading the step and stopping: 52 at
//         about 1.5 µs a dispatch (an estimate, not measured in Frizzle),
//         about 80 µs a token. The norm and attention kernels add a test of
//         the step to launches that run anyway.
//       - Memory: 3.2 MiB on the device for the store, and the reader's
//         (capture/reader.h).
//   - Alternatives considered:
//       - Fold the neurons' copy into the edit kernel, which runs at the
//         same point: no dispatch added. But one kernel would then change
//         with two axes, edits (Q) and captures (O).
//       - A launch the schedule skips in fed steps: no dispatch in decode.
//         But a new kind of geometry in the ported schedule and program
//         (axis D), to save about 80 µs a token, and speed is not this
//         demo's aim.
//       - Size the global layers' weights to the context the plan offers,
//         not the trained context: smaller for a small offered context,
//         but the store would be sized after the plan fits the context,
//         which places its working buffers first.
//       - Read back only the keys a run's last position read: up to 2 MiB
//         less a run for a short prompt, for a reader that knows the
//         layout. Not done until a run's read back is seen to matter.
//
// Corpus guidance
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work: the launches are tiny and run in every
//            scoring step; their cost is counted above, and the residual's
//            folded into the norm that already holds it.
//     GPU.1  Keep data on the device: the copies stay on the GPU inside the
//            pass; the store crosses to the CPU once a run, at its end
//            (capture/reader.h).
//     GPU.9  Suballocate: the store is one working buffer in the plan, not
//            one a slot.
//     GDSA.16 Never materialize the large intermediate: the attention
//            kernel still forms no score matrix; it writes the last row's
//            scores alone, one row of keys a head.
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed: a layer's slots
//            are named fields, not an index into an array of kinds.
//     I.5    State preconditions: the slot is layout()'s, and the widths
//            match.
//     I.7    State postconditions: every word of the store is written in
//            the step that ends the prompt.
//     ES.103 Don't overflow: layout() sums offsets in 64 bits and aborts
//            before a slot's first word passes a u32.
//     ES.46  Avoid lossy arithmetic conversions: each slot narrows to u32
//            only after that check.

#include <cstdint>
#include <vector>

#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace frizzle::capture {

// A range of the store, in 32-bit words.
struct Slot {
  std::uint32_t first;
  std::uint32_t width;
};

// One layer's slots (the header's list).
struct LayerSlots {
  Slot neurons;          // feed-forward width
  Slot residual;         // embedding width
  Slot attention_added;  // embedding width
  Slot mlp_added;        // embedding width
  Slot weights;          // query heads × keys; head h, key i at h × keys + i
  Slot stats;            // query heads × 2; head h's m at 2h, its l at 2h + 1
  std::uint32_t keys;    // min(attention_window, trained_context)
  std::uint32_t query_heads;
};

struct Layout {
  std::vector<LayerSlots> layers;  // one a layer, in layer order
  std::uint64_t neuron_words = 0;  // region 1: every layer's neurons
  std::uint64_t words = 0;         // the store's size
};

[[nodiscard]] Layout layout(const model::ModelDescription& model);

[[nodiscard]] kernels::Launch capture_launch(
    const residency::BufferRange& source, const residency::BufferRange& store,
    Slot slot);

}  // namespace frizzle::capture
