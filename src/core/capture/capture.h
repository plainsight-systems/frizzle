#pragma once

// Axis O: changes with a new captured value, or what is sent.
//
// The capture store and its launch: every layer's neurons at the last prompt
// position, as the model used them, copied inside the pass into a slot of
// their own, so the page is shown what the run computed and the buffer they
// came from may be reused by the next layer (docs/architecture/
// logical-overview.md, "Inside one layer"). Slice 4 captures the neurons
// alone; the residual and Q join with the pass map (slice 5).
//
// Contract
//   - layout(model): where each layer's slot lies in the store, in 32-bit
//     words: layer l's slot begins at the sum of the earlier layers'
//     feed-forward widths and is its own width long; the store is the sum of
//     every layer's. Pure. For Gemma 3 1B, 26 slots of 6,912: 179,712 words,
//     718,848 bytes.
//   - capture_launch(activation, store, slot): a layer's launch, binding
//     the activation (binding 2) and the store (binding 3); the slot is the
//     layer's, so the launch needs nothing else of it. In the
//     step that ends the prompt, and only there, it copies the activation's
//     last row into the layer's slot, word for word: the bits of every value,
//     NaN, infinity, negative zero and subnormals included, as the down
//     projection reads them. It writes nothing outside the slot, and in any
//     other step nothing at all.
//   - "The step that ends the prompt": the step that asks for logits with
//     its tokens given, not fed from the last draw (kernels/interface.h,
//     Step: logits 1, fed 0). The turn loop has exactly one a turn, the last
//     prefill step, and recomputes the last prompt position every turn,
//     even when the cache holds the whole prompt (runtime.h, cache/prefix.h).
//     Every decode step is fed; a prefill step that does not end the prompt
//     asks for no logits.
//   - The graph places the launch after the layer's edit launch and before
//     its down projection (graph/graph.h), so an edited neuron is captured
//     with the value it was set to: what the model used.
//   - Preconditions: `slot` is layout()'s for a layer of the model the graph
//     is built for; the activation's rows are the slot's width; the store is
//     layout()'s total words.
//   - The store keeps a slot's words until a later step that ends a prompt
//     writes it. It is not cleared between runs: which run it holds is the
//     engine's to say (runtime/engine.h), never the store's.
//
// Algorithm, the kernel (capture.wgsl), one workgroup of 256
//   1. If the step does not end the prompt (fed is 1), stop.
//   2. Invocation j copies words j, j + 256, ... below the slot's width:
//      store[first + i] = activation[(tokens − 1) × width + i], as u32, so
//      no value passes through a float register.
//
// Design and cost
//   - A launch over the last token (Rows::LastToken), so the program runs it
//     only in steps that ask for logits (kernels/schedule.h); the kernel
//     itself tells the step that ends the prompt from a decode step by
//     `fed`, which the step already carries. No new field in the step, no
//     change to the schedule.
//   - Both bound as arrays of u32: the activation is f32 to every other
//     kernel, and a buffer's binding type is each pipeline's own. A copy by
//     words keeps every bit, which a float load and store need not do for a
//     NaN's payload.
//   - The store is one working buffer the plan sizes at load, a scratch need
//     of layout()'s total words (graph decoder_working_set), its own GPU
//     buffer (residency/plan.h), so no dispatch binds one buffer both
//     read-only and writable. The slot is a word offset in the launch's
//     constants, not a binding offset: a storage binding's offset must be a
//     multiple of 256 bytes, which a slot need not be.
//   - Cost, counted for Gemma 3 1B (not measured):
//       - The step that ends the prompt: 26 dispatches of one workgroup,
//         each copying 27 KiB; 702 KiB written, once a run.
//       - Every decode step: the same 26 dispatches, each reading one word
//         of the step and stopping. At Charlotte's 1.5 µs a dispatch (one
//         machine, the browser), about 40 µs a token, as the edit launch
//         costs. Measuring decode speed is deferred (QUEUE, open
//         decisions); this is an estimate.
//       - Memory: 702 KiB on the device for the store, and the reader's
//         (capture/reader.h).
//   - Alternatives considered:
//       - Fold the copy into the edit kernel, which runs at the same point:
//         no dispatch added. But one kernel would then change with two
//         axes, edits (Q) and captures (O), and the edit kernel's contract
//         and test would change with every new capture.
//       - A launch the schedule skips in fed steps: no dispatch in decode.
//         But a new kind of geometry in the ported schedule and program
//         (axis D), for 40 µs a token that speed, not this demo's aim, has
//         not asked for.
//     This design is the simplest of the three and leaves edits and the
//     schedule as they are.
//
// Corpus guidance
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work: the launch is tiny and runs in every
//            scoring step; its cost is counted above, and the fused form
//            named, to be chosen by measurement if speed comes to matter.
//     GPU.1  Keep data on the device: the copy stays on the GPU inside the
//            pass; the store crosses to the CPU once a run, at its end
//            (capture/reader.h).
//     GPU.9  Suballocate: the store is one working buffer in the plan, not
//            one a layer.
//   C++ Core Guidelines
//     I.5    State preconditions: the slot is layout()'s, and the widths
//            match.
//     I.7    State postconditions: the slot holds the last row's bits after
//            the step that ends the prompt, and nothing else is written.

#include <cstdint>
#include <vector>

#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace frizzle::capture {

// A layer's slot in the store, in 32-bit words.
struct Slot {
  std::uint32_t first;
  std::uint32_t width;
};

struct Layout {
  std::vector<Slot> slots;  // one a layer, in layer order
  std::uint64_t words = 0;  // the store's size
};

[[nodiscard]] Layout layout(const model::ModelDescription& model);

[[nodiscard]] kernels::Launch capture_launch(
    const residency::BufferRange& activation,
    const residency::BufferRange& store, Slot slot);

}  // namespace frizzle::capture
