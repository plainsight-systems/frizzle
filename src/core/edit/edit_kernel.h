#pragma once

// Axis Q: changes with a kind of edit, or what one does.
//
// The edit kernel's launch: edit/edit.h's apply, on the GPU, inside the
// pass. One launch a layer, placed by the graph between the feed-forward
// block's gate·up product, which writes the activation, and its down
// projection, which reads it (graph/graph.cpp, gated_feed_forward).
//
// Contract
//   - edit_launch(layer, activation, table, width): the layer's launch,
//     binding the table (binding 2) and the activation (binding 3). It runs
//     in every step, prefill blocks and decode steps alike, and changes the
//     rows whose positions an edit's range holds: row r of a step is
//     position step.position + r, and an edit [first, end) of the table
//     (edit/edit.h, positions) changes that row's neuron when first <=
//     position < end. Every other row, and every other value, keeps its
//     bits. So a position's activation, and with it the cache entries the
//     later layers write there, is the same whether a prefill block or a
//     decode step computes it: a reply recomputed from its tokens and its
//     edits is the reply as generated (runtime/engine.h).
//   - For each of the layer's edits holding a row, the neuron's value
//     becomes apply(kind, value, neuron), in f32, as on the CPU: Set and
//     Zero store, Scale multiplies once, correctly rounded.
//     Scale by exactly 1 keeps the neuron's bits: WGSL lets an
//     implementation flush a subnormal product to zero, so the kernel does
//     not multiply by 1.
//   - A layer with no edits changes nothing.
//   - Preconditions: the table holds a valid set (edit::check) for the
//     model the graph was built for; the activation's rows are `width`
//     floats.
//
// Algorithm, the kernel (edit.wgsl), one workgroup of 64 a row
//   1. Workgroup r takes row r of the step, at position step.position + r,
//      and reads the layer's [first, count] from the table. A count of 0
//      ends.
//   2. Each invocation j takes the layer's edits j, j + 64, ...: reads
//      [neuron, kind, value, first, end] at word first + 5 × edit, and when
//      the row's position lies in [first, end), rewrites
//      activation[r × width + neuron]. Two edits of one neuron never hold
//      at one position (edit::check), so no two invocations write one
//      value. (A workgroup a row, because an EveryToken launch of 64
//      invocations a row is dispatched as one workgroup a row,
//      kernels/interface.h; one workgroup doing every row would be run
//      once a row, and a Scale applied that many times.)
//
// Design and cost
//   - The launch is Rows::EveryToken, so the schedule runs it in every
//     step (kernels/schedule.h), and the kernel tests each row's position:
//     a branch recomputes kept reply positions whose edits an accepted
//     reply held, inside prefill blocks, which a last-row launch never
//     reaches. Until this change it was Rows::LastToken, and edits reached
//     only the last prompt position and the decode steps.
//   - Each row's workgroup visits the layer's edits: a prefill block of
//     kPrefillBlock rows, at most 4,096 edits a layer; a decode step, one
//     row. With no edits, two words read a row; with a few pins, a few
//     hundred comparisons a block, nothing beside the block's matmuls.
//   - The table lives in one working buffer the plan sizes at load, a
//     scratch need of kMaxEdits edits (graph decoder_working_set); the
//     engine writes it once before each run.
//   - Cost when nothing is edited: 26 more dispatches in each step,
//     one workgroup each, that read two words and stop. At about 1.5 µs a
//     dispatch (an estimate, not measured in Frizzle), about 40 µs a token.
//     Not measured against Gemma 3's decode step.
//   - That cost is not zero. Accepted by the user
//     (docs/architecture/logical-overview.md, Edits): speed is not this
//     demo's aim. The alternatives
//     considered:
//       - Launches only in edited layers: a new graph and program for each
//         edit set, and with it a new turn loop and cache, so the whole
//         prompt is prefilled again on every change.
//       - The edit folded into the gate·up product's epilogue: a change to
//         the hottest ported kernel, paid on every row of every step.
//     This design is the simplest of the three.
//   - Changes to ported files: graph.cpp gains the launch and the scratch
//     need, and graph.h is restated to the header standard (it then
//     changes).
//
// Corpus guidance
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work: the launch is tiny and repeated; counted
//            above, and the fused alternative named, to be chosen by
//            measurement.
//     GPU.1  Keep data on the device: the table is written once a run, and
//            nothing is read back.
//   C++ Core Guidelines
//     I.7    State postconditions: only rows in an edit's range change, and
//            Scale by 1 changes nothing.

#include <cstdint>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"

namespace frizzle::edit {

[[nodiscard]] kernels::Launch edit_launch(
    std::uint32_t layer, const residency::BufferRange& activation,
    const residency::BufferRange& table, std::uint32_t width);

}  // namespace frizzle::edit
