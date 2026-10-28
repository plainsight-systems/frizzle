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
//     binding the table (binding 2) and the activation (binding 3). It
//     changes only the activation's last row, and only in steps that score
//     a token: the last prefill step of a prompt and every decode step. So
//     edits reach the last prompt position and every generated token, and
//     never an earlier position, whose cache entries stay as computed
//     without edits.
//   - For each of the layer's edits in the table (edit/edit.h), the
//     neuron's value becomes apply(kind, value, neuron), in f32, as on the
//     CPU: Set and Zero store, Scale multiplies once, correctly rounded.
//     Scale by exactly 1 keeps the neuron's bits: WGSL lets an
//     implementation flush a subnormal product to zero, so the kernel does
//     not multiply by 1.
//   - A layer with no edits changes nothing.
//   - Preconditions: the table holds a valid set (edit::check) for the
//     model the graph was built for; the activation's rows are `width`
//     floats.
//
// Algorithm, the kernel (edit.wgsl), one workgroup of 64
//   1. Read the layer's [first, count] from the table. A count of 0 ends.
//   2. Each invocation j takes the layer's edits j, j + 64, ...: reads
//      [neuron, kind, value] at word first + 3 × edit, and rewrites
//      activation[(tokens − 1) × width + neuron]. No two edits name one
//      neuron (edit::check), so no two invocations write one value.
//
// Design and cost
//   - The launch is the graph's own, Rows::LastToken: the program's
//     schedule already runs such launches only in steps that score, and
//     skips them in the others (kernels/schedule.h). The kernel needs no
//     test of the position.
//   - The table lives in one working buffer the plan sizes at load, a
//     scratch need of kMaxEdits edits (graph decoder_working_set); the
//     engine writes it once before each run.
//   - Cost when nothing is edited: 26 more dispatches in each scoring step,
//     one workgroup each, that read two words and stop. At Charlotte's
//     measured 1.5 µs a dispatch in the browser (one machine), about 40 µs a
//     token. Not yet measured against Gemma 3's decode step here; to measure
//     in slice 3's browser work, edits off and on.
//   - That cost is not zero. Accepted by the user (MEMORY, Engine
//     requirements): speed is not this demo's aim. The alternatives
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
//     I.7    State postconditions: only the last row changes, and Scale by
//            1 changes nothing.

#include <cstdint>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"

namespace frizzle::edit {

[[nodiscard]] kernels::Launch edit_launch(
    std::uint32_t layer, const residency::BufferRange& activation,
    const residency::BufferRange& table, std::uint32_t width);

}  // namespace frizzle::edit
