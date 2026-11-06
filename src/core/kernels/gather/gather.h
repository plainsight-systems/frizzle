#pragma once

#include <vector>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace frizzle::kernels {

// Axis E: changes with a new or optimized kernel.
//
// The embedding: each token of the step becomes its row of Gemma 3 1B's
// token embedding table, decoded from Q8_0 and multiplied by the graph's
// scale, written as f32 into the hidden working buffer, where the first
// layer's attention norm reads it. One kernel for both regimes: a step's rows
// are independent, so prefill and decode differ only in how many there are.
//
// Contract
//   - gather_launches(table, hidden, sampled, scale): the launches, one for
//     each piece of `table` in piece order, that together write rows 0 ..
//     tokens − 1 of `hidden`, row i the table's row for token i of the step,
//     each weight multiplied by `scale`. Rows from `tokens` on are not
//     written. Pure: no GPU object; the program carries the launches out
//     (kernels/interface.h).
//   - Inputs. `table` is the token embedding as the plan placed it:
//     262,144 rows of 1,152 weights in Q8_0, 320,864,256 bytes, cut by rows
//     into three pieces of at most 109,655 rows, the last 42,834, at the
//     default 128 MiB binding (residency/plan.h). `hidden` is the plan's
//     hidden buffer, kPrefillBlock (512) rows of 1,152 f32. `sampled` is the
//     draw's record, whose first word is the token the last step drew
//     (sampler/sampler.h). `scale` is the graph's: the square root of the
//     embedding width, sqrt(1,152) ≈ 33.94, formed in f32
//     (arch/gemma3/gemma3.cpp).
//   - Preconditions: the table's format is one the capability table lists,
//     whose unpack the launch composes (routes refused any other); a row is
//     a whole number of 32-weight groups (1,152 is 36); every identifier in
//     the step is below the vocabulary, which the runtime checks before it
//     writes a step (kernels/interface.h): the kernel cannot tell an
//     identifier past the table from one in another piece. A drawn token is
//     below the vocabulary by construction.
//   - Postconditions: every row 0 .. tokens − 1 is written by exactly one
//     launch, the one whose piece holds its token, since the pieces
//     partition the table's rows. A `fed` step (kernels/interface.h) writes
//     one row, from the draw's token, and reads no identifier.
//   - Ownership: each launch names the plan's buffers by index, offset and
//     size (Binding) and the embedded WGSL by view; the plan and the program
//     own the buffers, which outlive the launches.
//   - Failure: none at run time. A broken precondition is not detected here:
//     a format the capability table does not list leaves the launch without
//     an unpack, its WGSL does not compile, and the program's build fails
//     (kernels/program.h).
//   - Numerics: a Q8_0 weight is q × d, q an 8-bit code and d the block's
//     f16 scale; their product is exact in f32 (at most 8 significant bits
//     times 11). Multiplying by `scale` rounds once to nearest. WGSL leaves
//     reassociation to the backend (formats/format.h), and one that computed
//     q × (d × scale) would round twice; either way a weight is within 2
//     units in the last place of the exact product. The f16 scale decodes as
//     the format's contract states (formats/format.h).
//
// Algorithm, for invocation x of a launch over the piece holding rows
// first_row .. first_row + row_count − 1:
//   1. row = x / 36, group = x mod 36: 36 invocations a row, one for each
//      32-weight group. If row >= tokens, return: the last workgroup's tail,
//      since 36 × tokens need not be a multiple of 64.
//   2. token = the first word of `sampled` in a fed step, else the step's
//      identifier for the row, ids[row / 4][row mod 4] (kernels/step.wgsl).
//   3. If token is outside the piece, return: another launch writes the row.
//   4. Decode group (token − first_row) × 36 + group of the piece with the
//      format's unpack. For Q8_0 that reads the block's 8 code words and the
//      word holding its f16 scale; the piece holds every block's codes, then
//      every block's scales, two to a word (formats/q8_0/q8_0.h), which is
//      why the launch passes the piece's block count, row_count × 36.
//   5. Multiply each of the 32 weights by `scale`, and write them, 8 vec4s,
//      at vec4 (row × 36 + group) × 8 of `hidden`.
//   Launch constants (binding 1), as WGSL lays them out, 20 bytes:
//       struct Gather { first_row: u32, row_count: u32, groups_per_row: u32,
//                       blocks_in_piece: u32, scale: f32 }
//   Bindings 2, 3 and 4: the piece of the table, read-only; `hidden`,
//   written; the draw's record, read-only. Workgroups of 64 invocations.
//
// Design and cost
//   - One invocation a group, with no workgroup memory and no barrier: a
//     group's 32 weights share one scale, so an invocation decodes them from
//     its own block and needs nothing from another.
//   - A table split by rows is one launch a piece, each binding only its
//     piece, because every format's unpack reads the one binding named
//     `weights` (formats/format.h). Every launch covers every row of the
//     step, and an invocation whose token lies in another piece returns once
//     it has read the identifier.
//   - Launches, counted: three a step, one a piece, at about 1.5 µs of GPU
//     time each, measured in Frizzle (kernels/interface.h): about 4.5 µs a
//     step. Workgroups a launch: ceil(36 × tokens / 64), 1 for a decode step
//     and 288 for a 512-row prefill step.
//   - Bytes, counted: a row's 36 groups read 1,152 code bytes and 72 bytes of
//     scales, its 1,224 bytes as stored, and write 1,152 × 4 = 4,608 bytes.
//     As loads, each invocation reads 9 words, 36 bytes, the scale word
//     shared by two groups. A 512-row prefill step reads 612 KiB of the table
//     and writes 2.25 MiB; a decode step reads 1,224 bytes and writes 4.5
//     KiB, so its cost is its three launches. No row of the table the step
//     does not name is read.
//   - Each invocation of each launch also reads its row's identifier, 4
//     bytes: 144 bytes a row a launch as loads, every group of the row
//     reading the same word. How many of those reach device memory is the
//     GPU's; WebGPU states nothing of it. Loading it once a workgroup would
//     need workgroup memory and a barrier that every invocation waits at,
//     to spare loads of one word.
//   - Access pattern (GPU.2): adjacent invocations take adjacent groups, so
//     a workgroup's writes, taken together, are one contiguous run of
//     hidden and its code reads one contiguous run of the piece. Within one
//     store instruction, though, adjacent lanes are 128 bytes apart, each
//     writing vec4 v of its own group, and within one load 32 bytes apart.
//     Not measured.
//   - Lever not taken: one launch binding every piece and choosing by
//     identifier would save two launches, about 3 µs a step, and would need
//     an unpack for each binding where every format's reads the one named
//     `weights`.
//   - Tests: none of this kernel alone. The ported kernel families get no
//     unit tests of their arithmetic by decision (tests/README.md); the
//     oracle, frizzle_oracle, checks the gather end to end, as part of
//     Gemma 3 1B's logits against the frozen reference
//     (tests/fixtures/reference/). So the piece boundaries, the fed path and
//     the 2-unit bound above are locked by no test of their own.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.20   Prefer return values to out parameters: the launches are
//            returned.
//     I.5    State preconditions: the declaration's, and the identifiers'
//            bound, which the runtime checks and the kernel cannot.
//   C++ performance guidelines
//     GDSA.18 Decode block-scaled codes in the load path: the table stays in
//            Q8_0; only the rows named are decoded, in registers, and written
//            once as f32.
//     GPU.2  Shape data for coalesced lane access: the pattern above,
//            contiguous across a workgroup, 128 bytes apart within a store.
//     GPU.6  Batch tiny GPU work: three launches a step, counted, and the
//            lever not taken.

// The launches that embed a step's tokens: one for each piece of `table`,
// in order, writing rows of `hidden`, each weight multiplied by `scale`. The
// unpack is the one the capability table lists for the view's own format,
// so a view cannot be read with another format's. Preconditions: `table` is
// the token embedding, in a format the capability table lists — routes
// refused any other — with rows a whole number of 32-weight groups; `hidden`
// is the plan's hidden buffer; `sampled` its draw's record.
[[nodiscard]] std::vector<Launch> gather_launches(
    const residency::WeightView& table, const residency::BufferRange& hidden,
    const residency::BufferRange& sampled, float scale);

}  // namespace frizzle::kernels
