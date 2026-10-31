#pragma once

// Axis Q: changes with a kind of edit, or what one does.
//
// Pinning a neuron: what an edit is, which edit sets are valid, and what an
// edit does to a neuron's value, as a plain function. The edit kernel
// (edit/edit_kernel.h) does the same thing on the GPU; this file is its
// definition, tested on the CPU.
//
// Contract
//   - A neuron is one value of a layer's feed-forward activation: the gated
//     product gate·up, after the activation function, before the down
//     projection. Gemma 3 1B has 26 layers of 6,912.
//   - An edit names a layer, a neuron, a kind, a value, and the steps it
//     holds over:
//       Set    the neuron becomes `value`
//       Scale  the neuron is multiplied by `value`
//       Zero   the neuron becomes 0; `value` must be 0
//       [from, until)  the steps of the run it applies at; until may be
//              kToTheEnd, every later step. A run's step s is the position
//              that draws its reply token s: step 0 is the rendered
//              text's last token, step s > 0 the position whose input is
//              reply token s − 1, whether that token was generated in this
//              run or kept from an earlier one (runtime/engine.h). So an
//              edit from step k changes what reply token k and every later
//              one are drawn from, and nothing before.
//   - apply(kind, value, neuron) is what an edit does, in f32. Scale by 1 is
//     an exact no-op, bit for bit, as IEEE multiplication by 1 is.
//   - check(edits, model): an edit set is valid when there are at most
//     kMaxEdits, every kind is one of the three, every layer exists, every
//     neuron is below its layer's width, every value is finite (and 0 for
//     Zero), every range holds a step (from < until), and no two edits of
//     one neuron hold at a common step. One neuron may be edited at several
//     ranges that do not overlap: an accepted reply's pins up to a step,
//     then the pins held now from it (runtime/engine.h). Otherwise the set
//     is refused, named (EditError), with an edit's index: the first edit,
//     in order, that is invalid on its own; else the first whose range
//     overlaps an earlier edit's on the same neuron. Order does not
//     otherwise matter: a set, not a sequence.
//   - table(edits, model, origin): a valid set as the kernel reads it
//     (below), each range placed at positions: step s is position
//     origin + s, saturating at kToTheEnd, which no position the context
//     holds reaches (kernels::kMaxPositions is 2^24), so a step past every
//     position reads as "to the end". origin is the run's rendered text's
//     tokens less one. Precondition: check passed.
//   - placed(edits, origin): the same ranges, as positions, for comparing
//     runs whose origins differ (cache/prefix.h, first_edit_difference).
//     Precondition: check passed.
//
// The table, u32 words
//   - Per layer, in layer order: [first, count]: the word where the layer's
//     edits begin, and how many there are.
//   - Then each edit, by layer, neuron and first position: [neuron, kind,
//     value's bits, first position, end position], the end exclusive,
//     kToTheEnd for none. A layer's edit i is at words first + 5i ..
//     first + 5i + 4. A layer with no edits has [0, 0].
//     The kind is its enumerator's value: Set 0, Scale 1, Zero 2; edit.wgsl
//     reads the same numbers.
//   - Size for Gemma 3 1B: 52 words, plus 5 an edit; at most 20,532 words
//     (80 KiB) at kMaxEdits.
//
// Algorithm
//   check: refuse more than kMaxEdits; then each edit in order on its own;
//   then the edits in index order into an ordered map keyed (layer, neuron,
//   from): the ranges already in it never overlap one another, so an edit
//   overlaps an earlier one exactly when it overlaps the range just before
//   or just after its own key; the first edit that does is the offender.
//   (Scanning sorted neighbours alone names the wrong edit when a long
//   range given late covers two short ones.)
//   table: zero the [first, count] pairs; sort the edits by (layer,
//   neuron, from); write each after the pairs in that order, its range
//   placed at origin, setting its layer's first at the layer's first edit
//   and counting it.
//
// Design and cost
//   - Plain values and pure functions: no GPU, no model file. The table is
//     made once per run, on the CPU, sorted for its one consumer (CDSA.32).
//   - Each function sorts or maps one copy of at most kMaxEdits small
//     records, once (CDSA.10), and allocates that copy and its result: at
//     most a few hundred KiB a run, nothing a token.
//   - kMaxEdits = 4,096 ranged edits: the table's buffer is sized once, at
//     load. Nothing in the interface asks for more; the cap is named in the
//     refusal. A pin held unchanged across an accepted reply is one range,
//     not two: the page joins them (app/src/session/branch.ts), so history
//     alone does not reach the cap.
//   - Ranges in steps, not positions: the page knows where the reply
//     begins, not how many tokens its rendered text makes; the engine does,
//     and places them (origin).
//   - Edit kinds are a closed set (Enum.3). A new kind changes this file and
//     edit.wgsl together, which is axis Q.
//
// Corpus guidance
//   C++ Core Guidelines
//     E.27   If you can't throw, use error codes systematically: EditError.
//     I.7    State postconditions: apply's Scale-by-1 exactness is stated,
//            and is what the replay test relies on.
//     Enum.3 Prefer class enums: EditKind, EditError.
//     SL.io.2 Always consider ill-formed input: the set arrives from a host,
//            so a kind outside the enum is refused like any other fault.
//   C++ performance guidelines
//     CDSA.32 Transform data once into the layout its consumer reads: the
//            table, grouped by layer, so a layer's launch reads its edits as
//            one run.
//     CDSA.10 Choose the ordered structure by mutation pattern: built once,
//            then read, so a sorted vector.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/model/model_description.h"

namespace frizzle::edit {

inline constexpr std::size_t kMaxEdits = 4096;
inline constexpr std::uint32_t kToTheEnd = 0xFFFFFFFFu;
// The table's words for an edit (above), which the graph sizes its buffer
// by.
inline constexpr std::size_t kEditWords = 5;

enum class EditKind : std::uint8_t { Set = 0, Scale = 1, Zero = 2 };

// No member has a default: an edit initialized without its range has
// from = until = 0, an empty range, which check refuses as Range, so an
// edit written before ranges existed is refused rather than silently
// applied from step 0.
struct Edit {
  std::uint32_t layer;
  std::uint32_t neuron;
  EditKind kind;
  float value;
  std::uint32_t from;   // the first step it applies at
  std::uint32_t until;  // the step after the last, or kToTheEnd
};

enum class EditError : std::uint8_t {
  Ok,
  Kind,       // not one of the kinds
  Layer,      // no such layer
  Neuron,     // at or past the layer's width
  NotFinite,  // a NaN or infinite value
  ZeroValue,  // a Zero edit with a value other than 0
  Range,      // from not below until: no step
  Overlap,    // a neuron edited twice at one step
  TooMany,    // more than kMaxEdits
};

struct EditCheck {
  EditError error = EditError::Ok;
  std::size_t index = 0;  // the first offending edit
  std::string subject;    // what was refused, and why
  [[nodiscard]] bool ok() const noexcept { return error == EditError::Ok; }
};

[[nodiscard]] float apply(EditKind kind, float value, float neuron) noexcept;

[[nodiscard]] EditCheck check(std::span<const Edit> edits,
                              const model::ModelDescription& model);

[[nodiscard]] std::vector<std::uint32_t> table(
    std::span<const Edit> edits, const model::ModelDescription& model,
    std::uint32_t origin);

[[nodiscard]] std::vector<Edit> placed(std::span<const Edit> edits,
                                       std::uint32_t origin);

}  // namespace frizzle::edit
