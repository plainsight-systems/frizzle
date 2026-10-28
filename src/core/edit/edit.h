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
//   - An edit names a layer, a neuron, a kind and a value:
//       Set    the neuron becomes `value`
//       Scale  the neuron is multiplied by `value`
//       Zero   the neuron becomes 0; `value` must be 0
//   - apply(kind, value, neuron) is what an edit does, in f32. Scale by 1 is
//     an exact no-op, bit for bit, as IEEE multiplication by 1 is.
//   - check(edits, model): an edit set is valid when there are at most
//     kMaxEdits, every kind is one of the three, every layer exists, every
//     neuron is below its layer's width, every value is finite (and 0 for
//     Zero), and no neuron is edited twice. Otherwise it is refused, named
//     (EditError), with an edit's index: the first edit, in order, that is
//     invalid on its own; else the first that names a neuron an earlier edit
//     names. Order does not otherwise matter: a set, not a sequence.
//   - table(edits, model): a valid set as the kernel reads it (below).
//     Precondition: check passed.
//
// The table, u32 words
//   - Per layer, in layer order: [first, count]: the word where the layer's
//     edits begin, and how many there are.
//   - Then each edit, by layer and then neuron: [neuron, kind, value's bits].
//     A layer's edit i is at words first + 3i .. first + 3i + 2. A layer
//     with no edits has [0, 0].
//     The kind is its enumerator's value: Set 0, Scale 1, Zero 2; edit.wgsl
//     reads the same numbers.
//   - Size for Gemma 3 1B: 52 words, plus 3 an edit; at most 12,340 words
//     (48 KiB) at kMaxEdits.
//
// Algorithm
//   check: refuse more than kMaxEdits; then each edit in order on its own;
//   then sort (layer, neuron, index) triples and scan neighbours for a
//   repeat, keeping the least index that repeats.
//   table: zero the [first, count] pairs; sort the edits by (layer,
//   neuron); write each after the pairs in that order, setting its layer's
//   first at the layer's first edit and counting it.
//
// Design and cost
//   - Plain values and pure functions: no GPU, no model file. The table is
//     made once per run, on the CPU, sorted for its one consumer (CDSA.32).
//   - Each function sorts one copy of at most kMaxEdits small records, once
//     (CDSA.10), and allocates that copy and its result: at most 48 KiB a
//     run, nothing a token.
//   - kMaxEdits = 4,096: the table's buffer is sized once, at load. Nothing
//     in the interface asks for more; the cap is named in the refusal.
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

enum class EditKind : std::uint8_t { Set = 0, Scale = 1, Zero = 2 };

struct Edit {
  std::uint32_t layer;
  std::uint32_t neuron;
  EditKind kind;
  float value;
};

enum class EditError : std::uint8_t {
  Ok,
  Kind,       // not one of the kinds
  Layer,      // no such layer
  Neuron,     // at or past the layer's width
  NotFinite,  // a NaN or infinite value
  ZeroValue,  // a Zero edit with a value other than 0
  Duplicate,  // a neuron edited twice
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
    std::span<const Edit> edits, const model::ModelDescription& model);

}  // namespace frizzle::edit
