#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/gguf/types.h"

namespace frizzle::residency {

// Axis D: changes with the limits a device grants, or how the model is placed
// within them.
//
// The weight view: what a kernel is given for a weight. Where the bytes are
// and how to read them travel as one value: a kernel cannot receive a buffer
// without its format and shape, which is the defect that produces
// well-shaped nonsense.
//
// Contract
//   - WeightView(format, shape, pieces): a GGUF tensor's format and shape,
//     and the pieces it is placed in, in row order. It holds them by value
//     and owns its pieces; the accessors return them unchanged and are
//     noexcept.
//   - No default constructor: a view without a format is the defect this
//     type prevents, and the static_assert below keeps it so.
//   - WeightPiece: a run of whole rows, row_count of them from first_row,
//     placed contiguously in one buffer from `offset`. `length` is the bound
//     length: the rows' bytes rounded up to 4, as a storage binding
//     requires.
//   - A tensor larger than one storage binding is split by rows, one piece a
//     binding. The pieces are the residency plan's (residency/plan.cpp); the
//     view checks nothing about them.
//   - A view names buffers by BufferIndex, their position in the residency
//     plan, so it holds no GPU object and is built without a device, as the
//     kernels suite's support builds its f32 views
//     (tests/support/scratch_program.h).
//
// Algorithm: none; a value the plan builds and each kernel's launcher reads.
//
// Design and cost
//   - Gemma 3 1B's pinned file, counted: 340 tensors, a view each, built
//     once a load. The token embedding, 320,864,256 bytes, is 3 pieces in
//     matmul.h's count; every other tensor is one.
//   - The constructor takes its pieces by value and moves them in, so the
//     plan's vector is moved, never copied (COPY.3).
//   - Copyable without a default constructor, a departure from C.43 on
//     purpose: no default view is meaningful.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed: a weight's format
//            and shape travel with its buffers, and BufferIndex is a type of
//            its own, not a bare u32 a count or offset could be passed for.
//     C.2    Use class if the class has an invariant: a view always has a
//            format and a shape; WeightPiece's fields vary independently and
//            it is a struct.
//     C.43   Ensure that a copyable class has a default constructor: not
//            followed, on purpose (Design and cost).
//   C++ performance guidelines
//     COPY.3 Pick the right sink-parameter shape: the pieces, by value and
//            moved.

// A buffer's position in the residency plan.
enum class BufferIndex : std::uint32_t {};

// A run of whole rows placed contiguously in one buffer. `length` is the bound
// length: the rows' bytes rounded up to 4, as a storage binding requires.
struct WeightPiece {
  BufferIndex buffer;
  std::uint64_t offset;
  std::uint64_t length;
  std::uint64_t first_row;
  std::uint64_t row_count;
};

class WeightView {
 public:
  // No default: a view without a format is the defect this type prevents.
  WeightView() = delete;

  WeightView(gguf::TensorType format, const gguf::TensorShape& shape,
             std::vector<WeightPiece> pieces)
      : format_(format), shape_(shape), pieces_(std::move(pieces)) {}

  [[nodiscard]] gguf::TensorType format() const noexcept { return format_; }
  [[nodiscard]] const gguf::TensorShape& shape() const noexcept {
    return shape_;
  }
  [[nodiscard]] const std::vector<WeightPiece>& pieces() const noexcept {
    return pieces_;
  }

 private:
  gguf::TensorType format_;
  gguf::TensorShape shape_;
  std::vector<WeightPiece> pieces_;
};

static_assert(
    !std::is_default_constructible_v<WeightView>,
    "a weight view without a format is the defect this type prevents");

}  // namespace frizzle::residency
