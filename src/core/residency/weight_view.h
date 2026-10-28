#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/gguf/types.h"

namespace frizzle::residency {

// Contract 6: the weight view.
//
// What a kernel is given for a weight. Where the bytes are and how to read
// them travel as one value: a kernel cannot receive a buffer without its
// format and shape, which is the defect that produces well-shaped nonsense.
//
// A tensor larger than one storage binding is split by rows, and its view
// lists one piece per binding. A view names buffers by their index in the
// residency plan, so it holds no GPU object and is built and tested without a
// device.

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
