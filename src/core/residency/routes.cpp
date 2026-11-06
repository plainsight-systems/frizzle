// Axis D: changes with how the file's bytes reach the buffers the plan lays
// out.
//
// Contract, algorithm, cost and corpus guidance: routes.h. This file
// defines what that header states.

#include "core/residency/routes.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "core/formats/format.h"
#include "core/gguf/checked.h"

namespace frizzle::residency {
namespace {

using gguf::checked_mul;
using gguf::range_within;

RouteResult failure(RouteError error, std::string subject) {
  return {error, std::move(subject)};
}

std::uint64_t padded(std::uint64_t bytes) { return (bytes + 3) / 4 * 4; }

// Routes one tensor's pieces, each a run of whole rows, so whole blocks.
RouteResult route_tensor(const gguf::TensorEntry& entry,
                         const PlannedTensor& planned_tensor,
                         const ResidencyPlan& plan, std::uint64_t file_size,
                         FindFormat find_format, std::vector<Route>& routes) {
  // A format not listed is refused whatever the tensor's size, so the
  // refusal does not depend on whether it has rows.
  const formats::Format* format = find_format(entry.type);
  if (format == nullptr) {
    return failure(RouteError::UnsupportedFormat, entry.name);
  }
  if (!formats::steps_by_groups(entry) && !formats::is_scalar(entry)) {
    return failure(RouteError::RowNotSteppable, entry.name);
  }
  const formats::DeviceLayout& layout = format->layout();

  const auto& pieces = planned_tensor.view.pieces();
  std::uint64_t rows = 0;
  for (const WeightPiece& piece : pieces) rows += piece.row_count;
  // No pieces, or none with a row: nothing to write, and no row to divide
  // the tensor's bytes by.
  if (rows == 0) return {};
  const std::uint64_t row_bytes =
      entry.data_length / rows;  // the reader checked rows are whole blocks
  for (const WeightPiece& piece : pieces) {
    std::uint64_t offset_in_tensor = 0;
    std::uint64_t bytes = 0;
    if (!checked_mul(piece.first_row, row_bytes, offset_in_tensor) ||
        !checked_mul(piece.row_count, row_bytes, bytes) ||
        !range_within(offset_in_tensor, bytes, entry.data_length) ||
        !range_within(entry.data_offset, offset_in_tensor, file_size) ||
        !range_within(entry.data_offset + offset_in_tensor, bytes, file_size)) {
      return failure(RouteError::OutOfRange, entry.name);
    }
    const auto buffer = static_cast<std::size_t>(piece.buffer);
    if (buffer >= plan.buffers.size() ||
        !range_within(piece.offset, piece.length, plan.buffers[buffer].size) ||
        piece.length != padded(bytes) || bytes % layout.block_bytes != 0) {
      return failure(RouteError::OutOfRange, entry.name);
    }
    routes.push_back({planned_tensor.tensor,
                      entry.data_offset + offset_in_tensor,
                      bytes / layout.block_bytes, &layout, piece.buffer,
                      piece.offset, piece.length});
  }
  return {};
}

}  // namespace

RouteResult plan_routes(const gguf::TensorIndex& index,
                        const ResidencyPlan& plan, std::uint64_t file_size,
                        FindFormat find_format, std::vector<Route>& routes,
                        ResidencyPlan& out) {
  ResidencyPlan carried = plan;
  std::vector<Route> found;
  for (const PlannedTensor& t : plan.tensors) {
    if (auto r = route_tensor(index.tensor(t.tensor), t, plan, file_size,
                              find_format, found);
        !r.ok()) {
      return r;
    }
  }
  std::sort(found.begin(), found.end(), [](const Route& a, const Route& b) {
    return a.file_offset < b.file_offset;
  });

  routes = std::move(found);
  out = std::move(carried);
  return {};
}

}  // namespace frizzle::residency
