#pragma once

#include <cstdint>

namespace frizzle::gpu {

enum class DispatchStatus {
  Ok,
  InvalidWorkgroupSize,
  ExceedsDeviceLimit,
};

struct DispatchResult {
  DispatchStatus status;
  std::uint32_t workgroup_count;
};

// Number of workgroups needed to cover `element_count` elements when each
// workgroup runs `workgroup_size` invocations.
//
// Fails explicitly rather than clamping. A workload larger than
// `max_workgroups_per_dimension` must be tiled by the caller; silently
// truncating the count would leave part of the data unprocessed and the
// kernel would return a wrong answer rather than an error.
//
// An element count of zero yields zero workgroups, which is a valid dispatch.
DispatchResult dispatch_count(
    std::uint64_t element_count, std::uint32_t workgroup_size,
    std::uint32_t max_workgroups_per_dimension) noexcept;

}  // namespace frizzle::gpu
