#include "core/gpu/dispatch_math.h"

namespace frizzle::gpu {

DispatchResult dispatch_count(
    std::uint64_t element_count, std::uint32_t workgroup_size,
    std::uint32_t max_workgroups_per_dimension) noexcept {
  if (workgroup_size == 0) {
    return {DispatchStatus::InvalidWorkgroupSize, 0};
  }
  if (element_count == 0) {
    return {DispatchStatus::Ok, 0};
  }

  // Divide before adding the remainder so the arithmetic cannot overflow for
  // element counts near the top of the u64 range.
  const std::uint64_t whole = element_count / workgroup_size;
  const std::uint64_t partial = (element_count % workgroup_size) != 0 ? 1u : 0u;
  const std::uint64_t needed = whole + partial;

  if (needed > max_workgroups_per_dimension) {
    return {DispatchStatus::ExceedsDeviceLimit, 0};
  }
  return {DispatchStatus::Ok, static_cast<std::uint32_t>(needed)};
}

}  // namespace frizzle::gpu
