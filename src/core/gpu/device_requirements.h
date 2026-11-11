#pragma once

#include <cstdint>

namespace frizzle::gpu {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// What Frizzle requires of a WebGPU device: five limits, each the WebGPU
// default, requested as they stand. The request is a statement of need, not a
// report of what any GPU offers.
//
// Contract
//   - kRequirements is constant data, read by gpu/device.cpp only. The device
//     request sets these five limits to these values, and leaves every other
//     limit at its default (WGPU_LIMITS_INIT).
//   - After the device is granted, the five granted limits are checked against
//     these; a grant below any one fails the acquisition, though a conforming
//     implementation cannot grant less than was required. The failure says
//     the grant fell short; it does not name the limit.
//   - Nothing plans against these numbers. The residency plan, the program and
//     the kernels plan against the limits granted, read back after acquisition
//     (residency/plan.h, kernels/program.cpp), so a device that grants more is
//     used within what it granted.
//   - Adapter limits are reported as diagnostics and never planned against.
//   - The device is requested with no optional feature, neither `subgroups`
//     nor `shader-f16`. A diagnostics build that profiles adds timestamp
//     queries, and fails if the adapter lacks them.
//
// Each requirement, the WebGPU default, and what in Frizzle meets it:
//   max_buffer_size, 256 MiB: the plan caps every buffer it creates at the
//     granted size, cutting a weight into pieces of whole rows across buffers
//     (residency/plan.h). Gemma 3 1B's 722 MB file needs several buffers.
//   max_storage_buffer_binding_size, 128 MiB: the plan caps every bound range
//     at min(granted binding size, granted buffer size); a row wider than one
//     range is refused, naming the tensor.
//   max_storage_buffers_per_shader_stage, 8: the most any shader module
//     declares is 7, rope (rope.wgsl joined with rope_cache.wgsl) and
//     attention's main; norm 6, matmul 5, gather 3, topk 2, the self-check's
//     vector_add 3.
//   max_compute_invocations_per_workgroup, 256: norm's launches use 256, the
//     largest; attention, gather, matmul, rope and the self-check use 64.
//   max_compute_workgroups_per_dimension, 65,535: the program refuses a launch
//     wider than the granted maximum (kernels/program.cpp), and matmul's split
//     stays within it (kernels/matmul/matmul.h).
//
// Design and cost
//   - Requiring only the defaults is a product decision: Frizzle runs on any
//     device that implements WebGPU, at the cost of no limit the target,
//     Chrome on Apple silicon, could grant above them. Raising one excludes
//     conformant devices, so it waits on a measured benefit.
//   - Cost: none per token. The five values are compile-time constants, set
//     once in the device request and compared once against the grant.
//
// Verification
//   Every native run requests the device with these limits and passes the
//   granted check on the development machine; no test grants less.
//
// Corpus guidance
//   C++ performance guidelines
//     WASM.10 Request the device capability you need, not the one the adapter
//             offers: the five limits requested exactly, the grant read back
//             and planned from, adapter limits as diagnostics.
//   C++ Core Guidelines
//     Con.5   Use constexpr for values that can be computed at compile time:
//             kRequirements is constexpr data.

struct DeviceRequirements {
  std::uint64_t max_buffer_size;
  std::uint64_t max_storage_buffer_binding_size;
  std::uint32_t max_storage_buffers_per_shader_stage;
  std::uint32_t max_compute_invocations_per_workgroup;
  std::uint32_t max_compute_workgroups_per_dimension;
};

inline constexpr DeviceRequirements kRequirements{
    .max_buffer_size = 256ull * 1024 * 1024,
    .max_storage_buffer_binding_size = 128ull * 1024 * 1024,
    .max_storage_buffers_per_shader_stage = 8,
    .max_compute_invocations_per_workgroup = 256,
    .max_compute_workgroups_per_dimension = 65535,
};

}  // namespace frizzle::gpu
