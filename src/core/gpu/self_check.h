#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "core/gpu/device.h"

namespace frizzle::gpu {

struct SelfCheckResult {
  bool ok = false;
  std::string error;  // populated only when ok is false
  std::size_t elements = 0;
  std::size_t mismatches = 0;

  // The device the check ran against, handed back to the caller.
  //
  // Ownership flows through the call rather than being parked somewhere the
  // callback re-reads later: whatever the callback receives here is
  // definitionally the device that produced this result, on every path
  // including the failures.
  std::unique_ptr<Device> device;
};

// Runs the embedded vector_add shader over `elements` values and verifies the
// readback against the values computed on the CPU.
//
// This proves the toolchain reaches the GPU and returns correct data. It is
// not a model kernel: correctness here is self-evident, which is why it needs
// no reference oracle. Readback is asynchronous, so the result is delivered by
// continuation rather than returned.
//
// Takes ownership of the device for the duration and returns it in the result,
// so the device cannot be destroyed while a readback is still pending.
//
// Contract: `callback` must not be null. It is invoked exactly once, on every
// path including the failures, and always carries the device back.
using SelfCheckCallback = void (*)(SelfCheckResult result, void* userdata);

void run_self_check(std::unique_ptr<Device> device, std::size_t elements,
                    SelfCheckCallback callback, void* userdata);

}  // namespace frizzle::gpu
