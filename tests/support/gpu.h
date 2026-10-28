#pragma once

// Axis V: driving the GPU natively from a test. Test support only.
//
// Contract
//   - pump_until(instance, done, what): runs WebGPU's callbacks on this
//     thread (gpu/callback_mode.h) until `done`. A wait that has not ended
//     after two minutes fails the test, naming `what`, rather than hanging.
//   - acquire(instance): a device, requested as the engine requests one. No
//     device fails the test; it never skips.
//   - read_floats(instance, device, buffer, offset, count): `count` floats
//     of `buffer` from `offset`, copied after all work submitted before and
//     read through a mapping of the test's own. The buffer must allow
//     COPY_SRC, as the plan's working buffers do.
//
// Corpus guidance
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: a wait that never ends, a missing device and a failed
//            mapping each fail the test by name.

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"

namespace frizzle::testing {

inline void pump_until(WGPUInstance instance, const bool& done,
                       const char* what) {
  constexpr auto kTimeout = std::chrono::minutes{2};
  constexpr auto kInterval = std::chrono::microseconds{100};
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (!done) {
    wgpuInstanceProcessEvents(instance);
    if (done) break;
    if (std::chrono::steady_clock::now() > deadline) {
      FAIL("timed out waiting for " << what);
    }
    std::this_thread::sleep_for(kInterval);
  }
}

inline std::unique_ptr<gpu::Device> acquire(WGPUInstance instance) {
  struct Acquired {
    std::unique_ptr<gpu::Device> device;
    std::string error;
    bool done = false;
  } acquired;
  gpu::Device::request(
      instance,
      [](std::unique_ptr<gpu::Device> device, const char* error,
         void* userdata) {
        auto& a = *static_cast<Acquired*>(userdata);
        a.device = std::move(device);
        if (error != nullptr) a.error = error;
        a.done = true;
      },
      &acquired);
  pump_until(instance, acquired.done, "a device");
  REQUIRE_MESSAGE(acquired.device != nullptr, acquired.error);
  return std::move(acquired.device);
}

inline std::vector<float> read_floats(WGPUInstance instance,
                                      const gpu::Device& device,
                                      WGPUBuffer buffer, std::uint64_t offset,
                                      std::size_t count) {
  const std::uint64_t bytes = count * sizeof(float);
  WGPUBufferDescriptor desc{};
  desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  desc.size = bytes;
  const gpu::Buffer readback(wgpuDeviceCreateBuffer(device.handle(), &desc));
  REQUIRE(readback);
  {
    const gpu::CommandEncoder encoder(
        wgpuDeviceCreateCommandEncoder(device.handle(), nullptr));
    wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), buffer, offset,
                                         readback.get(), 0, bytes);
    const gpu::CommandBuffer commands(
        wgpuCommandEncoderFinish(encoder.get(), nullptr));
    WGPUCommandBuffer raw = commands.get();
    wgpuQueueSubmit(device.queue(), 1, &raw);
  }
  struct Mapped {
    WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    bool done = false;
  } mapped;
  const WGPUBufferMapCallbackInfo info{
      .nextInChain = nullptr,
      .mode = gpu::kCallbackMode,
      .callback =
          [](WGPUMapAsyncStatus status, WGPUStringView /*message*/,
             void* userdata, void* /*userdata2*/) {
            auto& m = *static_cast<Mapped*>(userdata);
            m.status = status;
            m.done = true;
          },
      .userdata1 = &mapped,
      .userdata2 = nullptr,
  };
  wgpuBufferMapAsync(readback.get(), WGPUMapMode_Read, 0,
                     static_cast<std::size_t>(bytes), info);
  pump_until(instance, mapped.done, "a buffer read back");
  REQUIRE(mapped.status == WGPUMapAsyncStatus_Success);
  std::vector<float> out(count);
  const void* range = wgpuBufferGetConstMappedRange(
      readback.get(), 0, static_cast<std::size_t>(bytes));
  REQUIRE(range != nullptr);
  std::memcpy(out.data(), range, static_cast<std::size_t>(bytes));
  wgpuBufferUnmap(readback.get());
  return out;
}

}  // namespace frizzle::testing
