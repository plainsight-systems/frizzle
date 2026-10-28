#include "core/gpu/self_check.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "core/gpu/callback_mode.h"
#include "core/gpu/dispatch_math.h"
#include "core/gpu/wgpu_handles.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::gpu {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;  // must match vector_add.wgsl

WGPUStringView view(std::string_view s) {
  return WGPUStringView{s.data(), s.size()};
}

// Everything the readback callback needs once the work has been submitted.
// Heap allocated: it outlives the call that created it.
struct ReadbackContext {
  std::unique_ptr<Device> device;  // kept alive until the readback resolves
  Buffer readback;
  std::vector<float> expected;
  SelfCheckCallback callback;
  void* userdata;

  void finish(SelfCheckResult result) {
    result.device = std::move(device);
    callback(std::move(result), userdata);
    delete this;  // releases `readback` via its destructor
  }
};

// Returns the device with the failure, so the caller's ownership is restored
// on every exit rather than only the successful one.
void fail(std::unique_ptr<Device> device, SelfCheckCallback callback,
          void* userdata, std::string message) {
  SelfCheckResult r;
  r.error = std::move(message);
  r.device = std::move(device);
  callback(std::move(r), userdata);
}

}  // namespace

void run_self_check(std::unique_ptr<Device> device, std::size_t elements,
                    SelfCheckCallback callback, void* userdata) {
  if (device == nullptr) {
    fail(nullptr, callback, userdata, "no device supplied");
    return;
  }
  const auto dispatch =
      dispatch_count(elements, kWorkgroupSize,
                     device->limits().max_compute_workgroups_per_dimension);
  if (dispatch.status != DispatchStatus::Ok) {
    fail(std::move(device), callback, userdata,
         dispatch.status == DispatchStatus::InvalidWorkgroupSize
             ? "invalid workgroup size"
             : "element count exceeds this device's dispatch limit");
    return;
  }

  std::vector<float> lhs(elements), rhs(elements), expected(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    lhs[i] = static_cast<float>(i);
    rhs[i] = static_cast<float>(i) * 0.5f;
    expected[i] = lhs[i] + rhs[i];
  }
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(elements) * sizeof(float);

  WGPUDevice dev = device->handle();

  // wgpuDeviceCreateBuffer is the one creation call the header declares
  // nullable, so allocation failure is a contract outcome and is checked.
  // Every early return below is safe because each handle owns itself.
  auto make_buffer = [&](WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = {};
    d.usage = usage;
    d.size = bytes;
    return Buffer(wgpuDeviceCreateBuffer(dev, &d));
  };
  Buffer lhs_buf =
      make_buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
  Buffer rhs_buf =
      make_buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
  Buffer out_buf =
      make_buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
  Buffer readback =
      make_buffer(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
  if (!lhs_buf || !rhs_buf || !out_buf || !readback) {
    fail(std::move(device), callback, userdata,
         "GPU buffer allocation failed; the device could not provide " +
             std::to_string(bytes * 4) + " bytes");
    return;
  }

  wgpuQueueWriteBuffer(device->queue(), lhs_buf.get(), 0, lhs.data(), bytes);
  wgpuQueueWriteBuffer(device->queue(), rhs_buf.get(), 0, rhs.data(), bytes);

  WGPUShaderSourceWGSL wgsl = {};
  wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl.code = view(shaders::vector_add);
  WGPUShaderModuleDescriptor module_desc = {};
  module_desc.nextInChain = &wgsl.chain;
  ShaderModule module(wgpuDeviceCreateShaderModule(dev, &module_desc));

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.compute.module = module.get();
  pipeline_desc.compute.entryPoint = view("main");
  ComputePipeline pipeline(
      wgpuDeviceCreateComputePipeline(dev, &pipeline_desc));

  // These are not declared nullable; failures arrive on the device's
  // uncaptured-error callback, which Device installs.
  BindGroupLayout layout(
      wgpuComputePipelineGetBindGroupLayout(pipeline.get(), 0));

  WGPUBindGroupEntry entries[3] = {};
  entries[0].binding = 0;
  entries[0].buffer = lhs_buf.get();
  entries[0].size = bytes;
  entries[1].binding = 1;
  entries[1].buffer = rhs_buf.get();
  entries[1].size = bytes;
  entries[2].binding = 2;
  entries[2].buffer = out_buf.get();
  entries[2].size = bytes;
  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = layout.get();
  bg_desc.entryCount = 3;
  bg_desc.entries = entries;
  BindGroup bind_group(wgpuDeviceCreateBindGroup(dev, &bg_desc));

  CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(dev, nullptr));
  {
    ComputePassEncoder pass(
        wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr));
    wgpuComputePassEncoderSetPipeline(pass.get(), pipeline.get());
    wgpuComputePassEncoderSetBindGroup(pass.get(), 0, bind_group.get(), 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass.get(),
                                             dispatch.workgroup_count, 1, 1);
    wgpuComputePassEncoderEnd(pass.get());
  }
  wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), out_buf.get(), 0,
                                       readback.get(), 0, bytes);
  CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
  WGPUCommandBuffer raw_commands = commands.get();
  wgpuQueueSubmit(device->queue(), 1, &raw_commands);

  // Ownership of the readback buffer moves into the callback context; every
  // other handle is released here by going out of scope. The submitted
  // commands hold their own references until they retire.
  auto* ctx = new ReadbackContext{std::move(device), std::move(readback),
                                  std::move(expected), callback, userdata};

  WGPUBufferMapCallbackInfo map_cb = {};
  map_cb.mode = kCallbackMode;
  map_cb.userdata1 = ctx;
  map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1,
                       void*) {
    auto* c = static_cast<ReadbackContext*>(ud1);
    SelfCheckResult r;
    r.elements = c->expected.size();

    if (status != WGPUMapAsyncStatus_Success) {
      r.error = "could not map the readback buffer";
      c->finish(std::move(r));
      return;
    }
    const auto byte_count = c->expected.size() * sizeof(float);
    const void* mapped =
        wgpuBufferGetConstMappedRange(c->readback.get(), 0, byte_count);
    if (mapped == nullptr) {
      r.error = "readback buffer mapped but produced no range";
      c->finish(std::move(r));
      return;
    }

    const auto* got = static_cast<const float*>(mapped);
    for (std::size_t i = 0; i < c->expected.size(); ++i) {
      if (got[i] != c->expected[i]) {
        ++r.mismatches;
      }
    }
    wgpuBufferUnmap(c->readback.get());

    r.ok = r.mismatches == 0;
    if (!r.ok) {
      r.error = "GPU result did not match the CPU computation";
    }
    c->finish(std::move(r));
  };
  wgpuBufferMapAsync(ctx->readback.get(), WGPUMapMode_Read, 0,
                     static_cast<std::size_t>(bytes), map_cb);
}

}  // namespace frizzle::gpu
