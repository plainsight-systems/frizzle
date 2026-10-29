// Axis O: changes with a new captured value, or what is sent.
//
// Contract, algorithm, cost and corpus guidance: reader.h. This file defines
// what that header states.

#include "core/capture/reader.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "core/gpu/callback_mode.h"
#include "core/gpu/userdata.h"
#include "core/gpu/wgpu_handles.h"

namespace frizzle::capture {

struct Reader::State {
  WGPUDevice device = nullptr;  // borrowed, as the store is
  WGPUQueue queue = nullptr;
  std::shared_ptr<const gpu::DeviceStatus> status;
  WGPUBuffer store = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  gpu::Buffer mappable;
  std::vector<float> host;  // the last read's words, made at load
  bool cancelled = false;   // the reader was destroyed
};

namespace {

std::string to_string(WGPUStringView view) {
  if (view.data == nullptr) return {};
  return view.length == WGPU_STRLEN ? std::string(view.data)
                                    : std::string(view.data, view.length);
}

// Creation's two scopes, pushed out of memory first, then validation; each
// slot keeps its scope's message, empty when it caught nothing.
struct Creation {
  std::shared_ptr<Reader::State> state;
  std::array<std::string, 2> caught;
  std::size_t remaining = 2;
  MadeCallback done;
  void* userdata;
};

struct Pop {
  std::shared_ptr<Creation> creation;
  std::size_t slot;  // the order its scope was pushed in
};

struct Read {
  std::shared_ptr<Reader::State> state;
  ReadCallback done;
  void* userdata;
};

}  // namespace

void Reader::make(const gpu::Device& device, WGPUBuffer store,
                  const residency::BufferRange& range, MadeCallback done,
                  void* userdata) {
  auto state = std::make_shared<State>();
  state->device = device.handle();
  state->queue = device.queue();
  state->status = device.status();
  state->store = store;
  state->offset = range.offset;
  state->bytes = range.length;
  state->host.resize(range.length / sizeof(float));

  wgpuDevicePushErrorScope(state->device, WGPUErrorFilter_OutOfMemory);
  wgpuDevicePushErrorScope(state->device, WGPUErrorFilter_Validation);
  WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
  descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  descriptor.size = state->bytes;
  state->mappable.reset(wgpuDeviceCreateBuffer(state->device, &descriptor));

  auto creation =
      std::make_shared<Creation>(Creation{state, {}, 2, done, userdata});
  // Innermost first: validation was pushed second.
  for (std::size_t slot = 2; slot-- > 0;) {
    // Value-initialized, every field set below: the _INIT macro casts 0 to
    // the callback mode, which is no mode at all.
    WGPUPopErrorScopeCallbackInfo info{};
    info.mode = gpu::kCallbackMode;
    info.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type,
                       WGPUStringView message, void* userdata1, void*) {
      const auto pop = gpu::take_back<Pop>(userdata1);
      Creation& c = *pop->creation;
      if (status != WGPUPopErrorScopeStatus_Success) {
        c.caught.at(pop->slot) = "an error scope could not be popped";
      } else if (type != WGPUErrorType_NoError) {
        c.caught.at(pop->slot) = to_string(message);
        if (c.caught.at(pop->slot).empty()) {
          c.caught.at(pop->slot) = "WebGPU gave no message";
        }
      }
      if (--c.remaining != 0) return;
      for (const std::string& m : c.caught) {
        if (!m.empty()) {
          c.done(nullptr, m, c.userdata);
          return;
        }
      }
      if (c.state->mappable.get() == nullptr) {
        c.done(nullptr, "the capture readback buffer was not created",
               c.userdata);
        return;
      }
      c.done(std::make_unique<Reader>(std::move(c.state)), {}, c.userdata);
    };
    info.userdata1 = gpu::hand_off(std::make_unique<Pop>(Pop{creation, slot}));
    wgpuDevicePopErrorScope(state->device, info);
  }
}

Reader::Reader(std::shared_ptr<State> state) : state_(std::move(state)) {}

Reader::~Reader() { state_->cancelled = true; }

void Reader::read(ReadCallback done, void* userdata) {
  State& s = *state_;
  gpu::CommandEncoder encoder(
      wgpuDeviceCreateCommandEncoder(s.device, nullptr));
  wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), s.store, s.offset,
                                       s.mappable.get(), 0, s.bytes);
  gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
  WGPUCommandBuffer submitted = commands.get();
  wgpuQueueSubmit(s.queue, 1, &submitted);

  WGPUBufferMapCallbackInfo info{};  // every field set below, as above
  info.mode = gpu::kCallbackMode;
  info.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1,
                     void*) {
    const auto r = gpu::take_back<Read>(userdata1);
    State& st = *r->state;
    if (st.cancelled) {
      r->done(residency::Mapped::Cancelled, {}, r->userdata);
      return;
    }
    const void* range =
        status == WGPUMapAsyncStatus_Success
            ? wgpuBufferGetConstMappedRange(st.mappable.get(), 0, st.bytes)
            : nullptr;
    const residency::Mapped outcome =
        residency::mapping_result(status, range != nullptr, *st.status);
    if (outcome != residency::Mapped::Ok || range == nullptr) {
      if (range != nullptr) wgpuBufferUnmap(st.mappable.get());
      r->done(outcome, {}, r->userdata);
      return;
    }
    std::memcpy(st.host.data(), range, st.bytes);
    wgpuBufferUnmap(st.mappable.get());
    r->done(residency::Mapped::Ok, st.host, r->userdata);
  };
  info.userdata1 =
      gpu::hand_off(std::make_unique<Read>(Read{state_, done, userdata}));
  wgpuBufferMapAsync(s.mappable.get(), WGPUMapMode_Read, 0, s.bytes, info);
}

}  // namespace frizzle::capture
