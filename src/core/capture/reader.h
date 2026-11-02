#pragma once

// Axis O: changes with a new captured value, or what is sent.
//
// Reads the capture store (capture/capture.h) back to the CPU, once a run,
// so the engine can hand the run's captures to its host with the run's
// result (runtime/engine.h).
//
// Contract
//   - Reader::make(device, store, range, done, userdata): makes, at load, a
//     mappable buffer of the store's size (MapRead | CopyDst) and a host
//     copy of it, inside the device's out-of-memory and validation error
//     scopes. `done` is called once: with the reader, or with no reader and
//     WebGPU's message when either scope reports an error. `store` is a
//     working buffer the upload made (COPY_SRC, residency/upload.h).
//   - read(done, userdata): copies the store into the mappable buffer in a
//     command buffer of its own, submitted on the device's queue, maps it,
//     and copies the mapping into the host copy; then calls `done` once:
//       - Ok, with the store's words as floats, bit for bit, valid only
//         during the call;
//       - a mapping's failure, by residency::Mapped (residency/mapping.h:
//         Cancelled, DeviceLost, Unexplained, Internal), with no words.
//     The queue orders the copy after every submission before it, so a read
//     started after a step was submitted reads what that step wrote. A
//     completed mapping is the one result a lost device cannot give
//     (gpu/device.h), so words are reported only through one.
//   - Preconditions: no read in flight (the engine runs one turn at a time,
//     and reads once a run); `device` and `store` outlive the reader.
//   - Destroying the reader with a read in flight reports it Cancelled; the
//     state its callback lands on is shared with it (F.27), as the
//     program's is.
//
// Algorithm, read
//   1. Encode copyBufferToBuffer(store → mappable, the store's bytes);
//      submit.
//   2. mapAsync(Read, whole buffer). On its callback, decide the outcome
//      with residency::mapping_result.
//   3. On Ok: memcpy the mapped range into the host copy; unmap; call
//      `done` with the host copy. Otherwise call `done` with the outcome.
//
// Design and cost
//   - Everything allocated at load (MEM.9): the mappable buffer, 3,356,480
//     bytes for Gemma 3 1B, and the host copy, as much again. A read
//     allocates nothing.
//   - One copy and one mapping a run, after the turn (GPU.1): the store is
//     final output, read once, never inside the step loop. Its cost is
//     a readback's fixed wait, about 0.5 ms at Charlotte's measure (one
//     machine, the browser), and 3.2 MiB moved twice (GPU to the mapping,
//     the mapping to the host copy); not measured here.
//   - Its own buffer, not the program's readback slots: those carry each
//     step's 16-byte draw record and are at most kMaxReadback, 4 KiB
//     (kernels/program.h); widening them would copy 3.2 MiB every step.
//   - The words cross as floats copied by memcpy, which keeps every bit;
//     the host decides what to show.
//
// Corpus guidance
//   C++ performance guidelines
//     GPU.1  Keep data on the device; every host-device round trip needs a
//            budget: one readback a run, at its end, budgeted above.
//     MEM.9  Allocate at init, not in steady state: both buffers at load.
//   C++ Core Guidelines
//     R.1    Manage resources with handles: the mappable buffer is held by a
//            handle the reader owns, released with it.
//     E.27   Use error codes systematically: every outcome is
//            residency::Mapped's, decided in one place.
//     I.7    State postconditions: words only through a completed mapping.
//     F.27   Use a shared_ptr to share ownership: the state, between the
//            reader and its read in flight.

#include <webgpu/webgpu.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "core/gpu/device.h"
#include "core/residency/mapping.h"
#include "core/residency/plan.h"

namespace frizzle::capture {

class Reader;

using MadeCallback = void (*)(std::unique_ptr<Reader> reader,
                              std::string_view message, void* userdata);
using ReadCallback = void (*)(residency::Mapped outcome,
                              std::span<const float> words, void* userdata);

class Reader {
 public:
  // Preconditions: `device` and `store` outlive the reader; `range` is the
  // plan's capture store in `store`, layout()'s total words long.
  static void make(const gpu::Device& device, WGPUBuffer store,
                   const residency::BufferRange& range, MadeCallback done,
                   void* userdata);

  // Preconditions: no read in flight.
  void read(ReadCallback done, void* userdata);

  ~Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  Reader(Reader&&) = delete;
  Reader& operator=(Reader&&) = delete;

  // What the reader and its read in flight share; defined only where the
  // reader is, so only make() can build one.
  struct State;
  explicit Reader(std::shared_ptr<State> state);

 private:
  std::shared_ptr<State> state_;
};

}  // namespace frizzle::capture
