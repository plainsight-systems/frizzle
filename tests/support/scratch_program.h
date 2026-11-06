#pragma once

// Axis V: one kernel's launches alone on the GPU, over buffers a test makes.
// Test support only.
//
// Contract
//   - ScratchProgram(sizes, make): acquires a device, plans one scratch
//     buffer for each size in bytes (each a multiple of 4), uploads the
//     plan with no file bytes, asks `make` for the launches given each
//     buffer's range, and builds a program of them. Any failure fails the
//     test, by name.
//   - range(i): buffer i's whole range.
//   - f32_view(i, width): a weight view over buffer i, F32, one row of
//     `width`, one piece: for a kernel that binds a weight, such as a
//     norm's gain, which the test then writes as an input like any other.
//   - run(step, inputs): writes each input into its buffer with the step,
//     runs the step and waits; a program error fails the test.
//   - words(i): buffer i's words as the GPU holds them, read after every
//     step submitted before.
//
// Design
//   - Every buffer is a working buffer (COPY_SRC and COPY_DST, residency/
//     upload.h), so any can be written as an input and read back.
//   - The capture test keeps its own harness: it predates this one and
//     reads the store through the capture reader, which it also locks.
//
// Corpus guidance
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: every failed stage fails the test by name.

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/index.h"
#include "core/gguf/types.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/kernels/interface.h"
#include "core/kernels/program.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/residency/weight_view.h"
#include "support/gpu.h"

namespace frizzle::testing {

class ScratchProgram {
 public:
  using Make = std::function<std::vector<kernels::Launch>(
      std::span<const residency::BufferRange>)>;

  ScratchProgram(std::span<const std::uint64_t> sizes, const Make& make)
      : instance_(wgpuCreateInstance(nullptr)),
        device_(acquire(instance_.get())) {
    REQUIRE(sizes.size() <= kNames.size());
    residency::ResidencyPlan plan;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      REQUIRE(sizes[i] % 4 == 0);
      plan.buffers.push_back({residency::Pool::Scratch, sizes[i]});
      plan.scratch.push_back(
          {kNames.at(i),
           {residency::BufferIndex{static_cast<std::uint32_t>(i)}, 0,
            sizes[i]}});
      plan.scratch_bytes += sizes[i];
    }
    plan.total_bytes = plan.scratch_bytes;

    struct Ready {
      std::unique_ptr<residency::Upload> upload;
      std::string why;
      bool done = false;
    } ready;
    const gguf::TensorIndex index;
    residency::Upload::begin(
        *device_, index, plan, 0, capability::find_format, kChunk,
        [](std::unique_ptr<residency::Upload> upload,
           residency::UploadError error, const char* subject, void* userdata) {
          auto& r = *static_cast<Ready*>(userdata);
          r.upload = std::move(upload);
          r.why = std::string(residency::to_string(error)) + " " +
                  (subject != nullptr ? subject : "");
          r.done = true;
        },
        &ready);
    pump_until(instance_.get(), ready.done, "the upload");
    REQUIRE_MESSAGE(ready.upload != nullptr, ready.why);
    upload_ = std::move(ready.upload);
    struct Finished {
      residency::UploadError error{};
      bool done = false;
    } finished;
    upload_->finish(
        [](residency::UploadError error, void* userdata) {
          auto& f = *static_cast<Finished*>(userdata);
          f.error = error;
          f.done = true;
        },
        &finished);
    pump_until(instance_.get(), finished.done, "the finish");
    REQUIRE(finished.error == residency::UploadError::Ok);

    for (const residency::PlannedScratch& s : upload_->plan().scratch) {
      ranges_.push_back(s.range);
    }
    struct Built {
      std::unique_ptr<kernels::Program> program;
      std::string message;
      bool done = false;
    } built;
    kernels::Program::build(
        *upload_, make(ranges_), std::nullopt,
        [](std::unique_ptr<kernels::Program> program,
           kernels::ProgramError /*error*/, std::string_view message,
           void* userdata) {
          auto& b = *static_cast<Built*>(userdata);
          b.program = std::move(program);
          b.message = message;
          b.done = true;
        },
        &built);
    pump_until(instance_.get(), built.done, "the program");
    REQUIRE_MESSAGE(built.program != nullptr, built.message);
    program_ = std::move(built.program);
  }

  [[nodiscard]] const residency::BufferRange& range(std::size_t i) const {
    return ranges_.at(i);
  }

  // A view over a buffer of the plan, which a launch may bind as a weight;
  // `sizes` gives its bytes, so make() may call it before the program is
  // built.
  [[nodiscard]] static residency::WeightView f32_view(std::size_t i,
                                                      std::uint32_t width) {
    gguf::TensorShape shape;
    shape.dimension_count = 1;
    shape.dimensions[0] = width;
    shape.element_count = width;
    return residency::WeightView(
        gguf::TensorType::F32, shape,
        {{residency::BufferIndex{static_cast<std::uint32_t>(i)}, 0,
          std::uint64_t{width} * 4, 0, 1}});
  }

  struct Input {
    std::size_t buffer;
    std::span<const std::uint32_t> words;
  };

  void run(const kernels::Step& step, std::span<const Input> inputs) {
    std::vector<kernels::RunInput> writes;
    for (const Input& in : inputs) {
      const residency::BufferRange& r = ranges_.at(in.buffer);
      writes.push_back({{r.buffer, r.offset, in.words.size_bytes()},
                        std::as_bytes(in.words)});
    }
    struct Ran {
      kernels::ProgramError error{};
      std::string message;
      bool done = false;
    } ran;
    program_->run(
        step, writes, {},
        [](kernels::ProgramError error, std::string_view message,
           std::span<const std::byte> /*readback*/, void* userdata) {
          auto& r = *static_cast<Ran*>(userdata);
          r.error = error;
          r.message = message;
          r.done = true;
        },
        &ran);
    pump_until(instance_.get(), ran.done, "the step");
    REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
  }

  [[nodiscard]] std::vector<std::uint32_t> words(std::size_t i) {
    const residency::BufferRange& r = ranges_.at(i);
    const std::vector<float> floats =
        read_floats(instance_.get(), *device_, upload_->buffer(r.buffer),
                    r.offset, static_cast<std::size_t>(r.length / 4));
    std::vector<std::uint32_t> out;
    out.reserve(floats.size());
    for (const float f : floats) out.push_back(std::bit_cast<std::uint32_t>(f));
    return out;
  }

 private:
  static constexpr std::size_t kChunk = std::size_t{1} << 20U;
  // The plan names its working buffers; these outlive every plan.
  static constexpr std::array<std::string_view, 12> kNames = {
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"};

  gpu::Instance instance_;
  std::unique_ptr<gpu::Device> device_;
  std::unique_ptr<residency::Upload> upload_;
  std::unique_ptr<kernels::Program> program_;
  std::vector<residency::BufferRange> ranges_;
};

}  // namespace frizzle::testing
