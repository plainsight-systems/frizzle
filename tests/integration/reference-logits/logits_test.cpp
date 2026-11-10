// Axis V: the oracle, an integration test (tests/README.md). Run when the
// inference path changes, not with the routine suite.
//
// Contract locked: on Dawn, Frizzle's inference path computes Gemma 3 1B's
// logits for a fixed token sequence within llama.cpp's own tolerance, for
// the same file.
//
// Reference: tests/fixtures/reference/gemma-3-1b-it-q4_0.inc, frozen. For a
// 600-token passage, past Gemma 3's 512-key window, decoded a token at a
// time, it holds at each position llama.cpp's 20 most likely tokens on
// Metal, their logits on Metal and on the CPU, and each backend's
// log-sum-exp (tools/reference_logits/).
//
// Accepted, at every position:
//   - every logit finite;
//   - each of the 20 tokens' log-probability within the largest difference
//     llama.cpp's own CPU and Metal backends show at that position: our sums
//     are not llama.cpp's, in order or in kernels, so bits are not compared;
//     its own two backends' disagreement is the tolerance, frozen with the
//     reference;
//   - the most likely token the same as llama.cpp's wherever its lead over
//     the next exceeds that difference.
//
// Algorithm
//   1. Read the pinned model file's index from its front; describe it;
//      plan it within the device's limits and the default load policy, as
//      the engine does (runtime/engine.h); upload it in 16 MiB chunks.
//   2. Build the graph and its program.
//   3. For each position p: run one step of the reference's token p at p,
//      read the logits row back, and compare it with the reference.
//
// Cost: 600 steps and 600 readbacks of 1 MiB; one row held at a time.
//
// Corpus guidance
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level:
//            llama.cpp's sums and ours are reductions in different orders,
//            so the tolerance is llama.cpp's own cross-backend spread, not
//            bit equality.
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: a NaN anywhere fails the comparison.

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/byte_source.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/kernels/program.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/tokenizer/tokenizer.h"
#include "support/files.h"
#include "support/gpu.h"

namespace {

namespace frz = frizzle;

// One position of llama.cpp's logits, as tools/reference_logits writes it.
constexpr std::size_t kTop = 20;

struct ReferencePosition {
  double metal_lse;
  double cpu_lse;
  std::array<std::uint32_t, kTop> ids;  // Metal's most likely, most first
  std::array<float, kTop> metal;
  std::array<float, kTop> cpu;
};

#include "fixtures/reference/gemma-3-1b-it-q4_0.inc"

constexpr std::size_t kChunk = std::size_t{16} << 20U;

// The larger of the two, and NaN when either is: std::max drops a NaN.
double worse(double so_far, double d) {
  return d > so_far || std::isnan(d) ? d : so_far;
}

struct Agreement {
  double worst = 0;  // 1 or less: within llama.cpp's own spread everywhere
  double worst_nats = 0;
  int top_disagreements = 0;
  std::size_t non_finite = 0;
};

// One position's row against its reference, folded into `a`.
void compare(const ReferencePosition& ref, std::span<const float> row,
             Agreement& a) {
  a.non_finite += static_cast<std::size_t>(std::count_if(
      row.begin(), row.end(), [](float x) { return !std::isfinite(x); }));
  const double top = *std::max_element(row.begin(), row.end());
  double sum = 0;
  for (const float x : row) sum += std::exp(static_cast<double>(x) - top);
  const double lse = top + std::log(sum);
  double spread = 0;
  for (std::size_t k = 0; k < kTop; ++k) {
    spread = std::max(spread, std::abs((ref.metal.at(k) - ref.metal_lse) -
                                       (ref.cpu.at(k) - ref.cpu_lse)));
  }
  for (std::size_t k = 0; k < kTop; ++k) {
    const double d = std::abs((row[ref.ids.at(k)] - lse) -
                              (ref.metal.at(k) - ref.metal_lse));
    a.worst = worse(a.worst, d / spread);
    a.worst_nats = worse(a.worst_nats, d);
  }
  const auto argmax = static_cast<std::uint32_t>(
      std::max_element(row.begin(), row.end()) - row.begin());
  if (ref.metal[0] - ref.metal[1] > spread && argmax != ref.ids[0]) {
    ++a.top_disagreements;
  }
}

std::unique_ptr<frz::residency::Upload> upload_model(
    WGPUInstance instance, const frz::gpu::Device& device,
    const frz::testing::Front& front,
    const frz::residency::ResidencyPlan& plan) {
  using frz::residency::Upload;
  using frz::residency::UploadError;
  struct Ready {
    std::unique_ptr<Upload> upload;
    UploadError error = UploadError::Internal;
    std::string subject;
    bool done = false;
  } ready;
  Upload::begin(
      device, front.index, plan, front.file_size, frz::capability::find_format,
      kChunk,
      [](std::unique_ptr<Upload> upload, UploadError error, const char* subject,
         void* userdata) {
        auto& r = *static_cast<Ready*>(userdata);
        r.upload = std::move(upload);
        r.error = error;
        if (subject != nullptr) r.subject = subject;
        r.done = true;
      },
      &ready);
  frz::testing::pump_until(instance, ready.done, "the buffers");
  REQUIRE_MESSAGE(ready.error == UploadError::Ok,
                  frz::residency::to_string(ready.error)
                      << " " << ready.subject);

  struct Reported {
    UploadError error = UploadError::Internal;
    bool done = false;
  };
  const auto on_reported = [](UploadError error, void* userdata) {
    auto& r = *static_cast<Reported*>(userdata);
    r.error = error;
    r.done = true;
  };
  std::ifstream file(FRIZZLE_MODEL_FILE, std::ios::binary);
  REQUIRE(file.is_open());
  std::vector<char> raw(kChunk);
  std::vector<std::byte> chunk(kChunk);
  for (std::uint64_t at = 0; at < front.file_size;) {
    const std::uint64_t left = front.file_size - at;
    const std::size_t length =
        left < kChunk ? static_cast<std::size_t>(left) : kChunk;
    file.read(raw.data(), static_cast<std::streamsize>(length));
    REQUIRE(static_cast<std::size_t>(file.gcount()) == length);
    for (std::size_t i = 0; i < length; ++i) {
      chunk[i] = static_cast<std::byte>(static_cast<unsigned char>(raw[i]));
    }
    Reported accepted;
    ready.upload->write(at, std::span(chunk).first(length), on_reported,
                        &accepted);
    frz::testing::pump_until(instance, accepted.done, "a chunk's acceptance");
    REQUIRE(accepted.error == UploadError::Ok);
    at += length;
  }
  Reported finished;
  ready.upload->finish(on_reported, &finished);
  frz::testing::pump_until(instance, finished.done, "the upload's finish");
  REQUIRE_MESSAGE(finished.error == UploadError::Ok,
                  frz::residency::to_string(finished.error));
  return std::move(ready.upload);
}

std::unique_ptr<frz::kernels::Program> build_program(
    WGPUInstance instance, const frz::residency::Upload& upload,
    std::vector<frz::kernels::Launch> launches) {
  struct Built {
    std::unique_ptr<frz::kernels::Program> program;
    frz::kernels::ProgramError error = frz::kernels::ProgramError::Build;
    std::string message;
    bool done = false;
  } built;
  frz::kernels::Program::build(
      upload, std::move(launches), std::nullopt,
      [](std::unique_ptr<frz::kernels::Program> program,
         frz::kernels::ProgramError error, std::string_view message,
         void* userdata) {
        auto& b = *static_cast<Built*>(userdata);
        b.program = std::move(program);
        b.error = error;
        b.message = message;
        b.done = true;
      },
      &built);
  frz::testing::pump_until(instance, built.done, "the program");
  REQUIRE_MESSAGE(built.error == frz::kernels::ProgramError::Ok, built.message);
  return std::move(built.program);
}

void run_step(WGPUInstance instance, frz::kernels::Program& program,
              std::uint32_t position, frz::tokenizer::TokenId token) {
  frz::kernels::Step step{};
  step.position = position;
  step.tokens = 1;
  step.logits = 1;
  step.ids[0] = static_cast<std::uint32_t>(token);
  struct Ran {
    frz::kernels::ProgramError error = frz::kernels::ProgramError::Step;
    std::string message;
    bool done = false;
  } ran;
  program.run(
      step,
      [](frz::kernels::ProgramError error, std::string_view message,
         std::span<const std::byte> /*readback*/, void* userdata) {
        auto& r = *static_cast<Ran*>(userdata);
        r.error = error;
        r.message = message;
        r.done = true;
      },
      &ran);
  frz::testing::pump_until(instance, ran.done, "a step");
  REQUIRE_MESSAGE(ran.error == frz::kernels::ProgramError::Ok, ran.message);
}

}  // namespace

TEST_CASE(
    "Gemma 3 1B's logits are within llama.cpp's own spread, past its "
    "window") {
  const frz::gpu::Instance instance{wgpuCreateInstance(nullptr)};
  REQUIRE(instance);
  const std::unique_ptr<frz::gpu::Device> device =
      frz::testing::acquire(instance.get());

  // 1. The file, described, planned and uploaded as the engine does it.
  const frz::testing::Front front =
      frz::testing::read_front(FRIZZLE_MODEL_FILE);
  std::string_view name;
  REQUIRE(front.index.read_string("general.architecture", name) ==
          frz::gguf::MetadataError::Ok);
  const frz::arch::Architecture* architecture =
      frz::capability::find_architecture(name);
  REQUIRE(architecture != nullptr);
  frz::model::ModelDescription description{};
  REQUIRE(architecture->describe(front.index, description).ok());
  REQUIRE(description.vocabulary_size == kReferenceVocabulary);
  const frz::gpu::DeviceLimits& granted = device->limits();
  const frz::residency::DeviceLimits limits{
      granted.max_buffer_size, granted.max_storage_buffer_binding_size,
      granted.min_storage_buffer_offset_alignment};
  frz::residency::ResidencyPlan plan;
  const frz::residency::PlanResult planned = frz::residency::plan_residency(
      front.index, description, limits, frz::policy::LoadPolicy{},
      frz::graph::decoder_working_set(description), plan);
  REQUIRE_MESSAGE(planned.ok(), planned.subject);
  const std::span<const std::uint32_t> tokens(kReferenceTokens);
  REQUIRE(plan.context_offered >= tokens.size());
  const std::unique_ptr<frz::residency::Upload> upload =
      upload_model(instance.get(), *device, front, plan);

  // 2. The graph and its program.
  const frz::residency::ResidencyPlan& placed = upload->plan();
  std::vector<frz::kernels::Launch> launches;
  const frz::graph::GraphResult built = architecture->graph(
      description, placed, *frz::capability::find_format(placed.cache_type),
      launches);
  REQUIRE_MESSAGE(built.ok(), built.subject);
  const std::unique_ptr<frz::kernels::Program> program =
      build_program(instance.get(), *upload, std::move(launches));
  const auto scratch =
      std::find_if(placed.scratch.begin(), placed.scratch.end(),
                   [](const frz::residency::PlannedScratch& s) {
                     return s.purpose == "logits";
                   });
  REQUIRE(scratch != placed.scratch.end());
  const frz::residency::BufferRange logits = scratch->range;

  // 3. Each position, against the reference.
  const std::span<const ReferencePosition> reference(kReferencePositions);
  REQUIRE(reference.size() == tokens.size());
  Agreement agreement;
  for (std::uint32_t p = 0; p < tokens.size(); ++p) {
    run_step(instance.get(), *program, p, frz::tokenizer::TokenId{tokens[p]});
    const std::vector<float> row = frz::testing::read_floats(
        instance.get(), *device, upload->buffer(logits.buffer), logits.offset,
        kReferenceVocabulary);
    compare(reference[p], row, agreement);
  }
  MESSAGE("worst deviation from llama.cpp's Metal logits: "
          << agreement.worst_nats << " nats, " << agreement.worst
          << " of its own CPU-to-Metal spread");
  CHECK(agreement.non_finite == 0);
  CHECK(agreement.worst <= 1.0);  // NaN fails too
  CHECK(agreement.top_disagreements == 0);
}
