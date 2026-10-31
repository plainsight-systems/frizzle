// Axis V: the kernels suite (tests/README.md), the edit kernel.
//
// Contract locked (edit/edit_kernel.h): one layer's edit launch, alone on
// the GPU, makes each listed neuron edit::apply's result, bit for bit, in
// exactly the rows whose positions its range holds, row r being position
// step.position + r, in any step, prefill or decode, scoring or not; a
// range that begins or ends inside a block edits the rows on its side
// only; two ranges of one neuron each edit their own rows; a decode step
// past a range's end changes nothing. The range is placed at the table's
// origin (edit::table). It applies its own layer's edits and no other's;
// more than one workgroup's worth (64) all apply; Scale × 1 keeps a
// subnormal's bits; every other value keeps its bits.
//
// Inputs, all made here: an activation of 5 rows of 200 floats, a table
// (edit::table) for a 3-layer description of that width, and a program of
// one launch, layer 1's, over a plan of just those two working buffers. No
// model file.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.7    State postconditions: every value the launch must not change is
//            compared too, not only the edited ones.
//   C++ performance guidelines
//     GPU.1  Keep data on the device: the inputs ride with the step's own
//            writes, and each case reads the activation back once.

#include "core/edit/edit_kernel.h"

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/capability/capability.h"
#include "core/edit/edit.h"
#include "core/gguf/index.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/kernels/interface.h"
#include "core/kernels/program.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "support/gpu.h"

namespace {

namespace frz = frizzle;
using frz::edit::Edit;
using frz::edit::EditKind;
using frz::edit::kToTheEnd;

constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kWidth = 200;
constexpr std::uint32_t kRows = 5;
// Steps count from the origin: step s is position kOrigin + s.
constexpr std::uint32_t kOrigin = 40;
constexpr std::uint32_t kLayer = 1;  // the launch under test
constexpr std::size_t kChunk = std::size_t{1} << 20U;

// A step's rows: `tokens` of them from `position`, named so the two are
// never swapped.
struct StepRows {
  std::uint32_t position;
  std::uint32_t tokens;
};

frz::model::ModelDescription description() {
  frz::model::ModelDescription model{};
  frz::model::LayerDescription layer{};
  layer.feed_forward_width = kWidth;
  model.layers.assign(kLayers, layer);
  return model;
}

std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }

// A plan of two working buffers: the activation and the table.
frz::residency::ResidencyPlan two_buffers() {
  const std::uint64_t activation = std::uint64_t{kRows} * kWidth * 4;
  const std::uint64_t table = ((std::uint64_t{2} * kLayers) +
                               (std::uint64_t{5} * frz::edit::kMaxEdits)) *
                              4;
  frz::residency::ResidencyPlan plan;
  plan.buffers = {{frz::residency::Pool::Scratch, activation},
                  {frz::residency::Pool::Scratch, table}};
  plan.scratch = {
      {"activation", {frz::residency::BufferIndex{0}, 0, activation}},
      {"edit_table", {frz::residency::BufferIndex{1}, 0, table}}};
  plan.scratch_bytes = activation + table;
  plan.total_bytes = plan.scratch_bytes;
  return plan;
}

// The device, an upload of the plan (no file bytes), and a program of
// layer 1's edit launch.
class Harness {
 public:
  Harness()
      : instance_(wgpuCreateInstance(nullptr)),
        device_(frz::testing::acquire(instance_.get())) {
    struct Ready {
      std::unique_ptr<frz::residency::Upload> upload;
      std::string why;
      bool done = false;
    } ready;
    const frz::gguf::TensorIndex index;
    frz::residency::Upload::begin(
        *device_, index, two_buffers(), 0, frz::capability::find_format, {},
        kChunk,
        [](std::unique_ptr<frz::residency::Upload> upload,
           frz::residency::UploadError error, const char* subject,
           void* userdata) {
          auto& r = *static_cast<Ready*>(userdata);
          r.upload = std::move(upload);
          r.why = std::string(frz::residency::to_string(error)) + " " +
                  (subject != nullptr ? subject : "");
          r.done = true;
        },
        &ready);
    frz::testing::pump_until(instance_.get(), ready.done, "the upload");
    REQUIRE_MESSAGE(ready.upload != nullptr, ready.why);
    upload_ = std::move(ready.upload);
    struct Finished {
      frz::residency::UploadError error{};
      bool done = false;
    } finished;
    upload_->finish(
        [](frz::residency::UploadError error, void* userdata) {
          auto& f = *static_cast<Finished*>(userdata);
          f.error = error;
          f.done = true;
        },
        &finished);
    frz::testing::pump_until(instance_.get(), finished.done, "the finish");
    REQUIRE(finished.error == frz::residency::UploadError::Ok);

    const frz::residency::ResidencyPlan& plan = upload_->plan();
    activation_ = plan.scratch.at(0).range;
    table_ = plan.scratch.at(1).range;
    std::vector<frz::kernels::Launch> launches;
    launches.push_back(
        frz::edit::edit_launch(kLayer, activation_, table_, kWidth));
    struct Built {
      std::unique_ptr<frz::kernels::Program> program;
      std::string message;
      bool done = false;
    } built;
    frz::kernels::Program::build(
        *upload_, std::move(launches), std::nullopt,
        [](std::unique_ptr<frz::kernels::Program> program,
           frz::kernels::ProgramError /*error*/, std::string_view message,
           void* userdata) {
          auto& b = *static_cast<Built*>(userdata);
          b.program = std::move(program);
          b.message = message;
          b.done = true;
        },
        &built);
    frz::testing::pump_until(instance_.get(), built.done, "the program");
    REQUIRE_MESSAGE(built.program != nullptr, built.message);
    program_ = std::move(built.program);
  }

  // Runs one step of `rows`, asking for logits or
  // not, over `activation` and the table of `edits` placed at kOrigin;
  // returns the activation after.
  std::vector<float> run(StepRows rows, bool logits,
                         std::span<const float> activation,
                         std::span<const Edit> edits) {
    REQUIRE(frz::edit::check(edits, description_).ok());
    const std::vector<std::uint32_t> table =
        frz::edit::table(edits, description_, kOrigin);
    const std::vector<frz::kernels::RunInput> inputs{
        {{activation_.buffer, activation_.offset, activation.size_bytes()},
         std::as_bytes(activation)},
        {{table_.buffer, table_.offset, table.size() * sizeof(std::uint32_t)},
         std::as_bytes(std::span(table))}};
    frz::kernels::Step step{};
    step.position = rows.position;
    step.tokens = rows.tokens;
    step.logits = logits ? 1 : 0;
    struct Ran {
      frz::kernels::ProgramError error{};
      std::string message;
      bool done = false;
    } ran;
    program_->run(
        step, inputs, {},
        [](frz::kernels::ProgramError error, std::string_view message,
           std::span<const std::byte> /*readback*/, void* userdata) {
          auto& r = *static_cast<Ran*>(userdata);
          r.error = error;
          r.message = message;
          r.done = true;
        },
        &ran);
    frz::testing::pump_until(instance_.get(), ran.done, "the step");
    REQUIRE_MESSAGE(ran.error == frz::kernels::ProgramError::Ok, ran.message);
    return frz::testing::read_floats(instance_.get(), *device_,
                                     upload_->buffer(activation_.buffer),
                                     activation_.offset, activation.size());
  }

 private:
  frz::gpu::Instance instance_;
  std::unique_ptr<frz::gpu::Device> device_;
  std::unique_ptr<frz::residency::Upload> upload_;
  std::unique_ptr<frz::kernels::Program> program_;
  frz::residency::BufferRange activation_{};
  frz::residency::BufferRange table_{};
  frz::model::ModelDescription description_ = description();
};

// Distinct, deterministic values, a subnormal and a negative zero among them.
constexpr std::uint32_t kSubnormalNeuron = 7;
constexpr std::uint32_t kNegativeZeroNeuron = 8;
constexpr std::uint32_t kSubnormalBits = 0x00000123;

std::vector<float> activation() {
  std::vector<float> a(std::size_t{kRows} * kWidth);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = (static_cast<float>(i % 97) * 0.375F) - 13.0F;
  }
  for (std::uint32_t row = 0; row < kRows; ++row) {
    a[(std::size_t{row} * kWidth) + kSubnormalNeuron] =
        std::bit_cast<float>(kSubnormalBits);
    a[(std::size_t{row} * kWidth) + kNegativeZeroNeuron] = -0.0F;
  }
  return a;
}

// Layer 1's edits over steps [from, until): more than 64, every kind,
// Scale × 1 on the subnormal and on the negative zero; and edits in layers
// 0 and 2 the launch must ignore, over every step.
std::vector<Edit> edits(std::uint32_t from = 0,
                        std::uint32_t until = kToTheEnd) {
  std::vector<Edit> e;
  constexpr std::uint32_t kMany = 150;
  for (std::uint32_t n = 10; n < 10 + kMany; ++n) {
    switch (n % 3) {
      case 0:
        e.push_back({kLayer, n, EditKind::Set, static_cast<float>(n) * 0.5F,
                     from, until});
        break;
      case 1:
        e.push_back({kLayer, n, EditKind::Scale, -1.75F, from, until});
        break;
      default:
        e.push_back({kLayer, n, EditKind::Zero, 0.0F, from, until});
        break;
    }
  }
  e.push_back({kLayer, kSubnormalNeuron, EditKind::Scale, 1.0F, from, until});
  e.push_back(
      {kLayer, kNegativeZeroNeuron, EditKind::Scale, 1.0F, from, until});
  e.push_back({0, 3, EditKind::Set, 999.0F, 0, kToTheEnd});
  e.push_back({2, 4, EditKind::Set, -999.0F, 0, kToTheEnd});
  e.push_back({0, 20, EditKind::Zero, 0.0F, 0, kToTheEnd});
  return e;
}

// What the launch must leave: `before`, with layer 1's edits applied by
// edit::apply to each of the step's `rows` whose position the edit's
// range, placed at kOrigin, holds.
std::vector<float> expected(std::span<const float> before,
                            std::span<const Edit> edits, StepRows rows) {
  std::vector<float> out(before.begin(), before.end());
  for (const Edit& e : frz::edit::placed(edits, kOrigin)) {
    if (e.layer != kLayer) continue;
    for (std::uint32_t row = 0; row < rows.tokens; ++row) {
      const std::uint32_t at = rows.position + row;
      if (at < e.from || at >= e.until) continue;
      float& v = out.at((std::size_t{row} * kWidth) + e.neuron);
      v = frz::edit::apply(e.kind, e.value, v);
    }
  }
  return out;
}

void same_bits(std::span<const float> got, std::span<const float> want) {
  REQUIRE(got.size() == want.size());
  std::size_t differing = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (bits(got[i]) != bits(want[i])) {
      if (differing == 0) {
        MESSAGE("first difference at row " << i / kWidth << " neuron "
                                           << i % kWidth << ": " << got[i]
                                           << " for " << want[i]);
      }
      ++differing;
    }
  }
  CHECK(differing == 0);
}

}  // namespace

TEST_CASE("an edit from the step that ends the prompt edits that row only") {
  Harness h;
  const std::vector<float> before = activation();
  const std::vector<Edit> e = edits();
  // Positions 36 to 40: step 0, position 40, is the block's last row.
  const auto got = h.run({.position = kOrigin - kRows + 1, .tokens = kRows},
                         true, before, e);
  same_bits(got, expected(before, e,
                          {.position = kOrigin - kRows + 1, .tokens = kRows}));
  same_bits(std::span(got).first(std::size_t{kRows - 1} * kWidth),
            std::span(before).first(std::size_t{kRows - 1} * kWidth));
}

TEST_CASE("a range that begins inside a block edits the rows from it on") {
  Harness h;
  const std::vector<float> before = activation();
  const std::vector<Edit> e = edits(2);  // from position 42
  // Positions 40 to 44, a block that does not score: rows 2 to 4 edited.
  same_bits(h.run({.position = kOrigin, .tokens = kRows}, false, before, e),
            expected(before, e, {.position = kOrigin, .tokens = kRows}));
}

TEST_CASE("a range that ends inside a block edits the rows before it") {
  Harness h;
  const std::vector<float> before = activation();
  const std::vector<Edit> e = edits(0, 3);  // positions 40 to 42
  same_bits(h.run({.position = kOrigin, .tokens = kRows}, true, before, e),
            expected(before, e, {.position = kOrigin, .tokens = kRows}));
}

TEST_CASE("two ranges of one neuron each edit their own rows") {
  Harness h;
  const std::vector<float> before = activation();
  const std::vector<Edit> e{{kLayer, 50, EditKind::Set, 1.0F, 0, 2},
                            {kLayer, 50, EditKind::Set, 2.0F, 2, 4},
                            {kLayer, 50, EditKind::Scale, -3.0F, 4, kToTheEnd}};
  const auto got =
      h.run({.position = kOrigin, .tokens = kRows}, true, before, e);
  same_bits(got, expected(before, e, {.position = kOrigin, .tokens = kRows}));
  CHECK(got.at(50) == 1.0F);
  CHECK(got.at((std::size_t{2} * kWidth) + 50) == 2.0F);
}

TEST_CASE("a decode step inside a range edits its row; past the end, nothing") {
  Harness h;
  const std::vector<float> before = activation();
  const std::vector<Edit> e = edits(0, 6);  // positions 40 to 45
  same_bits(h.run({.position = kOrigin + 5, .tokens = 1}, true, before, e),
            expected(before, e, {.position = kOrigin + 5, .tokens = 1}));
  same_bits(h.run({.position = kOrigin + 6, .tokens = 1}, true, before, e),
            before);
}

TEST_CASE("a block before every range changes nothing") {
  Harness h;
  const std::vector<float> before = activation();
  same_bits(h.run({.position = 0, .tokens = kRows}, false, before, edits()),
            before);
}

TEST_CASE("no edits change nothing") {
  Harness h;
  const std::vector<float> before = activation();
  same_bits(h.run({.position = kOrigin, .tokens = kRows}, true, before, {}),
            before);
}
