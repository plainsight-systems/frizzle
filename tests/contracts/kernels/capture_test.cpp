// Axis V: the kernels suite (tests/README.md), the capture.
//
// Contract locked (capture/capture.h, capture/reader.h): one layer's capture
// launch, alone on the GPU, copies the activation's last row into its
// layer's slot of the store, bit for bit, in the step that ends the prompt
// (asking for logits, its tokens given, not fed), whether that step is a
// prefill block or the one-token recompute of the last prompt position; it
// writes nothing in a decode step (fed) or a prefill step that does not end
// the prompt, and nothing outside its slot in any step. The store is read
// back through the reader, whose words are the store's bits. layout() puts
// each layer's slot after the earlier layers', its own width long.
//
// What it does not check, and what does:
//   - That the graph places the launch after the edit launch, so a pinned
//     neuron is captured as set: the browser scenario's read-back (slice 4,
//     step 2), which reads each pin's captured value from the page.
//   - That the captured values are the model's: the oracle, indirectly; the
//     activation the copy reads is the one the down projection reads, whose
//     logits it compares with llama.cpp's.
//   - The engine's rule that a Cancelled or failed turn reports NotTaken
//     (runtime/engine.h): no test. It needs a turn cancelled at a chosen
//     step on the real model.
//   - The reader's failures (residency::Mapped other than Ok): a device
//     cannot be made to fail a mapping on demand.
//
// Inputs, all made here: a 3-layer description of feed-forward widths 200,
// 136 and 264; an activation of 3 rows of layer 1's 136 words; a store of
// layout()'s 600 words, filled first with a pattern that no activation word
// shares; a program of one launch, layer 1's, over a plan of those two
// working buffers. The activation holds a NaN with a payload, both
// infinities, a negative zero and a subnormal in its last row. No model file.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.7    State postconditions: every word of the store is compared, the
//            other layers' slots too, not only the copied row.
//   C++ performance guidelines
//     GPU.1  Keep data on the device: each case writes its inputs with the
//            step and reads the store back once.

#include "core/capture/capture.h"

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include <array>
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
#include "core/capture/reader.h"
#include "core/gguf/index.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/kernels/interface.h"
#include "core/kernels/program.h"
#include "core/model/model_description.h"
#include "core/residency/mapping.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "support/gpu.h"

namespace {

namespace frz = frizzle;

constexpr std::array<std::uint32_t, 3> kWidths = {200, 136, 264};
constexpr std::uint32_t kLayer = 1;  // the launch under test
constexpr std::uint32_t kWidth = kWidths[kLayer];
constexpr std::uint32_t kRows = 3;
constexpr std::size_t kChunk = std::size_t{1} << 20U;

frz::model::ModelDescription description() {
  frz::model::ModelDescription model{};
  for (const std::uint32_t width : kWidths) {
    frz::model::LayerDescription layer{};
    layer.feed_forward_width = width;
    model.layers.push_back(layer);
  }
  return model;
}

// A plan of two working buffers: the activation and the store.
frz::residency::ResidencyPlan two_buffers(std::uint64_t store_words) {
  const std::uint64_t activation = std::uint64_t{kRows} * kWidth * 4;
  const std::uint64_t store = store_words * 4;
  frz::residency::ResidencyPlan plan;
  plan.buffers = {{frz::residency::Pool::Scratch, activation},
                  {frz::residency::Pool::Scratch, store}};
  plan.scratch = {
      {"activation", {frz::residency::BufferIndex{0}, 0, activation}},
      {"captures", {frz::residency::BufferIndex{1}, 0, store}}};
  plan.scratch_bytes = activation + store;
  plan.total_bytes = plan.scratch_bytes;
  return plan;
}

// Words that sit in the store before a step: none equal to any activation
// word below, so a word copied, or one left alone, is told apart.
std::vector<std::uint32_t> pattern(std::uint64_t words) {
  std::vector<std::uint32_t> p(words);
  for (std::size_t i = 0; i < p.size(); ++i) {
    p[i] = 0xA5A50000U + static_cast<std::uint32_t>(i);
  }
  return p;
}

// Every row distinct, with the values a float copy may not keep in each:
// a quiet NaN with a payload, a signalling NaN, both infinities, a negative
// zero and a subnormal.
constexpr std::array<std::uint32_t, 6> kSpecial = {0x7FC01234U, 0x7F800001U,
                                                   0x7F800000U, 0xFF800000U,
                                                   0x80000000U, 0x00000123U};

std::vector<std::uint32_t> activation() {
  std::vector<std::uint32_t> a(std::size_t{kRows} * kWidth);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = 0x40000000U + static_cast<std::uint32_t>(i);
  }
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::size_t k = 0; k < kSpecial.size(); ++k) {
      a[(std::size_t{row} * kWidth) + 3 + (17 * k)] = kSpecial.at(k);
    }
  }
  return a;
}

// The device, an upload of the plan (no file bytes), a program of layer 1's
// capture launch, and the reader over the store.
class Harness {
 public:
  Harness()
      : instance_(wgpuCreateInstance(nullptr)),
        device_(frz::testing::acquire(instance_.get())),
        layout_(frz::capture::layout(description())) {
    struct Ready {
      std::unique_ptr<frz::residency::Upload> upload;
      std::string why;
      bool done = false;
    } ready;
    const frz::gguf::TensorIndex index;
    frz::residency::Upload::begin(
        *device_, index, two_buffers(layout_.words), 0,
        frz::capability::find_format, {}, kChunk,
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
    store_ = plan.scratch.at(1).range;
    std::vector<frz::kernels::Launch> launches;
    launches.push_back(frz::capture::capture_launch(activation_, store_,
                                                    layout_.slots.at(kLayer)));
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

    struct Made {
      std::unique_ptr<frz::capture::Reader> reader;
      std::string message;
      bool done = false;
    } made;
    frz::capture::Reader::make(
        *device_, upload_->buffer(store_.buffer), store_,
        [](std::unique_ptr<frz::capture::Reader> reader,
           std::string_view message, void* userdata) {
          auto& m = *static_cast<Made*>(userdata);
          m.reader = std::move(reader);
          m.message = message;
          m.done = true;
        },
        &made);
    frz::testing::pump_until(instance_.get(), made.done, "the reader");
    REQUIRE_MESSAGE(made.reader != nullptr, made.message);
    reader_ = std::move(made.reader);
  }

  [[nodiscard]] const frz::capture::Layout& layout() const { return layout_; }

  // Runs one step of `tokens` rows, asking for logits or not, fed or not,
  // over `activation` and a store holding `before`; returns the store as
  // the reader reads it, as words.
  std::vector<std::uint32_t> run(std::uint32_t tokens, bool logits, bool fed,
                                 std::span<const std::uint32_t> activation,
                                 std::span<const std::uint32_t> before) {
    const std::vector<frz::kernels::RunInput> inputs{
        {{activation_.buffer, activation_.offset, activation.size_bytes()},
         std::as_bytes(activation)},
        {{store_.buffer, store_.offset, before.size_bytes()},
         std::as_bytes(before)}};
    frz::kernels::Step step{};
    step.position = 0;
    step.tokens = tokens;
    step.logits = logits ? 1 : 0;
    step.fed = fed ? 1 : 0;
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

    struct Read {
      frz::residency::Mapped outcome = frz::residency::Mapped::Internal;
      std::vector<std::uint32_t> words;
      bool done = false;
    } read;
    reader_->read(
        [](frz::residency::Mapped outcome, std::span<const float> words,
           void* userdata) {
          auto& r = *static_cast<Read*>(userdata);
          r.outcome = outcome;
          for (const float w : words) {
            r.words.push_back(std::bit_cast<std::uint32_t>(w));
          }
          r.done = true;
        },
        &read);
    frz::testing::pump_until(instance_.get(), read.done, "the read");
    REQUIRE(read.outcome == frz::residency::Mapped::Ok);
    return read.words;
  }

 private:
  frz::gpu::Instance instance_;
  std::unique_ptr<frz::gpu::Device> device_;
  frz::capture::Layout layout_;
  std::unique_ptr<frz::residency::Upload> upload_;
  std::unique_ptr<frz::kernels::Program> program_;
  std::unique_ptr<frz::capture::Reader> reader_;
  frz::residency::BufferRange activation_{};
  frz::residency::BufferRange store_{};
};

// What the store must hold: `before`, with the slot replaced by `row`, a
// row of the activation.
std::vector<std::uint32_t> copied(std::span<const std::uint32_t> before,
                                  const frz::capture::Slot& slot,
                                  std::span<const std::uint32_t> row) {
  std::vector<std::uint32_t> out(before.begin(), before.end());
  for (std::uint32_t i = 0; i < slot.width; ++i) {
    out.at(slot.first + i) = row[i];
  }
  return out;
}

// Row `row` of an activation of rows of kWidth.
std::span<const std::uint32_t> row_of(std::span<const std::uint32_t> a,
                                      std::uint32_t row) {
  return a.subspan(std::size_t{row} * kWidth, kWidth);
}

void same_words(std::span<const std::uint32_t> got,
                std::span<const std::uint32_t> want) {
  REQUIRE(got.size() == want.size());
  std::size_t differing = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      if (differing == 0) {
        MESSAGE("first difference at word " << i << ": " << std::hex << got[i]
                                            << " for " << want[i]);
      }
      ++differing;
    }
  }
  CHECK(differing == 0);
}

}  // namespace

TEST_CASE("layout puts each layer's slot after the earlier layers'") {
  const frz::capture::Layout l = frz::capture::layout(description());
  REQUIRE(l.slots.size() == kWidths.size());
  CHECK(l.slots[0].first == 0);
  CHECK(l.slots[0].width == 200);
  CHECK(l.slots[1].first == 200);
  CHECK(l.slots[1].width == 136);
  CHECK(l.slots[2].first == 336);
  CHECK(l.slots[2].width == 264);
  CHECK(l.words == 600);
}

TEST_CASE("a prefill step that ends the prompt copies its last row") {
  Harness h;
  const std::vector<std::uint32_t> a = activation();
  const std::vector<std::uint32_t> before = pattern(h.layout().words);
  same_words(h.run(kRows, true, false, a, before),
             copied(before, h.layout().slots.at(kLayer), row_of(a, kRows - 1)));
}

TEST_CASE("the one-token step that ends the prompt copies its one row") {
  Harness h;
  const std::vector<std::uint32_t> a = activation();
  const std::vector<std::uint32_t> before = pattern(h.layout().words);
  same_words(h.run(1, true, false, a, before),
             copied(before, h.layout().slots.at(kLayer), row_of(a, 0)));
}

TEST_CASE("a decode step changes nothing") {
  Harness h;
  const std::vector<std::uint32_t> before = pattern(h.layout().words);
  same_words(h.run(1, true, true, activation(), before), before);
}

TEST_CASE("a prefill step that does not end the prompt changes nothing") {
  Harness h;
  const std::vector<std::uint32_t> before = pattern(h.layout().words);
  same_words(h.run(kRows, false, false, activation(), before), before);
}
