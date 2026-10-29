// Axis O: changes with a new captured value, or what is sent.
//
// Contract, algorithm, cost and corpus guidance: capture.h. This file
// defines what that header states.

#include "core/capture/capture.h"

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "frizzle/shaders_generated.h"

namespace frizzle::capture {
namespace {

// Binding 1, as capture.wgsl's CaptureLaunch lays it out.
struct Constants {
  std::uint32_t first;
  std::uint32_t width;
};

// One workgroup a launch: in a decode step it reads one word and stops, and
// in the step that ends the prompt each invocation copies width / 256 words.
constexpr std::uint32_t kWorkgroup = 256;

kernels::Binding whole(const residency::BufferRange& range) {
  return {range.buffer, range.offset, range.length};
}

}  // namespace

Layout layout(const model::ModelDescription& model) {
  Layout out;
  out.slots.reserve(model.layers.size());
  std::uint64_t first = 0;
  for (const model::LayerDescription& l : model.layers) {
    out.slots.push_back(
        {static_cast<std::uint32_t>(first), l.feed_forward_width});
    first += l.feed_forward_width;
  }
  out.words = first;
  return out;
}

kernels::Launch capture_launch(const residency::BufferRange& activation,
                               const residency::BufferRange& store, Slot slot) {
  const Constants constants{slot.first, slot.width};
  std::vector<std::byte> bytes(sizeof constants);
  std::memcpy(bytes.data(), &constants, sizeof constants);
  return kernels::Launch{shaders::capture,
                         nullptr,
                         std::move(bytes),
                         {whole(activation), whole(store)},
                         kWorkgroup,
                         kWorkgroup,
                         kernels::Rows::LastToken,
                         {}};
}

}  // namespace frizzle::capture
