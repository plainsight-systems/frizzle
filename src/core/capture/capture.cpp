// Axis O: changes with a new captured value, or what is sent.
//
// Contract, algorithm, cost and corpus guidance: capture.h. This file
// defines what that header states.

#include "core/capture/capture.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
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
  out.layers.resize(model.layers.size());
  // Offsets are summed in 64 bits; a slot's first word is a u32 constant of
  // the launches that write it (ES.103).
  std::uint64_t first = 0;
  const auto take = [&first](std::uint64_t width) {
    if (first + width > std::numeric_limits<std::uint32_t>::max()) {
      std::abort();  // a store of 16 GiB: no plan could place it
    }
    const Slot slot{static_cast<std::uint32_t>(first),
                    static_cast<std::uint32_t>(width)};
    first += width;
    return slot;
  };
  // Region 1: every layer's neurons, contiguous.
  for (std::size_t i = 0; i < model.layers.size(); ++i) {
    out.layers[i].neurons = take(model.layers[i].feed_forward_width);
  }
  out.neuron_words = first;
  // Region 2: each layer's rest, in the header's order.
  const std::uint64_t width = model.embedding_width;
  for (std::size_t i = 0; i < model.layers.size(); ++i) {
    const model::LayerDescription& l = model.layers[i];
    LayerSlots& s = out.layers[i];
    s.keys = std::min(l.attention_window, model.trained_context);
    s.query_heads = l.query_heads;
    s.residual = take(width);
    s.attention_added = take(width);
    s.mlp_added = take(width);
    s.weights = take(std::uint64_t{l.query_heads} * s.keys);
    s.stats = take(std::uint64_t{l.query_heads} * 2);
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
