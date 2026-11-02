// Axis O: changes with a new captured value, or what is sent.
//
// Contract, algorithm, cost and corpus guidance: weights.h. This file
// defines what that header states.

#include "core/capture/weights.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "frizzle/shaders_generated.h"

namespace frizzle::capture {
namespace {

// Binding 1, as weights.wgsl's WeightsLaunch lays it out.
struct Constants {
  std::uint32_t weights;
  std::uint32_t stats;
  std::uint32_t keys;
  std::uint32_t query_heads;
};

// One workgroup a query head (weights.h).
constexpr std::uint32_t kWorkgroup = 256;

}  // namespace

kernels::Launch weights_launch(const residency::BufferRange& store,
                               const LayerSlots& slots) {
  const Constants constants{slots.weights.first, slots.stats.first, slots.keys,
                            slots.query_heads};
  std::vector<std::byte> bytes(sizeof constants);
  std::memcpy(bytes.data(), &constants, sizeof constants);
  return kernels::Launch{shaders::weights,
                         nullptr,
                         std::move(bytes),
                         {{store.buffer, store.offset, store.length}},
                         slots.query_heads * kWorkgroup,
                         kWorkgroup,
                         kernels::Rows::LastToken,
                         {}};
}

}  // namespace frizzle::capture
