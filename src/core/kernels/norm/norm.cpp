// Axis E: changes with a new or optimized kernel.
//
// Contract, algorithm, cost and corpus guidance: norm.h. This file
// defines what that header states.

#include "core/kernels/norm/norm.h"

#include <cstdint>
#include <cstring>

#include "frizzle/shaders_generated.h"

namespace frizzle::kernels {
namespace {

// Binding 1, as norm.wgsl's Norm lays it out.
struct Constants {
  float epsilon;
  float post_scale;
  std::uint32_t residual_slot;  // 0 where the launch captures no residual
  std::uint32_t added_slot;     // 0 where it captures no addition
};

constexpr std::uint32_t kWorkgroupSize = 256;

Binding whole(const residency::WeightView& view) {
  const residency::WeightPiece& piece = view.pieces().front();
  return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) {
  return {range.buffer, range.offset, range.length};
}

}  // namespace

Launch norm_launch(const NormLaunch& n) {
  const Constants constants{n.epsilon, n.post_scale,
                            n.capture.residual.value_or(0),
                            n.capture.added.value_or(0)};
  std::vector<std::byte> bytes(sizeof constants);
  std::memcpy(bytes.data(), &constants, sizeof constants);
  const bool post = n.post_gain != nullptr;
  // Without a post-norm, its binding takes the gain's place: read-only, as
  // the launch never writes it.
  const Binding gain = whole(*n.gain);
  const std::uint64_t width = n.gain->shape().dimensions[0];
  return Launch{
      shaders::norm,
      nullptr,
      std::move(bytes),
      {gain, post ? whole(*n.post_gain) : gain, whole(n.output),
       whole(n.hidden), whole(n.normed), whole(n.capture.store)},
      kWorkgroupSize,
      kWorkgroupSize,
      n.rows,
      {{"add", n.add ? 1.0 : 0.0},
       {"post_norm", post ? 1.0 : 0.0},
       {"capture_residual", n.capture.residual.has_value() ? 1.0 : 0.0},
       {"capture_added", n.capture.added.has_value() ? 1.0 : 0.0},
       {"width", static_cast<double>(width)}},
  };
}

}  // namespace frizzle::kernels
