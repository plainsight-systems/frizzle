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
  const bool scaled = n.scale != nullptr;
  const bool renorm = n.renorm_gain != nullptr;
  const bool unit = n.gain == nullptr;
  // Every absent read-only binding takes a present one's place: the gain,
  // or with a unit gain the block's output, which the launch never writes.
  const Binding fill = unit ? whole(n.output) : whole(*n.gain);
  const std::uint64_t width = unit ? n.width : n.gain->shape().dimensions[0];
  return Launch{
      shaders::norm,
      nullptr,
      std::move(bytes),
      {fill, post ? whole(*n.post_gain) : fill, whole(n.output),
       whole(n.hidden), whole(n.normed), scaled ? whole(*n.scale) : fill,
       renorm ? whole(*n.renorm_gain) : fill, whole(n.capture.store)},
      kWorkgroupSize * n.rows_per_token,
      kWorkgroupSize,
      n.rows,
      {{"add", n.add ? 1.0 : 0.0},
       {"post_norm", post ? 1.0 : 0.0},
       {"scaled", scaled ? 1.0 : 0.0},
       {"normalize", n.normalize ? 1.0 : 0.0},
       {"renorm", renorm ? 1.0 : 0.0},
       {"unit_gain", unit ? 1.0 : 0.0},
       {"rows_per_token", static_cast<double>(n.rows_per_token)},
       {"capture_residual", n.capture.residual.has_value() ? 1.0 : 0.0},
       {"capture_added", n.capture.added.has_value() ? 1.0 : 0.0},
       {"width", static_cast<double>(width)}},
  };
}

}  // namespace frizzle::kernels
