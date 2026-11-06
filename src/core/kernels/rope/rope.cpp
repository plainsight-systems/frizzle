// Axis E: changes with a new or optimized kernel.
//
// Contract, algorithm, cost and corpus guidance: rope.h. This file defines
// what that header states.

#include "core/kernels/rope/rope.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <string_view>

#include "frizzle/shaders_generated.h"

namespace frizzle::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;
// A head's values a lane holds, two vec4s: a head is d / 8 lanes (rope.h).
constexpr std::uint32_t kValuesPerLane = 8;

// The workgroup: the largest power of two from 64 down to a head's d / 8
// lanes that divides both a row's invocations and its rotating heads'
// invocations, so that every invocation of a workgroup works on one token
// and its heads all rotate or are all value heads (rope.h). For Gemma 3 1B
// a row is (4 + 2 × 1) × 32 = 192 invocations and its rotating heads
// (4 + 1) × 32 = 160; 64 does not divide 160, so the workgroup is 32, one
// head.
std::uint32_t workgroup_for(const model::LayerDescription& layer) {
  const std::uint32_t lanes = layer.head_dimension / kValuesPerLane;
  const std::uint32_t row =
      (layer.query_heads + 2 * layer.key_value_heads) * lanes;
  const std::uint32_t rotating =
      (layer.query_heads + layer.key_value_heads) * lanes;
  std::uint32_t size = kWorkgroupSize;
  while (size > lanes && (row % size != 0 || rotating % size != 0)) size /= 2;
  return size;
}
constexpr std::size_t kMaxPairs = 128;  // d / 2 for d up to 256

// Binding 1, as rope.wgsl's Rope lays it out: turns at offset 16, so
// 16 + kMaxPairs × 4 bytes.
constexpr std::size_t kConstantsBytes = 528;
struct Constants {
  std::uint32_t slots;
  float epsilon;
  std::array<std::uint32_t, 2> padding;
  std::array<float, kMaxPairs> turns;
};
static_assert(sizeof(Constants) == kConstantsBytes);

// The one form, assembled from two embedded sources joined at compile time:
// rope.wgsl, then its entry point, rope_cache.wgsl (rope.h). The program
// compares kernels by their text, so the joined form is one pipeline per
// shape: one for all of Gemma 3 1B's layers.
template <std::size_t N>
struct Joined {
  std::array<char, N> chars{};
  [[nodiscard]] constexpr std::string_view view() const {
    return {chars.data(), N};
  }
};

// The parts, in order, as one text of N characters, their total.
template <std::size_t N>
constexpr Joined<N> join(const std::array<std::string_view, 2>& parts) {
  Joined<N> out;
  std::size_t at = 0;
  for (const std::string_view part : parts) {
    for (const char c : part) {
      out.chars.at(at++) = c;
    }
  }
  return out;
}

constexpr auto kCacheForm =
    join<shaders::rope.size() + shaders::rope_cache.size()>(
        {shaders::rope, shaders::rope_cache});

Binding whole(const residency::WeightView& view) {
  const residency::WeightPiece& piece = view.pieces().front();
  return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) {
  return {range.buffer, range.offset, range.length};
}

// The launch's constants: its slots, ε, and each pair's θ_k / 2π, in f64,
// rounded once to f32: WGSL bounds pow only loosely (rope.h).
std::vector<std::byte> constants_of(const model::LayerDescription& layer,
                                    std::uint32_t slots, float epsilon) {
  const std::uint32_t d = layer.head_dimension;
  Constants constants{slots, epsilon, {}, {}};
  for (std::uint32_t k = 0; k < d / 2; ++k) {
    const double theta =
        std::pow(static_cast<double>(layer.rope_base), -2.0 * k / d);
    constants.turns.at(k) = static_cast<float>(theta / (2 * std::numbers::pi));
  }
  const auto bytes = std::as_bytes(std::span(&constants, 1));
  return {bytes.begin(), bytes.end()};
}

// The overrides: the shape and the angle tables' size, d / 2.
std::vector<Override> shape_of(const model::LayerDescription& layer) {
  const std::uint32_t d = layer.head_dimension;
  const std::uint32_t pairs = d / 2;
  return {{"head_dimension", static_cast<double>(d)},
          {"query_heads", static_cast<double>(layer.query_heads)},
          {"key_value_heads", static_cast<double>(layer.key_value_heads)},
          {"pairs_per_head", static_cast<double>(pairs)}};
}

}  // namespace

Launch rope_launch(const RopeLaunch& r) {
  const model::LayerDescription& layer = r.layer;
  const std::uint32_t d = layer.head_dimension;
  return Launch{
      kCacheForm.view(),
      nullptr,
      constants_of(layer, r.cache.slots, r.epsilon),
      {whole(*r.query_norm), whole(*r.key_norm), whole(r.query), whole(r.key),
       whole(r.value), whole(r.cache.keys), whole(r.cache.values)},
      (layer.query_heads + 2 * layer.key_value_heads) * (d / kValuesPerLane),
      workgroup_for(layer),
      Rows::EveryToken,
      shape_of(layer),
      r.cache_format,
  };
}

}  // namespace frizzle::kernels
