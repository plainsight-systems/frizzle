#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with a new weight format.
//
// F16: one half-precision value a block, two bytes; the KV cache's storage
// at load policy's default precision, and the weights a file stores at half
// precision. Its unpack is f16.wgsl; its pack and pack's inverse, unpack4,
// are f16_pack.wgsl; its layout,
// one stream, is kF16Layout (device_layout.h).
inline constexpr Format kF16{kF16Layout, shaders::f16, shaders::f16_pack};

}  // namespace frizzle::formats
