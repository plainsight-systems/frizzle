#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with a new weight format.
//
// Q4_1: 32 weights a block, an fp16 scale and minimum and 32 four-bit codes,
// 5 bits a weight. Its unpack is q4_1.wgsl; its layout, codes then scale and
// minimum, is kQ4_1Layout (device_layout.h).
inline constexpr Format kQ4_1{kQ4_1Layout, shaders::q4_1};

}  // namespace frizzle::formats
