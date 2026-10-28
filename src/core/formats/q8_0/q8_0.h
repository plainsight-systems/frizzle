#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with a new weight format.
//
// Q8_0: 32 weights a block, an fp16 scale and 32 signed 8-bit codes, 8.5
// bits a weight; Gemma 3's token embedding. Its unpack is q8_0.wgsl; its
// layout, codes then scales, is kQ8_0Layout (device_layout.h).
inline constexpr Format kQ8_0{kQ8_0Layout, shaders::q8_0};

}  // namespace frizzle::formats
