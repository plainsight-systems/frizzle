#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with a new weight format.
//
// Q4_0: 32 weights a block, an fp16 scale and 32 four-bit codes, 4.5 bits a
// weight. Its unpack is q4_0.wgsl; its layout, codes then scales, is
// kQ4_0Layout (device_layout.h).
inline constexpr Format kQ4_0{kQ4_0Layout, shaders::q4_0};

}  // namespace frizzle::formats
