#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with a new weight format.
//
// F32: one float a block, stored as the file holds it; the norms and other
// small vectors. Its unpack is f32.wgsl; its layout, one stream, is
// kF32Layout (device_layout.h).
inline constexpr Format kF32{kF32Layout, shaders::f32};

}  // namespace frizzle::formats
