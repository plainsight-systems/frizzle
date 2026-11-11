#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with the F32 format: its layout on the device or its
// unpack.
//
// F32: one float a block, stored as the file holds it; the norms' gains.
// Its unpack is f32.wgsl; its layout, one stream, is kF32Layout
// (device_layout.h).
//
// Contract
//   - kF32 is the Format of GGUF's F32 (format.h): an unpack and its layout.
//     It has no pack: the cache stores in F16.
//   - The format's definition: value i is the i-th IEEE 754 binary32 value,
//     four bytes, little-endian, as WGSL reads a u32.
//   - unpack(base, blocks_in_piece, group) returns values 32 × group ..
//     32 × group + 31 of the piece in order, as stored: nothing is
//     computed, so every bit comes back (format.h, Numbers).
//
// Algorithm, unpack of group g, in 4-byte words counted from `base`
//   1. For v = 0 .. 7: words 32g + 4v .. 32g + 4v + 3 are values
//      32g + 4v .. 32g + 4v + 3, reinterpreted as f32 in vec4 v. One
//      stream, so blocks_in_piece is not read.
//
// Design and cost
//   - A group, counted: 32 words read, all values; no arithmetic.
//   - Gemma 3 1B's pinned file, counted: 157 F32 tensors, every norm's gain:
//     105 of 1,152 values and 52 of 256, 134,272 values, 537,088 bytes.
//
// Corpus guidance
//   C++ Core Guidelines
//     Con.5  constexpr for values computable at compile time: kF32.
//   C++ performance guidelines
//     Searched for the cost of reading unquantized values in a kernel's load
//     path: the rules found are for lane layout, host transfers and passes
//     over memory, which format.h and each kernel answer; none applies to
//     this file's read alone.
inline constexpr Format kF32{kF32Layout, UnpackWgsl{shaders::f32}};

}  // namespace frizzle::formats
