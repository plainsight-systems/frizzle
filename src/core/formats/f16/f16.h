#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with the F16 format: its layout on the device, its unpack
// or its pack.
//
// F16: one half-precision value a block, two bytes; the KV cache's storage,
// and the weights a file stores at half precision. Its unpack is f16.wgsl;
// its pack and pack's inverse, unpack4, are f16_pack.wgsl; its layout, one
// stream, is kF16Layout (device_layout.h).
//
// Contract
//   - kF16 is the Format of GGUF's F16 (format.h): an unpack, a pack and its
//     layout. It is the one format with a pack, and the cache's only
//     precision (policy::CachePrecision), so every cached key and value is
//     stored in it.
//   - The format's definition: value i is the i-th IEEE 754 binary16 value,
//     two bytes, little-endian, so a u32 word holds value 2k in its lower
//     half and value 2k + 1 in its upper.
//   - unpack(base, blocks_in_piece, group) returns values 32 × group ..
//     32 × group + 31 of the piece in order, exact: every f16 is an f32.
//   - pack(values) returns the two words that store four consecutive values,
//     each saturated to ±65,504 and rounded once to F16; unpack4(words)
//     returns the four values two words store, exact. Neither reads or
//     writes a binding. The rounding point is format.h's (Numbers).
//
// Algorithm
//   unpack of group g, in 4-byte words counted from `base`:
//   1. For v = 0 .. 7: words 16g + 2v and 16g + 2v + 1 hold values
//      32g + 4v .. 32g + 4v + 3, lower half first; unpack2x16float of each
//      gives two, in vec4 v. One stream, so blocks_in_piece is not read.
//   pack of four values x, y, z, w:
//   1. Each is clamped to [−65,504, 65,504], F16's largest finite.
//   2. pack2x16float(x, y) is the first word and pack2x16float(z, w) the
//      second, the lower half first, as unpack reads them: one rounding a
//      value.
//   unpack4 of two words: unpack2x16float of each, the first word's two
//   values first.
//
// Design and cost
//   - A group of unpack, counted: 16 words read, all values; 16
//     unpack2x16float. pack: 4 clamps and 2 pack2x16float; unpack4: 2
//     unpack2x16float.
//   - The cache's bytes are counted where it is sized (residency/plan.h)
//     and written (kernels/rope/rope.h). Gemma 3 1B's pinned file stores no
//     F16 weight, counted.
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.46  Avoid lossy narrowing: pack narrows f32 to F16 on purpose, the
//            saturation first so no finite value becomes an infinity, and
//            states its one rounding.
//     Con.5  constexpr for values computable at compile time: kF16.
//   C++ performance guidelines
//     Searched for half-precision storage converted to f32 on load: the
//     rules found are for block-scaled codes and memory tiers; none applies
//     to a 16-bit value read on its own.
inline constexpr Format kF16{kF16Layout, UnpackWgsl{shaders::f16},
                             PackWgsl{shaders::f16_pack}};

}  // namespace frizzle::formats
