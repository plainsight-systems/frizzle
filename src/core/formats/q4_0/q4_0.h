#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with the Q4_0 format: its layout on the device or its
// unpack.
//
// Q4_0: 32 weights a block, an fp16 scale d and 32 four-bit codes, 18 bytes,
// 4.5 bits a weight. Gemma 3 1B's pinned file stores most of its matrices
// in it. Its unpack is q4_0.wgsl; its layout, codes then scales, is
// kQ4_0Layout (device_layout.h).
//
// Contract
//   - kQ4_0 is the Format of GGUF's Q4_0 (format.h): an unpack and its
//     layout. It has no pack: the cache never stores in a block-scaled
//     format.
//   - The format's definition: byte j of a block's 16 code bytes qs holds
//     code j in its low four bits and code j + 16 in its high four; weight
//     i is (code_i − 8) × d.
//   - unpack(base, blocks_in_piece, group) returns block `group`'s 32
//     weights in order, exact in f32 (format.h, Numbers).
//
// Algorithm, unpack of block g of a piece of n blocks, in 4-byte words
// counted from `base`
//   1. The scale: byte s = 16n + 2g of the piece, after every block's 16
//      code bytes; d is half (s / 2) mod 2 of word s / 4, by
//      unpack2x16float, exact (format.h, Numbers).
//   2. For w = 0 .. 3: word 4g + w holds the block's code bytes 4w .. 4w + 3,
//      low byte first. Each byte's low nibble, converted to f32 and less 8,
//      an integer in [−8, 7], times d gives weights 4w .. 4w + 3; its high
//      nibble's give weights 16 + 4w .. 19 + 4w. One multiply a weight,
//      exact.
//   3. Weights 4w .. 4w + 3 are vec4 w, and weights 16 + 4w .. 19 + 4w vec4
//      w + 4, so the eight vec4s hold weights 0 .. 31 in order.
//
// Design and cost
//   - A group, counted: five words read, four of codes and one holding its
//     scale and a neighbour's; 16 vec4 subtractions and 16 vec4 multiplies.
//   - Gemma 3 1B's pinned file, counted: 179 Q4_0 tensors, every matrix but
//     the token embedding and the first three ffn_down; 21,058,560 blocks,
//     379,054,080 bytes.
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.101 Use unsigned types for bit manipulation: codes are cut from u32
//            words by unsigned shifts and masks, and only whole small
//            integers reach f32.
//     Con.5  constexpr for values computable at compile time: kQ4_0.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path:
//            a block's codes decode in the kernel's registers, the scale
//            read from its own stream.
inline constexpr Format kQ4_0{kQ4_0Layout, UnpackWgsl{shaders::q4_0}};

}  // namespace frizzle::formats
