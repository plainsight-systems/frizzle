#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with the Q4_1 format: its layout on the device or its
// unpack.
//
// Q4_1: 32 weights a block, an fp16 scale d, an fp16 minimum m and 32
// four-bit codes, 20 bytes, 5 bits a weight. Its unpack is q4_1.wgsl; its
// layout, codes then scale and minimum, is kQ4_1Layout (device_layout.h).
//
// Contract
//   - kQ4_1 is the Format of GGUF's Q4_1 (format.h): an unpack and its
//     layout. It has no pack: the cache never stores in a block-scaled
//     format.
//   - The format's definition: byte j of a block's 16 code bytes qs holds
//     code j in its low four bits and code j + 16 in its high four; weight
//     i is code_i × d + m.
//   - unpack(base, blocks_in_piece, group) returns block `group`'s 32
//     weights in order, each rounded once in f32, at the add, the same
//     whether the target fuses the multiply-add or not (format.h, Numbers).
//
// Algorithm, unpack of block g of a piece of n blocks, in 4-byte words
// counted from `base`
//   1. The scale and minimum: word 4n + g, after every block's four words of
//      codes, by unpack2x16float: d its lower half, m its upper, exact
//      (format.h, Numbers).
//   2. For w = 0 .. 3: word 4g + w holds the block's code bytes 4w .. 4w + 3,
//      low byte first. Each byte's low nibble, converted to f32, an integer
//      in [0, 15], times d plus m gives weights 4w .. 4w + 3; its high
//      nibble's give weights 16 + 4w .. 19 + 4w. The product is exact; the
//      add rounds once.
//   3. Weights 4w .. 4w + 3 are vec4 w, and weights 16 + 4w .. 19 + 4w vec4
//      w + 4, so the eight vec4s hold weights 0 .. 31 in order.
//
// Design and cost
//   - A group, counted: five words read, all of them the block's; 16 vec4
//     multiplies and 16 vec4 adds, or 16 vec4 multiply-adds where the
//     target fuses them.
//   - Gemma 3 1B's pinned file, counted: 3 Q4_1 tensors, the first three
//     ffn_down; 746,496 blocks, 14,929,920 bytes.
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.101 Use unsigned types for bit manipulation: codes are cut from u32
//            words by unsigned shifts and masks, and only whole small
//            integers reach f32.
//     Con.5  constexpr for values computable at compile time: kQ4_1.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path:
//            a block's codes decode in the kernel's registers, its scale and
//            minimum read together from one word of their own stream.
inline constexpr Format kQ4_1{kQ4_1Layout, UnpackWgsl{shaders::q4_1}};

}  // namespace frizzle::formats
