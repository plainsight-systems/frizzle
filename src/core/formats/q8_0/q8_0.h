#pragma once

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "frizzle/shaders_generated.h"

namespace frizzle::formats {

// Axis B: changes with the Q8_0 format: its layout on the device or its
// unpack.
//
// Q8_0: 32 weights a block, an fp16 scale d and 32 signed 8-bit codes, 34
// bytes, 8.5 bits a weight; Gemma 3's token embedding. Its unpack is
// q8_0.wgsl; its layout, codes then scales, is kQ8_0Layout
// (device_layout.h).
//
// Contract
//   - kQ8_0 is the Format of GGUF's Q8_0 (format.h): an unpack and its
//     layout. It has no pack: a block's scale is set by its 32 values'
//     largest magnitude, which four values on their own cannot give.
//   - The format's definition: byte j of a block's 32 code bytes qs is code
//     j, a two's-complement integer in [−128, 127]; weight j is qs[j] × d.
//   - unpack(base, blocks_in_piece, group) returns block `group`'s 32
//     weights in order, exact in f32 (format.h, Numbers).
//
// Algorithm, unpack of block g of a piece of n blocks, in 4-byte words
// counted from `base`
//   1. The scale: word 8n + g / 2, after every block's eight words of codes;
//      d is its half g mod 2, by unpack2x16float, exact (format.h,
//      Numbers).
//   2. For v = 0 .. 7: word 8g + v holds codes 4v .. 4v + 3, low byte first.
//      Each is sign-extended by shifting its byte to the top of the word,
//      reading the word as i32 and shifting right by 24, which keeps the
//      sign; converted to f32 and times d, it gives weight 4v .. 4v + 3, in
//      vec4 v. One multiply a weight, exact.
//
// Design and cost
//   - A group, counted: nine words read, eight of codes and one holding its
//     scale and a neighbour's; 8 vec4 multiplies.
//   - Gemma 3 1B's pinned file, counted: one Q8_0 tensor, the token
//     embedding, 1,152 × 262,144; 9,437,184 blocks, 320,864,256 bytes. The
//     gather reads one row of it a token (kernels/gather/gather.h), and the
//     head, which the file ties to it, reads all of it in each step that
//     asks for logits (graph/graph.h).
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.101 Use unsigned types for bit manipulation: codes are moved within
//            u32 words by unsigned shifts; a word is read as i32 only for
//            the shift that extends its sign.
//     Con.5  constexpr for values computable at compile time: kQ8_0.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path:
//            a block's codes decode in the kernel's registers, the scale
//            read from its own stream.
inline constexpr Format kQ8_0{kQ8_0Layout, UnpackWgsl{shaders::q8_0}};

}  // namespace frizzle::formats
