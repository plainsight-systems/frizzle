#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/gguf/types.h"

namespace frizzle::formats {

// Axis B: changes with a weight format's layout on the device.
//
// How a format's blocks lie on the device. A file stores each block whole: a
// Q4_0 block is a 2-byte scale and 16 bytes of nibbles, 18 bytes in all. On
// the device a piece of a weight holds its blocks' fields as streams instead:
// every block's nibbles, in block order, then every block's scales. Upload
// writes a piece that way (residency/piece_writer.h) and a format's unpack
// reads it that way.
//
// Contract
//   - A DeviceLayout names its tensor type, its stored block size and its
//     streams in device order. Each stream is one field of the stored block,
//     copied as it is: its offset in the block and its width.
//   - A layout's streams cover its block exactly, each byte in one stream,
//     and every stream but the last is a multiple of 4 wide. So a piece of n
//     blocks takes exactly its stored bytes rounded up to 4, the length the
//     residency plan binds it at (residency/plan.h), whatever n is, and every
//     stream starts on a word.
//   - Every layout keeps kMaxBlockBytes and kMaxStreams, the bounds the piece
//     writer sizes what it holds between chunks by.
//   - keeps_its_promise(layout) checks all of this, and that block_bytes is
//     the GGUF type's block size (gguf/types.h), at compile time: every
//     layout here is checked by the static_assert below.
//   - A layout is part of its format: each Format names its own (format.h),
//     and the capability table, which lists the formats this build runs, is
//     the one place a layout is found. This file holds the layouts' shape,
//     the layouts of the formats listed, and the check; it is not a second
//     table of what is supported.
//
// Algorithm, piece of n blocks laid out (residency/piece_writer.h)
//   For stream s of width w_s at offset o_s, in device order: bytes
//   [Σ_{t<s} w_t × n, Σ_{t≤s} w_t × n) of the piece hold, for block i in
//   order, bytes [o_s, o_s + w_s) of stored block i. The piece's bound length
//   is n × block_bytes rounded up to 4; the bytes past the last stream are
//   zero.
//
// Design and cost
//   - Optimization: WGSL reads a storage buffer in aligned 32-bit words. An
//     18-byte block straddles them, so reading one stored whole takes two
//     loads and a shift for every field that crosses a word; as streams,
//     every field of every block starts on the alignment its width needs,
//     and adjacent invocations reading adjacent blocks touch adjacent
//     addresses. The difference this makes to a kernel's speed is not
//     measured in Frizzle.
//   - The transform is done once, at upload, a chunk at a time, never per
//     token; it moves no bits within a field, so the codes stay as the file
//     holds them and each format's unpack decodes them as it loads.
//   - No transformed copy is kept: the bytes live in device memory for one
//     page, so the build that writes a layout is the build that reads it, and
//     a layout needs no version. The layout is a contract with one consumer,
//     its format's unpack. No test checks an unpack alone; the oracle's
//     logits cover them end to end (tests/README.md).
//   - Gemma 3 1B's pinned file, counted: 179 Q4_0 tensors, 3 Q4_1 (the first
//     three ffn_down), 1 Q8_0 (the token embedding), 157 F32 (the norms); no
//     F16.
//
// Corpus guidance
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking: each layout's promise is a
//            static_assert, not a test.
//     Con.5  constexpr for values computable at compile time: the layouts are
//            constexpr constants, the check consteval.
//     SL.con.1 Prefer std::array to a C array: each layout's streams, and the
//            bytes the check has seen, which at() reads within the block.
//   C++ performance guidelines
//     GPU.2  Shape data for coalesced lane access: one field across many
//            blocks, adjacent for adjacent lanes, transposed at load.
//     CDSA.32 Transform static data once into the layout its consumer reads:
//            once, at upload; each layout a named constant checked against
//            the stored block. It asks for a version on a persisted copy;
//            none is kept.

// The bounds every layout here keeps, which the piece writer sizes what it
// holds between chunks by: one block, and a few bytes of each stream. Each is
// the largest among the layouts below: block_bytes of F32 4, F16 2, Q4_0 18,
// Q4_1 20 and Q8_0 34, so 34; streams of F32 and F16 one, Q4_0, Q4_1 and
// Q8_0 two, so 2. keeps_its_promise checks every layout against both.
inline constexpr std::size_t kMaxBlockBytes = 34;  // Q8_0's
inline constexpr std::size_t kMaxStreams = 2;      // Q4_0's, Q4_1's, Q8_0's

// One field of a block: where it lies in the stored block, and how wide it
// is.
struct Stream {
  std::uint16_t offset;
  std::uint16_t width;
};

struct DeviceLayout {
  gguf::TensorType type;
  std::uint32_t block_bytes;
  std::span<const Stream> streams;  // in device order
};

// The layouts of the formats the capability table lists: F32 for norms, Q4_0
// and Q4_1 for most weights, Q8_0 for the token embedding, and F16, the KV
// cache's format and that of weights a file stores at half precision. Each
// field's place is the GGUF block's: Q4_0 is d, an f16 scale, then 16 bytes
// of nibbles qs; Q4_1 is d and m, two f16, then 16 bytes of qs; Q8_0 is d,
// an f16, then 32 signed bytes qs. Each format's Format names its own.
namespace detail {
inline constexpr std::array<Stream, 1> kF32{{{0, 4}}};
inline constexpr std::array<Stream, 1> kF16{{{0, 2}}};
// qs, then d
inline constexpr std::array<Stream, 2> kQ4_0{{{2, 16}, {0, 2}}};
// qs, then d and m
inline constexpr std::array<Stream, 2> kQ4_1{{{4, 16}, {0, 4}}};
// qs, then d
inline constexpr std::array<Stream, 2> kQ8_0{{{2, 32}, {0, 2}}};
}  // namespace detail

inline constexpr DeviceLayout kF32Layout{gguf::TensorType::F32, 4,
                                         detail::kF32};
inline constexpr DeviceLayout kF16Layout{gguf::TensorType::F16, 2,
                                         detail::kF16};
inline constexpr DeviceLayout kQ4_0Layout{gguf::TensorType::Q4_0, 18,
                                          detail::kQ4_0};
inline constexpr DeviceLayout kQ4_1Layout{gguf::TensorType::Q4_1, 20,
                                          detail::kQ4_1};
inline constexpr DeviceLayout kQ8_0Layout{gguf::TensorType::Q8_0, 34,
                                          detail::kQ8_0};

// Whether `layout` keeps the promise above: its streams cover the block, its
// block size is the file format's, only its last stream may be unaligned, and
// it is within the bounds the piece writer holds.
[[nodiscard]] consteval bool keeps_its_promise(const DeviceLayout& layout) {
  if (layout.block_bytes > kMaxBlockBytes ||
      layout.streams.size() > kMaxStreams) {
    return false;
  }
  std::uint32_t covered = 0;
  // Which bytes of the block a stream has claimed: each below kMaxBlockBytes
  // once the test above has passed.
  std::array<bool, kMaxBlockBytes> seen{};
  for (std::size_t i = 0; i < layout.streams.size(); ++i) {
    const Stream s = layout.streams[i];
    if (i + 1 < layout.streams.size() && s.width % 4 != 0) return false;
    for (std::uint32_t b = s.offset; b < std::uint32_t{s.offset} + s.width;
         ++b) {
      if (b >= layout.block_bytes || seen.at(b)) return false;
      seen.at(b) = true;
    }
    covered += s.width;
  }
  // The file format's block size, looked up by value: GCC does not treat a
  // pointer comparison here as a constant expression under sanitizers.
  std::uint32_t stored_bytes = 0;
  for (const gguf::FormatLayout& stored : gguf::kFormatLayouts) {
    if (stored.type == layout.type) stored_bytes = stored.block_bytes;
  }
  return covered == layout.block_bytes && stored_bytes == layout.block_bytes;
}

static_assert(keeps_its_promise(kF32Layout) && keeps_its_promise(kF16Layout) &&
              keeps_its_promise(kQ4_0Layout) &&
              keeps_its_promise(kQ4_1Layout) && keeps_its_promise(kQ8_0Layout));

}  // namespace frizzle::formats
