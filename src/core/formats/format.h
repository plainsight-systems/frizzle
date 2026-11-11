#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "core/formats/device_layout.h"
#include "core/gguf/types.h"

namespace frizzle::formats {

// Axis B: changes with what every weight format supplies: its unpack, its
// pack and its device layout.
//
// What a weight format is to the rest of Frizzle. Each formats/<format>/
// defines one Format, and the capability table lists it under the GGUF type
// it implements (capability/capability.h). A format is the only code that
// knows what its blocks mean. It supplies:
//   - unpack, as WGSL a kernel composes with. No kernel knows a format;
//     adding one adds a directory here and a row in the capability table,
//     and touches no kernel.
//   - pack, as WGSL, for a format the KV cache stores in. Packing and
//     unpacking are one piece of knowledge, whether the data is a weight or
//     a cached key.
//   - its device layout: how its blocks' fields lie on the device, as
//     streams (device_layout.h). Upload writes a weight that way; unpack
//     reads it that way.
//
// Contract
//   - A Format exists only at compile time: its constructor is consteval,
//     and each formats/<format>/ defines one as a constexpr constant. It
//     takes its type from its layout, so the two cannot name different
//     types, and it holds its layout from construction. The constructor
//     refuses, at compile time, a layout that breaks the promise
//     device_layout.h states, and an empty unpack: a format with nothing to
//     read its weights is not one this build runs. The capability table
//     checks at compile time that every row's type is its Format's type. So
//     a format the table lists has a layout that matches it and keeps its
//     promise, and upload and the kernels cannot disagree on how it lies.
//   - Unpack and pack come as UnpackWgsl and PackWgsl, two types, so a
//     format's definition cannot swap them. The WGSL they view and the
//     layout are static storage; a Format owns nothing and holds no other
//     state.
//   - Unpack is one WGSL function, the same for every format:
//
//       // declared by the kernel, under this name, at the binding
//       // kernels/interface.h gives the weight a launch reads:
//       @group(0) @binding(2) var<storage, read> weights: array<u32>;
//       // supplied by the format, in formats/<format>/<format>.wgsl:
//       fn unpack(base: u32, blocks_in_piece: u32, group: u32)
//           -> array<vec4<f32>, 8>
//
//     It returns the 32 consecutive weights of `group`, in order, as eight
//     vec4s: a whole block of Q4_0, Q4_1 or Q8_0, or 32 values of F32 or
//     F16. `blocks_in_piece` locates each stream: a stream starts after
//     every earlier stream's fields for the whole piece (device_layout.h).
//     `base` is the piece's first word within the binding: 0 where the
//     binding is the piece, and a member's own first word where one binding
//     spans a fused group's members (kernels/matmul/matmul.h). Kernels pass
//     both as uniforms. Precondition: `group` lies within the piece; unpack
//     checks nothing.
//   - A kernel asks only for groups within a row, so a row must be a whole
//     number of groups: steps_by_groups says whether a tensor's are. A
//     tensor whose rows are not (possible only for F32 or F16, whose block
//     is one value) and that is not a scalar is refused by routes
//     (RowNotSteppable, residency/routes.h), so upload fails, naming it,
//     and no kernel is handed one. A scalar, one value, passes: no kernel
//     steps through one. Gemma 3 1B's file holds none, counted.
//   - row_elements(tensor) is the values in one row: every dimension but the
//     last, so a matrix of K × N is N rows of K, as GGUF stores it, and a
//     vector is one row. For every one- and two-dimensional tensor this is
//     the first dimension; a last dimension of 0 gives rows of 0.
//     Precondition: dimension_count is at most gguf::kMaxDimensions, as in
//     every entry the reader made.
//   - Pack is one WGSL function, for a format the KV cache stores in, with
//     its inverse beside it. Neither reads or writes a binding, so a kernel
//     composes them whatever it binds, and writes the words pack returns
//     itself (kernels/rope/rope.h):
//
//       // supplied by the format, in formats/<format>/<format>_pack.wgsl:
//       fn pack(values: vec4<f32>) -> vec2<u32>
//       fn unpack4(words: vec2<u32>) -> vec4<f32>
//
//     pack returns the two words that store four consecutive values, so a
//     cache format stores each value in 16 bits, on its own; unpack4 returns
//     the four values two words store, and attention reads the cache with
//     it (kernels/attention/attention.h). F16 is the one cache format
//     (formats/f16/f16.h). A block-scaled format cannot pack four values on
//     their own: Q8_0's scale is set by its 32 values' largest magnitude, a
//     reduction across invocations, so Q8_0 has no pack. A format without
//     one has an empty pack_wgsl, and the program refuses to build a launch
//     whose pack format has none, naming the launch (kernels/program.cpp),
//     rather than run it.
//   - Numbers, derived, not tested. Unpack decodes each format by its
//     definition, in f32 (each <format>.h states it). F32 and F16 values
//     come back exact, and so do Q4_0's and Q8_0's weights: an f16 scale's
//     11 significant bits times a code of at most 8 bits needs at most 19,
//     within f32's 24. Q4_1's weight q × d + m rounds once, at the add:
//     q × d is exact, so the result is the same whether the target fuses
//     the multiply-add, which WGSL allows, or not.
//   - WGSL allows two departures. A zero may lose its sign, which a dot
//     product cannot tell. unpack2x16float may flush a subnormal f16 to
//     zero, which would change the weights of a block whose scale or
//     minimum is subnormal; Gemma 3 1B's pinned file has no subnormal, zero,
//     infinite or NaN scale or minimum in its 31,242,240 blocks, counted.
//     NaN and infinity are not inputs.
//   - Pack's rounding point (formats/f16/f16_pack.wgsl): each value is
//     saturated to F16's largest finite, ±65,504, then rounded once to F16
//     by pack2x16float. WGSL leaves converting a value outside F16's range
//     indeterminate, so the saturation makes it the nearest finite value.
//     WGSL lets the conversion round to either neighbour; Frizzle's accuracy
//     statements assume round to nearest on the target (kernels/rope/
//     rope.h). No test checks which.
//
// Algorithm
//   1. row_elements: with fewer than two dimensions, the first. Otherwise L,
//      the last dimension in use, dimensions[dimension_count − 1]; the
//      tensor's element count / L, or 0 when L is 0.
//   2. steps_by_groups: row_elements is a multiple of kUnpackGroup, 32.
//   3. is_scalar: the element count is 1.
//   4. Format's constructor, at compile time: keeps_its_promise(layout)
//      (device_layout.h), and an unpack that is not empty. A check that
//      fails calls a function declared here and never defined, which is not
//      constexpr, so the definition does not compile.
//   5. On the GPU, a kernel's unpack call decodes one group from its words
//      to its 32 weights, as its format's <format>.h states.
//
// Design and cost
//   - A Format is constant data: no work at startup and no allocation. The
//     WGSL it views is the generated shaders' static strings.
//   - A format's WGSL is joined into the shader source of each launch that
//     reads its weights, or packs into its cache, at load
//     (kernels/program.cpp). Unpack is a function in the kernel's own
//     shader: no dispatch, buffer or copy of its own. Weights decode in the
//     kernel's load path, into registers, and are never written back out
//     expanded.
//   - Optimization (practice): the 32 weights come as eight vec4s, so a
//     kernel takes its dot products four values at a time.
//   - Optimization (browser): half-precision scales and values are read
//     from u32 words with unpack2x16float, so no format needs the
//     shader-f16 feature, which is optional in WebGPU and not every browser
//     offers; the device is requested with no optional feature
//     (gpu/device_requirements.h).
//   - Whether adjacent lanes read adjacent addresses depends on how a kernel
//     maps lanes to groups; the layout makes it possible, and each kernel
//     states its own mapping.
//   - Bits a weight, by format: F32 32, F16 16, Q8_0 8.5, Q4_1 5, Q4_0 4.5.
//     Gemma 3 1B's pinned file, counted: 340 tensors, 183 matrices and 157
//     vectors, none with more dimensions and none a scalar; rows of 256,
//     1,024, 1,152 and 6,912 values, each a whole number of groups, so
//     routes refuses none.
//   - No test checks an unpack or a pack alone; the oracle's logits cover
//     them end to end (tests/README.md), as they cover the layouts.
//   - Block sizes belong to the file format and are read from gguf/types.h;
//     a format does not restate them. There is no CPU dequantizer: nothing
//     in Frizzle materialises a dequantized weight.
//
// Corpus guidance
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking to run-time checking: a wrong
//            pairing of type and layout, or an empty unpack, does not
//            compile.
//     C.41   A constructor should create a fully initialized object: there
//            is no Format without a layout and an unpack.
//     I.24   Avoid adjacent parameters that can be swapped: unpack and
//            pack, both WGSL source, take two struct types, as the rule's
//            second alternative names its fields.
//     SL.con.3 Avoid bounds errors: row_elements reads the last dimension in
//            use as the end of a span over the dimensions in use, not by a
//            subscript; the span checks nothing, so the bound is the
//            precondition on dimension_count.
//     Con.5  constexpr for values computable at compile time: kUnpackGroup
//            and the three helpers.
//     F.6    noexcept where a function must not throw: the helpers and the
//            accessors only read.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path:
//            unpack decodes into registers; nothing expands a weight in
//            memory.
//     GPU.2  Shape data for coalesced lane access: the layout makes it
//            possible; each kernel's lane mapping claims it or not.
//     EMB.6  Push computation to compile time, consteval where it must be:
//            the constructor is consteval, so a Format built at run time
//            does not compile.
//   Searched, and no rule applies: the rounding of a single fused
//   multiply-add (the rules found are for reductions); a WGSL function a
//   kernel composes, with no dispatch of its own (the rules found are for
//   batching dispatches); subnormal values flushed to zero, on both servers
//   (the rules found are for overflow, underflow and bounds); wrapper types
//   for parameters of one type (none beyond I.24); loading four values at a
//   time as a vec4 (the rules found are for CPU SIMD layout).

// The weights unpack returns at a time: every kernel's step along a row.
inline constexpr std::uint64_t kUnpackGroup = 32;

// The values in one of `tensor`'s rows (Contract).
[[nodiscard]] constexpr std::uint64_t row_elements(
    const gguf::TensorEntry& tensor) noexcept {
  if (tensor.dimension_count < 2) return tensor.dimensions[0];
  const std::uint64_t last =
      std::span(tensor.dimensions).first(tensor.dimension_count).back();
  return last == 0 ? 0 : tensor.element_count / last;
}

// Whether a kernel can step through `tensor`'s rows, a group at a time.
[[nodiscard]] constexpr bool steps_by_groups(
    const gguf::TensorEntry& tensor) noexcept {
  return row_elements(tensor) % kUnpackGroup == 0;
}

// Whether a tensor is one value, which no kernel steps through (Contract).
[[nodiscard]] constexpr bool is_scalar(
    const gguf::TensorEntry& tensor) noexcept {
  return tensor.element_count == 1;
}

namespace detail {
// Not constexpr: reached only when a Format would be built wrong, which makes
// its construction fail to compile.
void layout_breaks_its_promise();
void format_has_no_unpack();
}  // namespace detail

// A format's unpack, as WGSL source.
struct UnpackWgsl {
  std::string_view source;
};

// A format's pack and its inverse, as WGSL source; empty for a format the
// cache never stores in.
struct PackWgsl {
  std::string_view source;
};

class Format {
 public:
  consteval Format(const DeviceLayout& layout, UnpackWgsl unpack,
                   PackWgsl pack = {})
      : layout_(&layout), unpack_wgsl_(unpack.source), pack_wgsl_(pack.source) {
    if (!keeps_its_promise(layout)) detail::layout_breaks_its_promise();
    if (unpack_wgsl_.empty()) detail::format_has_no_unpack();
  }

  [[nodiscard]] constexpr gguf::TensorType type() const noexcept {
    return layout_->type;
  }
  [[nodiscard]] constexpr const DeviceLayout& layout() const noexcept {
    return *layout_;
  }
  [[nodiscard]] constexpr std::string_view unpack_wgsl() const noexcept {
    return unpack_wgsl_;
  }
  [[nodiscard]] constexpr std::string_view pack_wgsl() const noexcept {
    return pack_wgsl_;
  }

 private:
  const DeviceLayout* layout_;  // never null: the constructor takes a reference
  std::string_view unpack_wgsl_;
  std::string_view pack_wgsl_;
};

}  // namespace frizzle::formats
