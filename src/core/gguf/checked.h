#pragma once

#include <cstdint>
#include <limits>

// Axis C: changes with the GGUF format.
//
// Checked arithmetic for offsets, lengths and shape products read from a
// file. The file arrives from a download, so every value in it is hostile
// until checked. `offset + length` on chosen values wraps, and a wrapped
// result compares as in bounds: the bounds check would pass a range that is
// not inside the file. The reader (reader.h), the array reads (arrays.h) and
// residency's plan, routes and piece writer use these.
//
// Contract
//   - checked_add(a, b, out) and checked_mul(a, b, out): true with `out` set
//     to the exact 64-bit result; false, with `out` untouched, when the
//     result does not fit in 64 bits.
//   - range_within(offset, length, total): true exactly when [offset,
//     offset + length) lies within [0, total), for any three values; an empty
//     range at offset == total is within.
//   - All three are constexpr and noexcept, and allocate nothing.
//
// Algorithm
//   1. checked_add: overflow exactly when a > max - b; else out = a + b.
//   2. checked_mul: overflow exactly when a != 0 and b > max / a; else
//      out = a * b. A zero factor never overflows.
//   3. range_within: offset <= total, then length <= total - offset. The
//      subtraction runs only once offset <= total, so it cannot wrap, and
//      no sum is formed.
//
// Design and cost
//   - Each check is a comparison and a subtraction or a division before the
//     operation itself, inline: no allocation, no branch beyond the check.
//   - Unsigned 64-bit throughout: file offsets and lengths are uint64 in the
//     format, and the checks rule out wrap rather than relying on it.
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.103   Don't overflow: every sum and product of file values is
//              checked before it is used, and ranges are tested by
//              subtraction.
//     SL.con.3 Avoid bounds errors: range_within is the test a read of the
//              file passes first.
//     F.6      Declare noexcept what must not throw: all three only compare
//              and compute.
namespace frizzle::gguf {

// a + b, false on overflow.
[[nodiscard]] constexpr bool checked_add(std::uint64_t a, std::uint64_t b,
                                         std::uint64_t& out) noexcept {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    return false;
  }
  out = a + b;
  return true;
}

// a * b, false on overflow.
[[nodiscard]] constexpr bool checked_mul(std::uint64_t a, std::uint64_t b,
                                         std::uint64_t& out) noexcept {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

// True when [offset, offset + length) lies wholly within `total`, with the
// addition itself checked. Written as a subtraction so it is correct even if
// the caller passes values near the top of the range.
[[nodiscard]] constexpr bool range_within(std::uint64_t offset,
                                          std::uint64_t length,
                                          std::uint64_t total) noexcept {
  return offset <= total && length <= total - offset;
}

}  // namespace frizzle::gguf
