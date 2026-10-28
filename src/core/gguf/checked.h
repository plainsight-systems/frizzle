#pragma once

#include <cstdint>
#include <limits>

// Checked arithmetic for offsets, lengths and shape products.
//
// The file arrives from a CDN, so every value in it is hostile until proven
// otherwise. `offset + length` on attacker-chosen values wraps, and a wrapped
// result compares as in-bounds — which is exactly how a bounds check becomes
// decorative (ES.103, SL.con.3).
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
