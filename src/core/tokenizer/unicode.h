#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace frizzle::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Unicode as tokenization needs it: UTF-8 decoding, and the character classes
// the split patterns test. The classes come from tables generated from one
// version of the Unicode Character Database (unicode_tables.h) and compiled
// in, so every browser classifies the same text the same way.

// A code point's class as the split patterns test it. A code point is in at
// most one of these: no letter or number is white space.
enum class CharClass : std::uint8_t {
  Other,
  Letter,      // \p{L}: general category L*
  Number,      // \p{N}: general category N*
  Whitespace,  // \s: the White_Space property
};

[[nodiscard]] CharClass char_class(char32_t code_point) noexcept;

// One character decoded from UTF-8.
struct Utf8Char {
  char32_t code_point;
  std::uint8_t length;  // bytes it took, 1 to 4
};

// What a byte says when it begins a UTF-8 character, by the well-formed byte
// sequences of the Unicode Standard, Table 3-7: how many bytes the character
// takes, and the range its second byte must lie in. Every later byte lies in
// 0x80..0xBF. The narrowed second range is what rejects overlong encodings,
// surrogates and values past U+10FFFF.
struct Utf8Lead {
  std::uint8_t length;       // 0: no character begins with this byte
  unsigned char second_low;  // meaningful when length is 2 or more
  unsigned char second_high;
};

[[nodiscard]] constexpr Utf8Lead utf8_lead(unsigned char byte) noexcept {
  if (byte < 0x80) return {1, 0, 0};
  if (byte >= 0xC2 && byte <= 0xDF) return {2, 0x80, 0xBF};
  if (byte == 0xE0) return {3, 0xA0, 0xBF};  // below is overlong
  if (byte == 0xED) return {3, 0x80, 0x9F};  // above is a surrogate
  if (byte >= 0xE1 && byte <= 0xEF) return {3, 0x80, 0xBF};
  if (byte == 0xF0) return {4, 0x90, 0xBF};  // below is overlong
  if (byte == 0xF4) return {4, 0x80, 0x8F};  // above is past U+10FFFF
  if (byte >= 0xF1 && byte <= 0xF3) return {4, 0x80, 0xBF};
  return {0, 0, 0};  // a continuation byte, an overlong lead, or past U+10FFFF
}

static_assert(utf8_lead(0x80).length == 0 && utf8_lead(0xC1).length == 0 &&
              utf8_lead(0xF5).length == 0);

// Decodes the character that starts at text[at]. False when the bytes there
// are not well-formed UTF-8: a stray continuation byte, a truncated sequence,
// an overlong encoding, a surrogate, or a value past U+10FFFF.
// Precondition: at < text.size().
[[nodiscard]] bool decode_utf8(std::string_view text, std::size_t at,
                               Utf8Char& out) noexcept;

// Appends the UTF-8 encoding of `code_point`. Precondition: it is a Unicode
// scalar value (at most U+10FFFF, not a surrogate).
void append_utf8(char32_t code_point, std::string& out);

}  // namespace frizzle::tokenizer
