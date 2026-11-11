#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// UTF-8 as the tokenizer reads and writes it: what a lead byte says, decoding
// one character, and encoding one.
//
// Contract
//   - utf8_lead(byte): for a byte that begins a well-formed UTF-8 character,
//     its length, 1 to 4, and the range its second byte must lie in; length 0
//     for any other byte. The ranges are the Unicode Standard's Table 3-7,
//     well-formed byte sequences; every byte after the second lies in
//     0x80..0xBF. constexpr, so the static_assert below checks it at compile
//     time.
//   - decode_utf8(text, at, out): true with the character starting at
//     text[at] in `out`, its code point and its length; false for bytes that
//     are not well-formed there (a stray continuation byte, a truncated
//     sequence, an overlong encoding, a surrogate, a value past U+10FFFF),
//     `out` then untouched. Never reads outside `text`. Precondition:
//     at < text.size().
//   - append_utf8(code_point, out): appends its 1 to 4 bytes. Precondition: a
//     Unicode scalar value, at most U+10FFFF and not a surrogate; nothing
//     checks it.
//   Pure: no state, no allocation but `out`'s growth, no locale.
//
// Algorithm
//   decode_utf8: a byte below 0x80 is its own character. Otherwise
//   utf8_lead gives the length n and the second byte's range; fewer than n
//   bytes left, or a lead of length 0, is false. The lead's low 7 − n bits
//   begin the value; each later byte, checked against its range, adds its
//   low 6 bits.
//
// Who uses it: SentencePiece BPE steps through text a character at a time
// (bpe/sentencepiece_bpe.cpp); Utf8Stream joins token bytes into whole
// characters, writing U+FFFD for bytes that cannot be one (utf8_stream.cpp);
// the tokenizer suite checks decoded text is UTF-8.
//
// Design and cost
//   - The narrowed second-byte range is what rejects overlong forms,
//     surrogates and values past U+10FFFF, so no check follows the decode.
//   - Counted: decode_utf8 reads at most 4 bytes and makes at most 4
//     comparisons of range; append_utf8 writes at most 4 bytes.
//
// Verification
//   decode_utf8 runs on every reference text the tokenizer suite encodes,
//   all well-formed; no test feeds it ill-formed bytes or checks
//   append_utf8 alone.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.io.2 When reading, always consider ill-formed input: every
//             ill-formed sequence is refused at the byte it is found.
//     F.4     If a function might have to be evaluated at compile time,
//             declare it constexpr: utf8_lead, checked by static_assert.
//     I.5     State preconditions: decode_utf8's position, append_utf8's
//             scalar value.
//     ES.45   Avoid "magic constants"; use symbolic constants: Table 3-7's
//             byte values are named once each, by the role they play.

// One character decoded from UTF-8.
struct Utf8Char {
  char32_t code_point;
  std::uint8_t length;  // bytes it took, 1 to 4
};

// The byte values of well-formed UTF-8, the Unicode Standard's Table 3-7.
namespace utf8 {

inline constexpr unsigned char kAsciiEnd = 0x80;  // a byte below is a character
// Every byte after a lead, but the second where a lead narrows it (below).
inline constexpr unsigned char kContinuationLow = 0x80;
inline constexpr unsigned char kContinuationHigh = 0xBF;
// Leads. 0xC0 and 0xC1 would begin only overlong forms, 0xF5 and above only
// values past U+10FFFF.
inline constexpr unsigned char kLead2First = 0xC2;
inline constexpr unsigned char kLead2Last = 0xDF;
inline constexpr unsigned char kLead3First = 0xE0;
inline constexpr unsigned char kSurrogateLead = 0xED;
inline constexpr unsigned char kLead3Last = 0xEF;
inline constexpr unsigned char kLead4First = 0xF0;
inline constexpr unsigned char kLead4Last = 0xF4;
// The narrowed second bytes: below each low, or above each high, is an
// overlong form, a surrogate, or a value past U+10FFFF.
inline constexpr unsigned char kAfterLead3FirstLow = 0xA0;
inline constexpr unsigned char kAfterSurrogateLeadHigh = 0x9F;
inline constexpr unsigned char kAfterLead4FirstLow = 0x90;
inline constexpr unsigned char kAfterLead4LastHigh = 0x8F;

}  // namespace utf8

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
  if (byte < utf8::kAsciiEnd) return {1, 0, 0};
  if (byte >= utf8::kLead2First && byte <= utf8::kLead2Last) {
    return {2, utf8::kContinuationLow, utf8::kContinuationHigh};
  }
  if (byte == utf8::kLead3First) {
    return {3, utf8::kAfterLead3FirstLow, utf8::kContinuationHigh};
  }
  if (byte == utf8::kSurrogateLead) {
    return {3, utf8::kContinuationLow, utf8::kAfterSurrogateLeadHigh};
  }
  if (byte > utf8::kLead3First && byte <= utf8::kLead3Last) {
    return {3, utf8::kContinuationLow, utf8::kContinuationHigh};
  }
  if (byte == utf8::kLead4First) {
    return {4, utf8::kAfterLead4FirstLow, utf8::kContinuationHigh};
  }
  if (byte == utf8::kLead4Last) {
    return {4, utf8::kContinuationLow, utf8::kAfterLead4LastHigh};
  }
  if (byte > utf8::kLead4First && byte < utf8::kLead4Last) {
    return {4, utf8::kContinuationLow, utf8::kContinuationHigh};
  }
  return {0, 0, 0};  // a continuation byte, an overlong lead, or past U+10FFFF
}

static_assert(utf8_lead(utf8::kContinuationLow).length == 0 &&
              utf8_lead(utf8::kLead2First - 1).length == 0 &&
              utf8_lead(utf8::kLead4Last + 1).length == 0);

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
