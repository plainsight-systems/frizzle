// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Contract, algorithm, cost and corpus guidance: unicode.h. This file defines
// what that header states.

#include "core/tokenizer/unicode.h"

namespace frizzle::tokenizer {
namespace {

// A continuation byte is 10xxxxxx: the mark, then six bits of the value.
constexpr unsigned char kContinuationMark = 0x80;
constexpr unsigned kContinuationBits = 6;
constexpr char32_t kContinuationMask = 0x3F;
// A lead of n bytes keeps the low 7 - n bits of this mask as value bits.
constexpr char32_t kLeadValueMask = 0x7F;
// Each length's lead mark, and the first code point that needs the length.
constexpr char32_t kLead2Mark = 0xC0;
constexpr char32_t kLead3Mark = 0xE0;
constexpr char32_t kLead4Mark = 0xF0;
constexpr char32_t kTwoBytesFrom = 0x80;
constexpr char32_t kThreeBytesFrom = 0x800;
constexpr char32_t kFourBytesFrom = 0x10000;

}  // namespace

bool decode_utf8(std::string_view text, std::size_t at,
                 Utf8Char& out) noexcept {
  const auto byte = [&](std::size_t i) {
    return static_cast<unsigned char>(text[i]);
  };
  const unsigned char first = byte(at);
  if (first < utf8::kAsciiEnd) {
    out = {first, 1};
    return true;
  }
  const Utf8Lead lead = utf8_lead(first);
  if (lead.length == 0 || text.size() - at < lead.length) return false;
  // The lead byte's value bits: 5 of a two-byte character's, 4 of three, 3 of
  // four.
  char32_t value = first & (kLeadValueMask >> lead.length);
  for (std::uint8_t i = 1; i < lead.length; ++i) {
    const unsigned char next = byte(at + i);
    const unsigned char low = i == 1 ? lead.second_low : utf8::kContinuationLow;
    const unsigned char high =
        i == 1 ? lead.second_high : utf8::kContinuationHigh;
    if (next < low || next > high) return false;
    value = (value << kContinuationBits) | (next & kContinuationMask);
  }
  out = {value, lead.length};
  return true;
}

void append_utf8(char32_t code_point, std::string& out) {
  const auto put = [&](char32_t bits) {
    out.push_back(static_cast<char>(bits));
  };
  // The continuation byte that has `later` more after it.
  const auto continuation = [&](unsigned later) {
    put(kContinuationMark |
        ((code_point >> (later * kContinuationBits)) & kContinuationMask));
  };
  if (code_point < kTwoBytesFrom) {
    put(code_point);
  } else if (code_point < kThreeBytesFrom) {
    put(kLead2Mark | (code_point >> kContinuationBits));
    continuation(0);
  } else if (code_point < kFourBytesFrom) {
    put(kLead3Mark | (code_point >> (2 * kContinuationBits)));
    continuation(1);
    continuation(0);
  } else {
    put(kLead4Mark | (code_point >> (3 * kContinuationBits)));
    continuation(2);
    continuation(1);
    continuation(0);
  }
}

}  // namespace frizzle::tokenizer
