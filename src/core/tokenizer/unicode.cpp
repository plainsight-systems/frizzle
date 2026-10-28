#include "core/tokenizer/unicode.h"

#include <algorithm>

#include "core/tokenizer/unicode_tables.h"

namespace frizzle::tokenizer {

CharClass char_class(char32_t code_point) noexcept {
  const auto table = unicode_tables::classes();
  // The first range that ends at or after the code point; it holds the code
  // point if it also starts at or before it.
  const auto range = std::lower_bound(table.begin(), table.end(), code_point,
                                      [](const unicode_tables::ClassRange& r,
                                         char32_t c) { return r.last < c; });
  return range != table.end() && range->first <= code_point ? range->char_class
                                                            : CharClass::Other;
}

bool decode_utf8(std::string_view text, std::size_t at,
                 Utf8Char& out) noexcept {
  const auto byte = [&](std::size_t i) {
    return static_cast<unsigned char>(text[i]);
  };
  const unsigned char first = byte(at);
  if (first < 0x80) {
    out = {first, 1};
    return true;
  }
  const Utf8Lead lead = utf8_lead(first);
  if (lead.length == 0 || text.size() - at < lead.length) return false;
  // The lead byte's value bits: 5 of a two-byte character's, 4 of three, 3 of
  // four.
  char32_t value = first & (0x7Fu >> lead.length);
  for (std::uint8_t i = 1; i < lead.length; ++i) {
    const unsigned char next = byte(at + i);
    const unsigned char low = i == 1 ? lead.second_low : 0x80;
    const unsigned char high = i == 1 ? lead.second_high : 0xBF;
    if (next < low || next > high) return false;
    value = (value << 6) | (next & 0x3Fu);
  }
  out = {value, lead.length};
  return true;
}

void append_utf8(char32_t code_point, std::string& out) {
  const auto put = [&](char32_t bits) {
    out.push_back(static_cast<char>(bits));
  };
  if (code_point < 0x80) {
    put(code_point);
  } else if (code_point < 0x800) {
    put(0xC0 | (code_point >> 6));
    put(0x80 | (code_point & 0x3F));
  } else if (code_point < 0x10000) {
    put(0xE0 | (code_point >> 12));
    put(0x80 | ((code_point >> 6) & 0x3F));
    put(0x80 | (code_point & 0x3F));
  } else {
    put(0xF0 | (code_point >> 18));
    put(0x80 | ((code_point >> 12) & 0x3F));
    put(0x80 | ((code_point >> 6) & 0x3F));
    put(0x80 | (code_point & 0x3F));
  }
}

}  // namespace frizzle::tokenizer
