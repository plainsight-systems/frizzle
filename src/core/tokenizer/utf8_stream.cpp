#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/unicode.h"

namespace frizzle::tokenizer {
namespace {

constexpr char32_t kReplacement = 0xFFFD;

}  // namespace

// Bytes that cannot be a character are replaced one maximal subpart at a time,
// as the Unicode Standard recommends (section 3.9) and as Rust's
// from_utf8_lossy, Python's "replace" and the browser's TextDecoder do: the
// bytes of a character begun well but broken off become one U+FFFD, and any
// byte that cannot begin a character becomes one of its own. Whichever way the
// bytes are split between pushes, the text is the same.
void Utf8Stream::push(std::string_view token_bytes, std::string& out) {
  for (const char c : token_bytes) {
    const auto byte = static_cast<unsigned char>(c);
    if (pending_count_ > 0) {
      const Utf8Lead lead = utf8_lead(static_cast<unsigned char>(pending_[0]));
      const unsigned char low = pending_count_ == 1 ? lead.second_low : 0x80;
      const unsigned char high = pending_count_ == 1 ? lead.second_high : 0xBF;
      if (byte >= low && byte <= high) {
        if (pending_count_ + 1 == lead.length) {
          out.append(pending_.data(), pending_count_);
          out.push_back(c);
          pending_count_ = 0;
        } else {
          pending_[pending_count_++] = c;
        }
        continue;
      }
      // The character cannot be completed. What was held is replaced,
      // and this byte is read afresh below.
      append_utf8(kReplacement, out);
      pending_count_ = 0;
    }
    const Utf8Lead lead = utf8_lead(byte);
    if (lead.length == 1) {
      out.push_back(c);
    } else if (lead.length == 0) {
      append_utf8(kReplacement, out);
    } else {
      pending_[0] = c;
      pending_count_ = 1;
    }
  }
}

bool Utf8Stream::finish(std::string& out) {
  if (pending_count_ == 0) return true;
  append_utf8(kReplacement, out);
  pending_count_ = 0;
  return false;
}

}  // namespace frizzle::tokenizer
