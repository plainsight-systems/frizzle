// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Utf8Stream (tokenizer.h): the bytes of successive tokens as whole UTF-8
// characters. A token can end partway through a character; its bytes are
// held until the next token completes them.
//
// Contract
//   - push(token_bytes, out): appends to `out` every character the bytes
//     complete, in order. Bytes that cannot be a character are replaced one
//     maximal subpart at a time, the Unicode Standard's practice: the bytes
//     of a character begun well but broken off become one U+FFFD, and a byte
//     that cannot begin a character becomes one of its own. However the
//     bytes are split between pushes, the text is the same.
//   - finish(out): false, with one U+FFFD appended, when bytes were pending,
//     the stream stopped inside a character; true otherwise. The stream is
//     then empty, ready for another.
//   - Invariant: at most 3 bytes are held, the start of a character of up to
//     4, each well-formed so far (unicode.h's ranges).
//
// Algorithm, push, for each byte b
//   1. Bytes held: b continues the character when it lies in the range the
//      held lead allows at that place (the narrowed second-byte range, then
//      0x80..0xBF). It completes the character, which is appended with the
//      held bytes, or is held. Otherwise the held bytes become one U+FFFD and
//      b is read afresh in step 2.
//   2. Nothing held: b below 0x80 is appended; a lead of 2 to 4 bytes is
//      held; any other byte becomes U+FFFD.
//
// Design and cost
//   - Counted: a push reads each byte once and writes at most 3 bytes for
//     each, U+FFFD's three, plus one U+FFFD for the bytes held before them;
//     it allocates only `out`'s growth, which the Detokenizer reserves
//     (detokenizer.h). The held bytes live in the stream, 3 bytes and a
//     count.
//
// Verification
//   The tokenizer suite decodes every reference case a token at a time
//   through a stream and checks the text whole and well-formed. No case has
//   ill-formed bytes or ends inside a character, so no test reaches U+FFFD.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.io.2  When reading, always consider ill-formed input: a token's
//              bytes are checked one at a time against the ranges of
//              well-formed UTF-8, and ill-formed ones are replaced, never
//              passed on.
//     SL.con.3 Avoid bounds errors: the held bytes are written through a
//              span at an index below 3, as a 4-byte character completes
//              on its fourth byte.
//   Searched, and no rule applies: decoding ill-formed UTF-8 with
//   replacement characters, and holding a partial character across chunks
//   of a stream (the rules found are for naming, input and streaming
//   summaries).

#include <span>

#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/unicode.h"

namespace frizzle::tokenizer {
namespace {

constexpr char32_t kReplacement = 0xFFFD;

}  // namespace

// Replaces ill-formed bytes one maximal subpart at a time (above).
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
          const std::span pending(pending_);
          pending[pending_count_++] = c;
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
