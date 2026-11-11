#pragma once

#include <string>
#include <string_view>

#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// A reply's tokens as text: each token decoded (tokenizer.h) and pushed
// through a UTF-8 stream, giving the whole characters it completes. The
// generator (runtime/generator.h) makes one at load and passes what it
// gives to the page.
//
// Contract
//   - Detokenizer(tokenizer): borrows `tokenizer`, which outlives it (I.11),
//     and reserves its two buffers.
//   - push(token): the characters `token` completes, possibly none, for a
//     token whose bytes only begin a character another token finishes.
//     Bytes that cannot begin or continue a character become U+FFFD
//     (Utf8Stream, tokenizer.h). Precondition: the token is below the
//     vocabulary's size.
//   - finish(): ends the reply: U+FFFD where bytes were pending, rather than
//     dropping them unseen, else nothing. The stream is then empty, ready
//     for another reply.
//   - The views push and finish return are valid until the next call.
//   - longest(): the vocabulary's longest token text, in bytes, at least 1.
//   - No GPU and no callback: the same tokens, pushed in the same order from
//     an empty stream, give the same text.
//
// Algorithm
//   Construction: longest = the longest token text in the vocabulary, at
//   least 1; reserve longest bytes for a token's bytes and 3 × longest + 3
//   for the text they complete.
//   push: clear both buffers; decode the token's bytes into the first; push
//   them through the stream into the second; return it.
//   finish: clear the text; let the stream end into it; return it.
//
// Design and cost
//   - Counted for Gemma 3 1B: the longest token text is 48 bytes, so the
//     buffers take 48 and 147 bytes, reserved once, at load. A decoded token
//     is never longer than its text (a ▁ becomes a 1-byte space, a byte
//     token's "<0xXX>" its one byte), so it fits the first. At the worst
//     each of its bytes is one the stream cannot place, written as U+FFFD's
//     three bytes, and the up to three bytes held before them leave as
//     themselves, completed, or as one U+FFFD: 3 × 48 + 3, so it fits the
//     second. Nothing is allocated a token.
//   - Construction reads every token text once, 262,144 of them, to find
//     the longest.
//
// Verification
//   The tokenizer suite decodes every reference case's IDs a token at a time
//   through Utf8Stream, the stream this uses; no test drives a Detokenizer
//   itself, and none reaches U+FFFD.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer or reference: the
//            tokenizer is borrowed, held by pointer, owned by the generator.
//   C++ performance guidelines
//     MEM.9  Allocate at init, not in steady state: both buffers, at
//            construction.
//   Searched, and no rule applies: reusing a reserved string buffer across
//   calls (the rules found are for virtual memory, ring buffers and
//   GPU heaps).
class Detokenizer {
 public:
  // Borrows `tokenizer`, which outlives it (above).
  explicit Detokenizer(const Tokenizer& tokenizer);

  // The characters `token` completes, possibly none (above). Precondition:
  // the token is below the vocabulary's size.
  [[nodiscard]] std::string_view push(TokenId token);

  // Ends the reply: U+FFFD where bytes were pending, else nothing (above).
  [[nodiscard]] std::string_view finish();

  // The vocabulary's longest token, in bytes.
  [[nodiscard]] std::size_t longest() const noexcept { return longest_; }

 private:
  const Tokenizer* tokenizer_;
  std::size_t longest_ = 1;
  Utf8Stream stream_;
  std::string bytes_;
  std::string text_;
};

}  // namespace frizzle::tokenizer
