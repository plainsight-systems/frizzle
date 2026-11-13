#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// A reply's tokens as text: each token decoded (tokenizer.h) and pushed
// through a UTF-8 stream, giving the whole characters it completes. The
// generator (runtime/generator.h) makes one at load and passes what it
// gives to the page. And a turn's input tokens as text, all at once
// (texts, below), for the page to name the tokens attention read.
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
//   - texts(tokenizer, tokens, pool, offsets): each token's text as a fresh
//     stream gives it, pushed in order, token i's the characters it
//     completes and the last's also what finishing the stream gives (U+FFFD
//     where bytes were pending). The texts are appended to `pool` back to
//     back, and `offsets` is replaced by tokens.size() + 1 entries: token
//     i's text is pool[offsets[i], offsets[i + 1]) of what this call
//     appended, offsets[0] the pool's size before it. So each token's text
//     is whole characters, a token that only begins a character has none,
//     and the texts joined are the tokens' text: a ▁ a space, a control
//     token as the text that encodes to it, the type telling it apart
//     (vocabulary.h). Precondition: every token is below the vocabulary's
//     size. Never touches a Detokenizer: its own stream, so a reply's is
//     left as it was.
//
// Algorithm
//   Construction: longest = the longest token text in the vocabulary, at
//   least 1; reserve longest bytes for a token's bytes and 3 × longest + 3
//   for the text they complete.
//   push: clear both buffers; decode the token's bytes into the first; push
//   them through the stream into the second; return it.
//   finish: clear the text; let the stream end into it; return it.
//   texts: offsets = [pool's size]; for each token, decode it into one
//   reused buffer, push it through a local stream onto `pool`, and append
//   the pool's size; at the end, finish the stream onto `pool` and set the
//   last offset to the pool's size.
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
//   - texts: one pool and one index of offsets (GDSA.9), not a string a
//     token: a turn's input is up to the context offered, 32,768 tokens,
//     and each token's text is a few bytes. Two allocations a call at most,
//     the pool's and the offsets' growth, once a run (runtime/engine.h).
//
// Verification
//   The tokenizer suite decodes every reference case's IDs a token at a time
//   through Utf8Stream, the stream this uses; no test drives a Detokenizer
//   itself, and none reaches U+FFFD. texts is locked by the suite over the
//   same cases: one text a token, each the stream's characters for it, the
//   offsets as stated, the pool the case's text.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer or reference: the
//            tokenizer is borrowed, held by pointer, owned by the generator.
//   C++ performance guidelines
//     MEM.9  Allocate at init, not in steady state: both buffers, at
//            construction.
//     GDSA.9 Store variable-length groups as one pool plus offsets: texts'
//            pool and n + 1 offsets, built in token order.
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

// Each token's text, appended to `pool`, with tokens.size() + 1 offsets
// into it (above). Precondition: every token is below the vocabulary's size.
void texts(const Tokenizer& tokenizer, std::span<const TokenId> tokens,
           std::string& pool, std::vector<std::uint32_t>& offsets);

}  // namespace frizzle::tokenizer
