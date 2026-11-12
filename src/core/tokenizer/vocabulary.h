#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/arrays.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// The vocabulary, read from tokenizer.ggml.tokens and
// tokenizer.ggml.token_type: every token's text and type by identifier, and
// the identifier of a given text. Also the outcome every tokenizer load
// returns, LoadError and LoadResult.
//
// Contract
//   - load_vocabulary(source, index, out): Ok with `out` holding every token,
//     identifier i the i-th entry of both arrays; or the first refusal, `out`
//     untouched:
//       MissingKey, WrongKeyType, Unreadable: either array missing; its
//         value not an array; or the array not readable whole as the type
//         read (strings, INT32), its elements of another type or its bytes
//         short; the subject names the key (load.h).
//       CountMismatch: the arrays differ in length; the subject names
//         tokenizer.ggml.token_type.
//       UnknownTokenType: a type outside 0..6; the subject names the token.
//       DuplicateToken: two tokens with the same text; the subject names the
//         text. The references disagree on which repeat would win, so a
//         repeat is refused rather than read by a guess.
//   - A loaded Vocabulary is immutable. text(id) and type(id) require id <
//     size(); text's view lives as long as the Vocabulary. find(text) is the
//     token whose text is exactly `text`, or empty.
//   - LoadError, by producer: the three key errors from load.h's readers, and
//     WrongKeyType also for add_space_prefix; CountMismatch from this load,
//     implied_merges and the SentencePiece load (scores against tokens);
//     UnknownTokenType and DuplicateToken from this load; DuplicateMerge and
//     InvalidScore from implied_merges (bpe/sentencepiece_merges.h);
//     MissingByte and Unsupported from load_sentencepiece_bpe
//     (bpe/sentencepiece_bpe.h). The subject says where.
//
// Algorithm, load_vocabulary
//   1. Read the texts into a StringTable and the types into INT32s; check
//      the counts agree and each type is 0..6.
//   2. Size the table: the smallest power of two S of at least 16 slots with
//      3 S >= 4 n, for n tokens; 524,288 for Gemma 3's 262,144.
//   3. For each token in order: h = FNV-1a 64 of its text; slot = the top
//      log2 S bits of h × 0x9E3779B97F4A7C15; tag = h's low 32 bits. Probe
//      forward from the slot, wrapping; a slot of equal tag and equal text is
//      DuplicateToken; the first free slot takes the token and tag.
//   find follows the same probe, comparing text only where the tag matches,
//   and stops at the first free slot.
//
// For Gemma 3 1B's pinned file: 262,144 tokens, 2,315,366 bytes of text;
// types 255,474 normal, 6,251 control, 163 user-defined, 256 byte.
//
// Design and cost
//   - Optimization (practice): a flat open-addressing table probed linearly,
//     at most three quarters full, each slot a token and 32 bits of its
//     text's hash, so a lookup compares strings only where they almost surely
//     match. The SentencePiece load and its implied merges look up every
//     normal token's splits, and encoding looks up every character; a probe
//     is usually one slot where bisecting 262,144 sorted texts is 18 steps,
//     each a likely cache miss (counted).
//   - Resident for Gemma 3, counted: the text, 2.2 MiB; its ends, 8 bytes a
//     token natively and 4 in the browser, 2 or 1 MiB; the types, 1 byte a
//     token, 0.25 MiB; the slots, 8 bytes each, 4 MiB: 7.5 MiB in the
//     browser. The load's INT32 types, 1 MiB, are freed when it returns.
//
// Verification
//   The tokenizer suite loads the pinned tokenizer through this and encodes
//   every reference text. No test reaches a refusal.
//
// Corpus guidance
//   C++ performance guidelines
//     CACHE.3  Prefer contiguous storage: the slots are one flat array.
//     CDSA.11  Pick a trie or a hash by query shape: find answers exact-text
//              point queries only, so a hash; special tokens, a longest-match
//              query, are a trie (special.h).
//   C++ Core Guidelines
//     E.27     If you can't throw exceptions, use error codes systematically:
//              LoadError and its subject, for every tokenizer load.
//     I.5      State preconditions: text and type take an identifier below
//              size().
//     Enum.7   Specify an enumeration's underlying type only when necessary:
//              TokenType and LoadError take one byte, the smallest that holds
//              them; a type a token, so the types take 0.25 MiB, not 1.

// Token types, numbered as GGUF numbers them. One byte each: the load checks
// GGUF's INT32 lies in 0..6 before narrowing it.
enum class TokenType : std::uint8_t {
  Undefined = 0,
  Normal = 1,
  Unknown = 2,
  Control =
      3,  // special: written as text by the chat template, never merged into
  UserDefined = 4,  // special: matched in the text once spaces are ▁
  Unused = 5,
  Byte = 6,  // one byte, <0x00> to <0xFF>, for a character with no token
};

enum class LoadError : std::uint8_t {
  Ok,
  MissingKey,        // the subject names it
  WrongKeyType,      // the subject names it
  Unreadable,        // the array could not be read; the subject says why
  CountMismatch,     // the arrays disagree on how many tokens there are
  UnknownTokenType,  // the subject names the token
  DuplicateToken,    // the subject names the text
  DuplicateMerge,
  MissingByte,   // a byte <0xXX> has no token of type Byte; the subject names
                 // it
  InvalidScore,  // a token's score cannot be ordered (NaN); the subject names
                 // the token
  Unsupported,   // the file asks for behaviour this build does not implement;
                 // the subject says what
};

struct LoadResult {
  LoadError error = LoadError::Ok;
  std::string subject;

  [[nodiscard]] bool ok() const noexcept { return error == LoadError::Ok; }
};

class Vocabulary {
 public:
  [[nodiscard]] std::size_t size() const noexcept { return types_.size(); }

  // Precondition: the identifier is below size().
  [[nodiscard]] std::string_view text(TokenId id) const noexcept {
    return tokens_[static_cast<std::size_t>(id)];
  }
  [[nodiscard]] TokenType type(TokenId id) const noexcept {
    return types_[static_cast<std::size_t>(id)];
  }

  // The token whose text is exactly `text`, if there is one.
  [[nodiscard]] std::optional<TokenId> find(
      std::string_view text) const noexcept;

 private:
  friend LoadResult load_vocabulary(gguf::ByteSource& source,
                                    const gguf::TensorIndex& index,
                                    Vocabulary& out);

  // The table find probes (above).
  struct Slot {
    std::uint32_t token;  // kFree when the slot is free
    std::uint32_t tag;    // the low 32 bits of its text's hash
  };
  static constexpr std::uint32_t kFree = ~std::uint32_t{0};
  // 2^64 divided by the golden ratio: multiplying by it and keeping the top
  // bits spreads hashes across the table (Knuth's multiplicative hashing).
  static constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15;

  [[nodiscard]] std::size_t first_slot(std::uint64_t hash) const noexcept {
    return static_cast<std::size_t>((hash * kGolden) >> shift_);
  }

  gguf::StringTable tokens_;
  std::vector<TokenType> types_;
  std::vector<Slot> slots_;  // a power of two; empty when there are no tokens
  // A slot is the top log2(slots) bits of the mixed hash.
  unsigned shift_ = std::numeric_limits<std::uint64_t>::digits;
};

// Reads the vocabulary. `out` is left untouched unless the read succeeds.
[[nodiscard]] LoadResult load_vocabulary(gguf::ByteSource& source,
                                         const gguf::TensorIndex& index,
                                         Vocabulary& out);

}  // namespace frizzle::tokenizer
