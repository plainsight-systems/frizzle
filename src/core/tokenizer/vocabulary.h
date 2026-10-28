#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/arrays.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// A model's vocabulary, read from tokenizer.ggml.tokens and
// tokenizer.ggml.token_type: every token's text and type, by identifier, and
// the identifier of a given text. Shared by every algorithm.
//
// A token's text is unique. No vocabulary listed here repeats one, and the
// references disagree on which repeat would win, so a file that repeats one is
// refused rather than read by a guess.

// Token types, numbered as GGUF numbers them.
enum class TokenType : std::int32_t {
  Undefined = 0,
  Normal = 1,
  Unknown = 2,
  Control =
      3,  // special: written as text by the chat template, never merged into
  UserDefined = 4,  // special: matched in the raw text as it stands
  Unused = 5,
  Byte = 6,  // one byte, for algorithms that fall back to bytes
};

enum class LoadError {
  Ok,
  MissingKey,        // the subject names it
  WrongKeyType,      // the subject names it
  Unreadable,        // the array could not be read; the subject says why
  CountMismatch,     // the arrays disagree on how many tokens there are
  UnknownTokenType,  // the subject names the token
  DuplicateToken,    // the subject names the text
  MalformedMerge,    // not two tokens separated by one space; the subject names
                     // it
  UnknownMergeToken,  // a merge names a token the vocabulary lacks
  DuplicateMerge,
  MissingByte,        // a byte-level vocabulary lacks a byte's token
  UnmappedCharacter,  // a byte-level token holds a character no byte stands
                      // for; the subject names it
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

  // Optimization (practice): find looks a text up in a flat table keyed by a
  // hash of it, at most three quarters full and probed linearly; each slot
  // holds a token and 32 bits of its text's hash, so a lookup compares
  // strings only where they almost surely match (CACHE.3). Loading looks up
  // hundreds of thousands of texts, and bisecting the sorted texts took about
  // 155 ns each where this takes about 20: Qwen3's tokenizer loads in 19 ms,
  // not 101, Llama 3.2's in 27, not 155, and Gemma 3's merges derive in 38,
  // not 363. It costs memory: 8 bytes a slot, 4 MiB for Gemma 3 where the
  // sorted identifiers took 1 MiB.
  struct Slot {
    std::uint32_t token;  // kFree when the slot is free
    std::uint32_t tag;    // the low 32 bits of its text's hash
  };
  static constexpr std::uint32_t kFree = ~std::uint32_t{0};

  [[nodiscard]] std::size_t first_slot(std::uint64_t hash) const noexcept {
    return static_cast<std::size_t>((hash * 0x9E3779B97F4A7C15) >> shift_);
  }

  gguf::StringTable tokens_;
  std::vector<TokenType> types_;
  std::vector<Slot> slots_;  // a power of two; empty when there are no tokens
  unsigned shift_ = 64;  // a slot is the top log2(slots) bits of the mixed hash
};

// Reads the vocabulary. `out` is left untouched unless the read succeeds.
[[nodiscard]] LoadResult load_vocabulary(gguf::ByteSource& source,
                                         const gguf::TensorIndex& index,
                                         Vocabulary& out);

}  // namespace frizzle::tokenizer
