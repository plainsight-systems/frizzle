#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// SentencePiece BPE, the algorithm GGUF names "llama", as Gemma 3's file
// configures it: its load, its encode and decode, and the Tokenizer the
// capability table hands the runtime (kSentencePiece).
//
// It is BPE. The file lists no merges, only a score for each token, and
// llama.cpp merges any two adjacent symbols whose text joined is a token,
// highest score first, leftmost on a tie. The merges a vocabulary implies,
// every split of a normal token into two normal tokens ranked by the score
// of the token they make (sentencepiece_merges.h), give that search's
// result through the ordinary merge step (merge.h), so they are derived once,
// at load. Gemma 3's vocabulary implies 513,511. Charlotte (e003048) found
// them the same rules, in the same order, as Hugging Face's merges for
// Gemma 3, whose other 1,395 make runs of tabs, newlines and spaces, tokens
// matched as special before anything merges; not re-checked in Frizzle.
//
// Contract
//   load_sentencepiece_bpe(source, index, out): Ok with `out` loaded, or the
//   first refusal, `out` untouched, in this order:
//     - tokenizer.ggml.add_space_prefix: not declared, Unsupported (llama.cpp
//       reads a missing key as true); not BOOL, WrongKeyType; true,
//       Unsupported. The references place that space differently, llama.cpp
//       after every special token and Hugging Face only at the start, so
//       neither placement is chosen by a guess.
//     - Every refusal of load_vocabulary (vocabulary.h).
//     - tokenizer.ggml.scores: load.h's three key errors; a count other than
//       the vocabulary's, CountMismatch.
//     - Every refusal of implied_merges: InvalidScore, DuplicateMerge.
//     - A one-character token that is not normal, control, user-defined or
//       byte: Unsupported, naming it. The references would encode that
//       character as the token; this would spell it in bytes.
//     - A byte <0x00> to <0xFF> with no token of type Byte: MissingByte,
//       naming it; byte tokens other than those 256: Unsupported.
//     - Two special tokens spelled alike once spaces are ▁: Unsupported,
//       naming the spelling.
//   encode(raw, max_tokens, out): appends the tokens `raw` makes, or refuses
//   with `out` untouched: TooLong or TooManyTokens as tokenizer.h's bound
//   states, InvalidUtf8 for text that is not UTF-8. Returns the counts
//   admission made (admission.h) and the raw bytes. BOS is never added: the
//   chat template writes it as text.
//   decode(token, out): appends a normal token's text with each ▁ a space, a
//   byte token's byte, any other token's text as written. A ▁ typed in the
//   text comes back a space, as in SentencePiece and both references: the
//   vocabulary cannot tell the two apart. Precondition: loaded, and the token
//   below vocabulary().size().
//   longest_cover(): the longest token text in the vocabulary, the most
//   normalized bytes one token covers (tokenizer.h); 48 for Gemma 3, a normal
//   token.
//   A loaded SentencePieceBpe is not changed by encode or decode.
//
// Algorithm, encode
//   1. admit_raw on the raw bytes, the limit and the longest special token's
//      spelling (admission.h): past it, TooLong, unread.
//   2. Each space becomes ▁ (U+2581, three bytes), as the vocabulary spells
//      it. Nothing else is normalized, and no space is added before the text.
//   3. Special tokens are split out of that text, each spelled the same way
//      (special.h), stopping past the limit's worth as TooManyTokens. So
//      Gemma 3's user-defined runs of spaces match runs of ▁, whether the
//      text held spaces or the character ▁ itself, as SentencePiece matches
//      its user-defined pieces after spaces become ▁. Both references differ
//      where ▁ is typed beside a space: llama.cpp spells those tokens in
//      spaces and never matches a typed ▁, and Hugging Face matches typed
//      runs only where they stand alone.
//   4. admit over the specials and each ordinary segment's bytes, with
//      longest_cover() (admission.h).
//   5. In each ordinary segment, each character becomes its normal token, or,
//      where it has none, one byte token for each of its bytes; a byte that
//      does not begin a well-formed character is InvalidUtf8. The run is
//      merged (merge.h), cut before a ▁ only where no merge could cross it
//      (cut_before, below). A special token is its own identifier.
//   6. The tokens are appended to `out` together.
//
// For Gemma 3 1B's pinned file: add_space_prefix false; 262,144 tokens; 256
// byte tokens; 6,414 special tokens, the longest spelled in 93 bytes (a run of
// 31 spaces as ▁); 513,511 implied merges; one normal token with a ▁ after
// its first character, ">▁</".
//
// Design and cost
//   - Counted, once a load: the vocabulary and its table (vocabulary.h); the
//     scores, 1 MiB, freed when it returns; the merge table, 16 MiB
//     (merge_table.h); the special tokens' trie (special.h); a pass over the
//     vocabulary for each of the one-character check, the byte tokens, the
//     special tokens, the cover and the straddles, and 256 lookups.
//   - Counted, an encode of R raw bytes: the escaped copy, at most 3 R; the
//     segments; a vocabulary lookup a character; the merges (merge.h).
//   - BOS and the chat template's other control tokens are text the template
//     writes, so encode adds nothing.
//
// Verification
//   The tokenizer suite loads the pinned tokenizer through the capability
//   table and encodes every reference case to its IDs, and decodes them back,
//   whole and a token at a time. No test reaches a load refusal, TooLong,
//   TooManyTokens or InvalidUtf8.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.io.2  When reading, always consider ill-formed input: the file's
//              vocabulary is checked whole at load, and text before any merge.
//     E.27     If you can't throw exceptions, use error codes systematically:
//              LoadResult and EncodeResult.
//     I.5      State preconditions: decode's token.
//     C.139    Use final on classes sparingly: SentencePieceTokenizer, the one
//              implementation, is final; Tokenizer, the interface, is not. No
//              speed is claimed for it.
//   C++ performance guidelines
//     WASM.4   Reduce indirect dispatch in hot paths: final lets the compiler
//              call it directly where the type is known; the runtime calls
//              through Tokenizer once a turn and once a token.
class SentencePieceBpe {
 public:
  SentencePieceBpe() = default;

  [[nodiscard]] const Vocabulary& vocabulary() const noexcept {
    return vocabulary_;
  }

  // Appends the tokens `raw` makes, or refuses (above).
  [[nodiscard]] EncodeResult encode(std::string_view raw,
                                    std::uint32_t max_tokens,
                                    std::vector<TokenId>& out) const;

  // The longest token text in the vocabulary (above).
  [[nodiscard]] std::size_t longest_cover() const noexcept { return cover_; }

  // Appends the bytes `token` stands for (above). They can end partway
  // through a character; Utf8Stream makes text of them.
  void decode(TokenId token, std::string& out) const;

 private:
  friend LoadResult load_sentencepiece_bpe(gguf::ByteSource& source,
                                           const gguf::TensorIndex& index,
                                           SentencePieceBpe& out);

  // A normal token holding "▁" after its first character, and where.
  struct Straddle {
    std::string text;
    std::vector<std::uint32_t> marks;  // the byte offset of each such ▁
  };

  // Whether a stretch may be cut before the "▁" at text[at].
  //
  // Optimization (practice): a run goes from one special token to the next,
  // in Gemma 3 often a whole line, and merge.h merges a run of more than 64
  // symbols through a heap, 156 bytes a symbol (tokenizer.h). A merge that
  // crossed a cut would make a normal token holding that ▁ after its first
  // character, whose text would stand in the text across the cut; where none
  // does, the pieces on either side merge exactly as they would together.
  // So each is cut off and merged alone, a word at a time, most short enough
  // for merge.h's rescan, which allocates nothing. Gemma 3 has one such
  // token, ">▁</". A vocabulary with more than kMaxStraddles is not cut at
  // all.
  [[nodiscard]] bool cut_before(std::string_view text,
                                std::size_t at) const noexcept;

  static constexpr std::size_t kMaxStraddles = 64;
  static constexpr std::size_t kByteTokens = 256;  // <0x00> to <0xFF>

  Vocabulary vocabulary_;
  SpecialTokens special_;
  MergeTable merges_;
  std::array<TokenId, kByteTokens> byte_tokens_{};
  std::vector<Straddle> straddles_;
  bool cuts_ = false;  // false when there are more than kMaxStraddles
  std::size_t cover_ = 1;
};

// The algorithm, as the capability table lists it under "llama".
extern const Algorithm kSentencePiece;

// Loads the algorithm from the file, or refuses (above).
[[nodiscard]] LoadResult load_sentencepiece_bpe(gguf::ByteSource& source,
                                                const gguf::TensorIndex& index,
                                                SentencePieceBpe& out);

// The algorithm as a Tokenizer (tokenizer.h), each call passed to bpe_.
class SentencePieceTokenizer final : public Tokenizer {
 public:
  [[nodiscard]] const Vocabulary& vocabulary() const noexcept override;
  [[nodiscard]] EncodeResult encode(std::string_view text,
                                    std::uint32_t max_tokens,
                                    std::vector<TokenId>& out) override;
  [[nodiscard]] std::size_t longest_cover() const noexcept override;
  void decode(TokenId token, std::string& out) const override;

 private:
  friend LoadResult load_sentencepiece_tokenizer(
      gguf::ByteSource& source, const gguf::TensorIndex& index,
      std::unique_ptr<Tokenizer>& out);
  SentencePieceBpe bpe_;
};

// kSentencePiece's load (tokenizer.h's LoadFn): load_sentencepiece_bpe, its
// result a SentencePieceTokenizer. `out` is left untouched unless it
// succeeds.
[[nodiscard]] LoadResult load_sentencepiece_tokenizer(
    gguf::ByteSource& source, const gguf::TensorIndex& index,
    std::unique_ptr<Tokenizer>& out);

}  // namespace frizzle::tokenizer::bpe
