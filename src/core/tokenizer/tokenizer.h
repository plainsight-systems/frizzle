#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace frizzle::gguf {
class ByteSource;
class TensorIndex;
}  // namespace frizzle::gguf

namespace frizzle::tokenizer {

class Vocabulary;
struct LoadResult;

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// The tokenizer's interface: how the runtime turns rendered text into token
// identifiers and identifiers back into bytes, whatever loaded them, and the
// bound every encode keeps. Its one implementation is SentencePiece BPE
// (bpe/sentencepiece_bpe.h), which the capability table lists under the
// tokenizer.ggml.model value "llama", the pinned file's.
//
// Contract
//   - An Algorithm is constant data: its tokenizer.ggml.model name and the
//     function that loads it. The capability table is the one list of them;
//     the runtime and the boundary name none.
//   - load(source, index, out): builds the tokenizer from the vocabulary
//     arrays the index locates, or returns the LoadResult naming what it
//     refused (vocabulary.h); `out` is left untouched unless it succeeds.
//   - Encoding turns rendered text into identifiers. The chat template writes
//     special tokens as text; they encode as their single identifiers, never
//     as the characters that spell them.
//   - Decoding gives each token's bytes back, special tokens as the text that
//     encodes to them, and rewrites nothing beyond the algorithm's
//     normalizing: decoding what was encoded gives the text back with each ▁
//     the space it escaped. Whether a control token is shown is the caller's
//     choice; it has the token's type. Bytes become text through Utf8Stream.
//   - A loaded tokenizer is used through Tokenizer, below: an abstract
//     interface with no data, its implementation chosen once at load and
//     final. The runtime calls it once a turn to encode and once a token to
//     decode.
//   - Encoding is bounded, by the caller's token limit and by the bytes one
//     encode takes, before any run is merged, as the merge's work and memory
//     grow with a run's length (bpe/merge.h). encode takes the limit,
//     max_tokens, and returns an EncodeResult: the error, the least tokens
//     and the normalized bytes admission counted, and the raw bytes. In order:
//       1. admit_raw (admission.h): raw text past 7/2 × kMaxEncodeBytes plus
//          max_tokens × the longest special token's spelling is TooLong,
//          unread. Text that could be admitted is within it: at most the
//          limit's special tokens, and ordinary bytes that normalize to at
//          most kMaxEncodeBytes; escaping spaces as ▁ never shrinks text.
//          The 7/2 is looser than this tokenizer needs: its normalizing
//          never shrinks text, so a ratio of 1 would bound it too. A looser
//          early refusal is still sound: text it lets through that is too
//          long is refused at step 3.
//       2. Each space becomes ▁, and the special tokens, spelled that way,
//          are split out, stopping past the limit's worth of them as
//          TooManyTokens, so the segments are at most two a token allowed.
//       3. admit: each special token is a token, and a segment of n
//          normalized bytes makes at least ceil(n / longest_cover()) tokens,
//          as no token spans a special or two segments; their sum, in 64
//          bits, over the limit is TooManyTokens, naming that least count;
//          then normalized text past kMaxEncodeBytes is TooLong, naming its
//          bytes. The token check comes first, so text a context could never
//          hold is named by its tokens, by which the page shortens a
//          conversation. The count is a lower bound: admitted text may still
//          make more tokens than the limit, which the caller counts after
//          encoding; none of the three refuses text that would fit.
//       4. Only admitted text is merged.
//     A refusal leaves `out` untouched. kNoTokenLimit, a test's, leaves step
//     1 nothing to refuse.
//   - longest_cover() is the most normalized bytes one token covers: the
//     longest token text in the vocabulary, computed at load; 48 for
//     Gemma 3, a normal token, every other type shorter.
//
// Algorithm
//   The runtime, at load: reads tokenizer.ggml.model, finds its Algorithm in
//   the capability table, and calls load once. A turn: one encode of the
//   rendered prompt with the context offered as max_tokens
//   (runtime/generator.cpp), then one decode a token drawn, through the
//   Detokenizer (detokenizer.h).
//
// Design and cost
//   - An abstract interface rather than the class itself, so the runtime and
//     the boundary depend on no algorithm's headers, and the capability table
//     stays the one place that names one. Its cost is one indirect call an
//     encode and one a token.
//   - kMaxEncodeBytes, 512 KiB of normalized text, is sized from the
//     browser module's worst case during an encode (WASM.1, EMB.4), counted
//     for Gemma 3 1B in the 32-bit browser build, with T the token limit, at
//     most the trained context, 32,768, and N at most 512 KiB of normalized
//     ordinary bytes:
//       Resident, once loaded: the vocabulary, 7.5 MiB (vocabulary.h); the
//         merge table, 16 MiB (bpe/merge_table.h); the special tokens'
//         trie, which holds their texts, under 0.5 MiB (special.h): under
//         25 MiB. Natively, 64-bit, about 30.2 MiB, an estimate, not
//         measured in Frizzle.
//       The encode, before merging: the raw text, at most 7/2 × 512 KiB +
//         93 T, 4.7 MiB; its escaped copy, at most 3 times that, doubled by
//         its growth, 28 MiB; the segments, 16 bytes each, at most 2 T + 1,
//         doubled, 2 MiB; the ordinary segments' lengths, 8 bytes each,
//         doubled, 0.5 MiB: 35 MiB.
//       The encode, merging, per normalized byte: the run's symbols, 4,
//         doubled; the merge of a run holding all of it, its chain, 12, and
//         its candidate heap, at most 2 a symbol live, 24 bytes each, whose
//         vector's capacity, at most 4 a symbol, is copied once more as it
//         grows, 144; the tokens, 4, doubled, and the generator's copy, 4:
//         176 bytes, at most 88 MiB.
//       Together, 123 MiB, 246 with EMB.4's margin of 2. Beside the resident
//       tokenizer, the 16 MiB chunk buffer and the 16 MiB front
//       (CMakeLists.txt), that is under 310 MiB: past the 128 MiB initial
//       heap, so a worst-case encode grows the heap, and under the 1 GiB
//       maximum. The worst case is counted, not measured.
//   - 512 KiB is 16 normalized bytes a token over a context of 32,768. What
//     it refuses that a context could hold is text averaging more than that
//     over the whole context, such as long runs of letters: a stated limit
//     (WASM.1's caveat), refused as TooLong, naming its bytes.
//
// Verification
//   The tokenizer suite encodes every reference case with no limit to its
//   IDs, and decodes them back. Untested: admit_raw and admit, TooLong,
//   TooManyTokens, the limit met exactly and passed by one, and the bound
//   above.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.25    Prefer empty abstract classes as interfaces to class
//             hierarchies: Tokenizer holds no data.
//     C.121   If a base class is used as an interface, make it a pure
//             abstract class: every member is pure virtual.
//     C.127   A class with a virtual function should have a virtual or
//             protected destructor: the loaded tokenizer is owned through
//             std::unique_ptr<Tokenizer>, so public and virtual.
//     I.4     Make interfaces precisely and strongly typed: TokenId is its own
//             type, never a count or a position.
//     SL.io.2 When reading, always consider ill-formed input: a prompt of any
//             size is input; its size is checked before work grows with it.
//     E.27    If you can't throw exceptions, use error codes systematically:
//             EncodeResult, with its counts.
//     C.67    A polymorphic class should suppress public copy/move: Tokenizer
//             has no data, so its copy and move are deleted.
//     Enum.7  Specify an enumeration's underlying type only when necessary:
//             EncodeError takes one byte, the smallest that holds it.
//   C++ performance guidelines
//     WASM.4  Reduce indirect dispatch in hot paths: the dispatch is once a
//             turn and once a token, the implementation final.
//     WASM.1  Size linear memory to the real high-water mark: the encode's
//             worst case counted against the heap's initial size and maximum,
//             the cap stated.
//     EMB.4   Size the worst case of dynamic allocation, with margin: 2.

// A token's identifier. Its own type, so it cannot be passed where a count or
// a position is meant.
enum class TokenId : std::uint32_t {};

// The most normalized bytes one encode takes (above).
inline constexpr std::uint64_t kMaxEncodeBytes = 512ull * 1024;

// A token limit that refuses nothing by count, for an encode bounded by
// kMaxEncodeBytes alone.
inline constexpr std::uint32_t kNoTokenLimit = UINT32_MAX;

enum class EncodeError : std::uint8_t {
  Ok,
  InvalidUtf8,
  TooManyTokens,  // more tokens, at least, than the caller's limit
  TooLong,        // past kMaxEncodeBytes, normalized
};

// An encode's outcome: the least tokens the text makes and its normalized
// bytes, as admission counted them (admission.h), and its raw bytes. Text
// refused before it is segmented or normalized has only its raw bytes; text
// refused for its special tokens alone, its least tokens.
struct EncodeResult {
  EncodeError error = EncodeError::Ok;
  std::uint64_t least_tokens = 0;
  std::uint64_t normalized_bytes = 0;
  std::uint64_t raw_bytes = 0;
};

// A loaded tokenizer, whatever its algorithm.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  Tokenizer() = default;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) = delete;
  Tokenizer& operator=(Tokenizer&&) = delete;

  [[nodiscard]] virtual const Vocabulary& vocabulary() const noexcept = 0;

  // Appends the tokens `text` encodes to, refusing text that makes more
  // than `max_tokens` at least, or is too long to encode (above); `out` is
  // left untouched if it refuses or cannot. Not const, so an implementation
  // may keep what it finds between calls; SentencePiece BPE keeps nothing.
  [[nodiscard]] virtual EncodeResult encode(std::string_view text,
                                            std::uint32_t max_tokens,
                                            std::vector<TokenId>& out) = 0;

  // The most normalized bytes one token covers (above).
  [[nodiscard]] virtual std::size_t longest_cover() const noexcept = 0;

  // Appends the bytes `token` stands for. Precondition: the token is below
  // vocabulary().size().
  virtual void decode(TokenId token, std::string& out) const = 0;
};

// Loads an algorithm's tokenizer from the file the index describes (above).
// `out` is left untouched unless it succeeds.
using LoadFn = LoadResult (*)(gguf::ByteSource& source,
                              const gguf::TensorIndex& index,
                              std::unique_ptr<Tokenizer>& out);

struct Algorithm {
  std::string_view name;
  LoadFn load;
};

// Turns the bytes of successive tokens into text. A token can end partway
// through a UTF-8 character; those bytes are held until the next token
// completes them, so every string emitted is whole characters. Bytes that
// cannot begin or continue a character are replaced with U+FFFD.
class Utf8Stream {
 public:
  // Appends every character completed by `token_bytes` to `out`.
  void push(std::string_view token_bytes, std::string& out);

  // Ends the stream. Returns false if bytes were pending — generation
  // stopped inside a character — in which case U+FFFD is appended in their
  // place rather than dropping them unseen. The stream is then empty, ready
  // for another.
  [[nodiscard]] bool finish(std::string& out);

 private:
  std::array<char, 3> pending_{};
  std::uint8_t pending_count_ = 0;
};

}  // namespace frizzle::tokenizer
