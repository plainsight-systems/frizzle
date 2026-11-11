#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Special tokens found in text. The chat template writes them as text, and
// each encodes as its one identifier, never as the characters that spell
// it. Every control and user-defined token is matched as the text stands
// once spaces are ▁, the one change SentencePiece BPE makes, with its
// tokens spelled to match (bpe/sentencepiece_bpe.h); nothing in the file
// marks one to be normalized further, stripped or matched as a whole word.
// Text is searched from the left, and at each position the longest
// matching token wins. That matters: Gemma 3's special tokens include runs
// of spaces, each a prefix of the next.
//
// Contract
//   - SpecialTokens(entries): matches exactly these, by the text each is
//     given. The texts live on only in its trie, so it depends on nothing
//     after construction, the vocabulary included. Precondition, not checked:
//     no text is empty and no two are the same; SentencePiece BPE's load
//     refuses two spelled alike before it builds this.
//   - SpecialTokens(vocabulary): every control and user-defined token with
//     text, spelled as the vocabulary spells it. The engine does not call
//     it; SentencePiece BPE passes its entries spelled with ▁.
//   - SpecialTokens(): none; every text is ordinary.
//   - segment(text, out, max_specials): appends to `out` the text's
//     ordinary runs and special tokens, in order, covering it exactly, each
//     by its byte range; ordinary runs are never empty. Returns false once
//     it has found more than max_specials special tokens, `out` holding what
//     it found up to them, so a text of special tokens grows `out` by at
//     most two segments a token allowed (tokenizer.h, encoding is bounded).
//     Precondition: `text` is no longer than a Segment can address, 4 GiB.
//   - longest(): the longest text a special token matches, 0 with none:
//     what one special token can take of a text's bytes.
//   - A built SpecialTokens is immutable.
//
// Algorithm
//   Build, from n entries:
//     1. Sort the entries by text, so each node's children are made in order
//        of byte and a byte already followed from a node is its last child.
//     2. Insert each text into a trie of byte edges from the root, node 0,
//        marking the node its last byte reaches with the token.
//     3. Lay the trie out flat: first_[b], the node the root's edge b leads
//        to; for node k, its edges at edge_start_[k] .. edge_start_[k + 1] − 1
//        of edge_byte_ and edge_child_, in order of byte; token_[k], the
//        token node k spells in full, or none.
//   segment, from position i = 0:
//     1. Walk from first_[text[i]], following one edge a byte by a linear
//        search of the node's edges; the deepest node passed that spells a
//        token is the longest match.
//     2. No match: i moves on one byte, the byte joining the ordinary run.
//        A match: the ordinary run before it, if any, is appended, then the
//        token, and i moves past it.
//     3. At the end, the ordinary run left, if any, is appended.
//
// For Gemma 3 1B's pinned file, counted: 6,414 special tokens, 6,251
// control and 163 user-defined, 6,321 of them starting "<"; their texts,
// spelled with ▁, are 76,774 bytes, the longest 93 (a run of 31 spaces).
// Their trie has 12,962 nodes and 12,961 edges; the root's edges start with
// 5 different bytes.
//
// Design and cost
//   - Optimization (practice): a byte trie of the tokens' texts, laid out
//     flat. A position whose byte starts no token costs one read of first_;
//     from there each byte follows one edge, so a match costs its length, at
//     most 93 bytes for Gemma 3, and the deepest token the walk passes is
//     the longest match. Matching the tokens one by one would scan
//     thousands of them at every "<" in the text. Timed outside Frizzle on
//     a corpus of text, the walk took 0.8 ms where the scan took 14.0; an
//     estimate, not measured in Frizzle.
//   - Resident for Gemma 3, counted: first_, 1 KiB; edge_start_ and token_,
//     4 bytes a node, 51,852 and 51,848 bytes; edge_child_, 4 bytes an edge,
//     51,844; edge_byte_, 1 byte an edge, 12,961: 169,529 bytes, 166 KiB,
//     built once at load. The entries, 76,774 bytes of text, and the
//     build's nested vectors of edges are freed when it returns.
//   - segment allocates only `out`'s growth: at most 2 × max_specials + 1
//     segments, 16 bytes each.
//
// Verification
//   The tokenizer suite encodes every reference case, among them rendered
//   chat prompts whose control tokens must segment to their identifiers.
//   No test passes a limit segment stops at.
//
// Corpus guidance
//   C++ performance guidelines
//     CDSA.11  Pick a trie or a hash by query shape: the longest special
//              token at a position is a longest-match query, so a trie.
//     CDSA.12  Compile static key sets into flat, pointer-free trie arrays:
//              the special tokens are fixed at load, built from sorted
//              texts into index arrays with no pointers.
//     CACHE.4  Choose the layout by the fields the hot loop touches: edge
//              bytes apart from edge children, so the search of a node's
//              edges reads only bytes.
//   C++ Core Guidelines
//     SL.con.3 Avoid bounds errors: first_ is reached through a span by a
//              byte, which cannot pass its 256 entries.
//     ES.45    Avoid "magic constants"; use symbolic constants: the 256
//              first bytes are kByteValues.

// A run of ordinary text, or one special token, by its byte range.
struct Segment {
  std::uint32_t offset;
  std::uint32_t length;
  std::optional<TokenId> special;  // empty for ordinary text
};

class SpecialTokens {
 public:
  // A token and the text that matches it.
  struct Entry {
    std::string text;
    TokenId id;
  };

  // None: every text is ordinary.
  SpecialTokens() = default;

  // Every control and user-defined token with text (above).
  explicit SpecialTokens(const Vocabulary& vocabulary);

  // Exactly these (above). Precondition: no text is empty, and no two are
  // the same.
  explicit SpecialTokens(std::vector<Entry> entries);

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  // The longest text a special token matches, 0 with none (above).
  [[nodiscard]] std::size_t longest() const noexcept { return longest_; }

  // Splits `text` into ordinary runs and special tokens, appended to `out`;
  // false past `max_specials` special tokens (above). Precondition: `text`
  // is no longer than a Segment can address, 4 GiB.
  bool segment(std::string_view text, std::vector<Segment>& out,
               std::uint64_t max_specials = UINT64_MAX) const;

 private:
  // The trie (above).
  static constexpr std::uint32_t kNone = ~std::uint32_t{0};
  static constexpr std::size_t kByteValues = 256;  // one first_ entry a byte

  std::array<std::uint32_t, kByteValues>
      first_{};  // the node each first byte leads to, or kNone
  // Node n's edges are [edge_start_[n], edge_start_[n + 1]), ordered by byte.
  std::vector<std::uint32_t> edge_start_;
  std::vector<unsigned char> edge_byte_;
  std::vector<std::uint32_t> edge_child_;
  std::vector<std::uint32_t>
      token_;  // the token node n spells in full, or kNone
  std::size_t size_ = 0;
  std::size_t longest_ = 0;
};

}  // namespace frizzle::tokenizer
