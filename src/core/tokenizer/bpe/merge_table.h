#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// The merge rules BPE runs by: which two adjacent tokens merge, into which
// token, and how early. A lower rank merges first, and rules may share a
// rank. Gemma 3's file lists no merges; its rules are derived from the
// vocabulary's scores at load (sentencepiece_merges.h), and this table holds
// them for the merge step (merge.h).
//
// Contract
//   - from_ranked_rules(rules): a table holding every rule at the rank it
//     carries, in any order; empty if two rules merge the same pair, or a
//     rank is 2^32 − 1, which the merge step reserves for a pair with no rule.
//   - find(left, right): the rule for `left` followed by `right`, its rank and
//     the token it makes, or empty. A default-constructed table finds
//     nothing.
//   - A built table is immutable; size() is how many rules it holds.
//
// Algorithm
//   1. S, the slot count: the smallest power of two of at least 16 with
//      3 S >= 4 r, for r rules; 1,048,576 for Gemma 3's 513,511.
//   2. A pair's key: left in the high 32 bits, right in the low. No pair is
//      the free marker, 2^64 − 1, as that needs two tokens numbered
//      2^32 − 1 and the vocabulary has 262,144.
//   3. Its slot: the top log2 S bits of key × 0x9E3779B97F4A7C15. Insert
//      probes forward from there, wrapping, and fails on the key already
//      present; find probes until the key or a free slot.
//
// Design and cost
//   - Optimization (practice): one flat array of slots, probed linearly, at
//     most three quarters full. The merge step looks up every adjacent pair
//     it meets; a lookup is usually one slot where bisecting 513,511 sorted
//     pairs is 19 steps, each a likely cache miss (counted).
//   - Resident for Gemma 3, counted: 1,048,576 slots of 16 bytes, a key and
//     a Merge, 16 MiB. Built once, at load.
//
// Verification
//   The tokenizer suite encodes every reference text through the table
//   Gemma 3's implied merges build. Neither refusal is reached by a test.
//
// Corpus guidance
//   C++ performance guidelines
//     CACHE.3  Prefer contiguous storage: the slots are one flat array.
//     CDSA.11  Pick a trie or a hash by query shape: every query is one exact
//              pair, so a hash.
//   C++ Core Guidelines
//     E.27     If you can't throw exceptions, use error codes systematically:
//              from_ranked_rules returns an empty table to refuse, which its
//              caller names DuplicateMerge.

struct MergeRule {
  TokenId left;
  TokenId right;
  TokenId result;
};

struct Merge {
  std::uint32_t rank;
  TokenId result;
};

// A rule with the rank it merges at, below 2^32 - 1 (above).
struct RankedRule {
  MergeRule rule;
  std::uint32_t rank;
};

class MergeTable {
 public:
  MergeTable() = default;

  // A table of `rules` at their ranks, or empty (above).
  [[nodiscard]] static std::optional<MergeTable> from_ranked_rules(
      std::span<const RankedRule> rules);

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  // The rule for `left` followed by `right`, if there is one.
  [[nodiscard]] std::optional<Merge> find(TokenId left,
                                          TokenId right) const noexcept;

 private:
  // Sized for `rules` rules, every slot free.
  explicit MergeTable(std::size_t rules);

  // Records `merge` for the pair. False if the pair is recorded already, or
  // cannot be a key.
  [[nodiscard]] bool insert(TokenId left, TokenId right, Merge merge);

  struct Slot {
    std::uint64_t
        pair;  // left in the high 32 bits, right in the low; kEmpty if free
    Merge merge;
  };

  // The table find probes (above).
  std::vector<Slot> slots_;  // a power of two
  std::size_t size_ = 0;
  // A hash keeps its top log2(slots) bits.
  unsigned shift_ = std::numeric_limits<std::uint64_t>::digits;
};

}  // namespace frizzle::tokenizer::bpe
