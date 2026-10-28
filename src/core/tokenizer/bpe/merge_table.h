#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// A BPE model's merge rules: which two adjacent tokens merge, into which
// token, and how early. The lower rank merges first. A rule's rank is its
// place in the list the file holds, or, where the file holds none, given with
// the rule: rules may then share a rank.

struct MergeRule {
  TokenId left;
  TokenId right;
  TokenId result;
};

struct Merge {
  std::uint32_t rank;
  TokenId result;
};

// A rule with the rank it merges at. Below 2^32 - 1, which the merge step
// reserves for a pair with no rule.
struct RankedRule {
  MergeRule rule;
  std::uint32_t rank;
};

class MergeTable {
 public:
  MergeTable() = default;

  // A table of `rules` in rank order, the first ranked 0. Empty if two
  // rules merge the same pair: the references disagree on which would win,
  // and no vocabulary listed here repeats one.
  [[nodiscard]] static std::optional<MergeTable> from_rules(
      std::span<const MergeRule> rules);

  // A table of `rules` at the ranks they carry, in any order. Empty if two
  // rules merge the same pair, or a rank is 2^32 - 1.
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

  // Optimization (practice): an open-addressing hash table, one flat array of
  // slots probed linearly, at most three quarters full. The merge loop looks
  // up every pair it meets, and a lookup is now usually one slot where
  // bisecting 280,147 sorted pairs was about eighteen steps, each a likely
  // cache miss. It costs memory: 8 MiB for Llama 3.2's merges where the sorted
  // pairs took 4.5 MiB. Hugging Face tokenizers uses a perfect hash, which is
  // smaller and probes once, at the cost of a slower build.
  std::vector<Slot> slots_;  // a power of two
  std::size_t size_ = 0;
  unsigned shift_ = 64;  // a hash keeps its top log2(slots) bits
};

// Reads tokenizer.ggml.merges: each "left right", two tokens' text separated
// by one space, merging to the token whose text is both. A byte-level
// vocabulary spells a space as "Ġ", so no token text holds one. Every token
// named must be in `vocabulary`. `out` is left untouched unless the read
// succeeds.
[[nodiscard]] LoadResult load_merges(gguf::ByteSource& source,
                                     const gguf::TensorIndex& index,
                                     const Vocabulary& vocabulary,
                                     MergeTable& out);

}  // namespace frizzle::tokenizer::bpe
