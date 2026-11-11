// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Contract, algorithm, cost and corpus guidance: merge_table.h. This file
// defines what that header states.

#include "core/tokenizer/bpe/merge_table.h"

#include <bit>
#include <limits>
#include <type_traits>

namespace frizzle::tokenizer::bpe {
namespace {

// A token identifier's width: a key holds `left` above `right`.
constexpr int kTokenBits =
    std::numeric_limits<std::underlying_type_t<TokenId>>::digits;

std::uint64_t pair_key(TokenId left, TokenId right) {
  return (std::uint64_t{static_cast<std::uint32_t>(left)} << kTokenBits) |
         static_cast<std::uint32_t>(right);
}

// A free slot. No pair has this key: it would need two tokens numbered
// 2^32 - 1 (merge_table.h).
constexpr std::uint64_t kEmpty = ~std::uint64_t{0};

// 2^64 divided by the golden ratio. Multiplying by it and keeping the top bits
// spreads keys that differ only in their low bits, as packed token pairs do,
// across the table (Knuth's multiplicative hashing).
constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15;

// The fewest slots a table has (merge_table.h).
constexpr std::size_t kMinSlots = 16;

}  // namespace

MergeTable::MergeTable(std::size_t rules) {
  // The smallest power of two that keeps the table at most three quarters full.
  std::size_t slots = kMinSlots;
  unsigned bits = std::countr_zero(kMinSlots);
  while (slots * 3 < rules * 4) {
    slots *= 2;
    ++bits;
  }
  slots_.assign(slots, Slot{kEmpty, {}});
  shift_ = std::numeric_limits<std::uint64_t>::digits - bits;
}

bool MergeTable::insert(TokenId left, TokenId right, Merge merge) {
  const std::uint64_t key = pair_key(left, right);
  if (key == kEmpty) return false;
  const std::size_t mask = slots_.size() - 1;
  std::size_t at = static_cast<std::size_t>((key * kGolden) >> shift_);
  while (slots_[at].pair != kEmpty) {
    if (slots_[at].pair == key) return false;  // the pair repeats
    at = (at + 1) & mask;
  }
  slots_[at] = {key, merge};
  ++size_;
  return true;
}

std::optional<MergeTable> MergeTable::from_ranked_rules(
    std::span<const RankedRule> rules) {
  MergeTable table(rules.size());
  for (const RankedRule& r : rules) {
    if (r.rank == std::numeric_limits<std::uint32_t>::max()) {
      return std::nullopt;
    }
    if (!table.insert(r.rule.left, r.rule.right, {r.rank, r.rule.result})) {
      return std::nullopt;
    }
  }
  return table;
}

std::optional<Merge> MergeTable::find(TokenId left,
                                      TokenId right) const noexcept {
  if (slots_.empty()) return std::nullopt;
  const std::uint64_t key = pair_key(left, right);
  const std::size_t mask = slots_.size() - 1;
  for (std::size_t at = static_cast<std::size_t>((key * kGolden) >> shift_);;
       at = (at + 1) & mask) {
    if (slots_[at].pair == key) return slots_[at].merge;
    if (slots_[at].pair == kEmpty) return std::nullopt;
  }
}

}  // namespace frizzle::tokenizer::bpe
