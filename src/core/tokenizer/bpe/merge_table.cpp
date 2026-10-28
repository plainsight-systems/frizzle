#include "core/tokenizer/bpe/merge_table.h"

#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "core/tokenizer/load.h"

namespace frizzle::tokenizer::bpe {
namespace {

std::uint64_t pair_key(TokenId left, TokenId right) {
  return (std::uint64_t{static_cast<std::uint32_t>(left)} << 32) |
         static_cast<std::uint32_t>(right);
}

// A free slot. No pair has this key: it would need two tokens numbered
// 2^32 - 1, and no vocabulary is that large (GGUF arrays hold under 2^24).
constexpr std::uint64_t kEmpty = ~std::uint64_t{0};

// 2^64 divided by the golden ratio. Multiplying by it and keeping the top bits
// spreads keys that differ only in their low bits, as packed token pairs do,
// across the table (Knuth's multiplicative hashing).
constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15;

}  // namespace

MergeTable::MergeTable(std::size_t rules) {
  // The smallest power of two that keeps the table at most three quarters full.
  std::size_t slots = 16;
  unsigned bits = 4;
  while (slots * 3 < rules * 4) {
    slots *= 2;
    ++bits;
  }
  slots_.assign(slots, Slot{kEmpty, {}});
  shift_ = 64 - bits;
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

std::optional<MergeTable> MergeTable::from_rules(
    std::span<const MergeRule> rules) {
  MergeTable table(rules.size());
  for (std::size_t rank = 0; rank < rules.size(); ++rank) {
    const MergeRule& r = rules[rank];
    if (!table.insert(r.left, r.right,
                      {static_cast<std::uint32_t>(rank), r.result}))
      return std::nullopt;
  }
  return table;
}

std::optional<MergeTable> MergeTable::from_ranked_rules(
    std::span<const RankedRule> rules) {
  MergeTable table(rules.size());
  for (const RankedRule& r : rules) {
    if (r.rank == std::numeric_limits<std::uint32_t>::max())
      return std::nullopt;
    if (!table.insert(r.rule.left, r.rule.right, {r.rank, r.rule.result}))
      return std::nullopt;
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

LoadResult load_merges(gguf::ByteSource& source, const gguf::TensorIndex& index,
                       const Vocabulary& vocabulary, MergeTable& out) {
  constexpr std::string_view kMerges = "tokenizer.ggml.merges";
  gguf::StringTable merges;
  if (auto r = read_string_array(source, index, kMerges, merges); !r.ok())
    return r;

  std::vector<MergeRule> rules;
  rules.reserve(merges.size());
  std::string joined;
  for (std::size_t i = 0; i < merges.size(); ++i) {
    const std::string_view merge = merges[i];
    const std::size_t space = merge.find(' ');
    if (space == std::string_view::npos || space == 0 ||
        space + 1 == merge.size() ||
        merge.find(' ', space + 1) != std::string_view::npos) {
      return {LoadError::MalformedMerge, "merge " + std::to_string(i)};
    }
    const std::string_view left = merge.substr(0, space);
    const std::string_view right = merge.substr(space + 1);
    joined.assign(left).append(right);
    const auto l = vocabulary.find(left);
    const auto r = vocabulary.find(right);
    const auto result = vocabulary.find(joined);
    if (!l || !r || !result)
      return {LoadError::UnknownMergeToken, "merge " + std::to_string(i)};
    rules.push_back({*l, *r, *result});
  }
  auto table = MergeTable::from_rules(rules);
  if (!table) return {LoadError::DuplicateMerge, std::string(kMerges)};
  out = std::move(*table);
  return {};
}

}  // namespace frizzle::tokenizer::bpe
