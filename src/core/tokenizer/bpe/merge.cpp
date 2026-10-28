#include "core/tokenizer/bpe/merge.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>

namespace frizzle::tokenizer::bpe {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// Pieces up to this many bytes are merged by rescanning; longer ones by the
// heap below.
// Optimization (practice): tiktoken rescans pieces under 100 bytes and Hugging
// Face tokenizers rescans short words, because for a few symbols a scan of
// adjacent ranks beats a heap. Measured here on real text, the rescan is about
// 20% faster from 3 to 32 bytes, even at 64, and slower beyond; 99% of pieces
// are 16 bytes or under.
constexpr std::size_t kShortPiece = 64;

// The rank of a pair with no rule, above every real rank.
constexpr std::uint32_t kNoRule = std::numeric_limits<std::uint32_t>::max();

// A short piece: the symbols in a row, and the rule of each adjacent pair. Each
// round takes the lowest rank, the leftmost on a tie, merges it, closes the
// gap, and looks up only the two pairs the new token is part of. Fixed arrays
// on the stack: a short piece allocates nothing.
void merge_short(std::span<const TokenId> symbols, const MergeTable& merges,
                 std::vector<TokenId>& out) {
  std::array<TokenId, kShortPiece> parts;
  std::array<std::uint32_t, kShortPiece>
      ranks;  // ranks[i] is the pair parts[i], parts[i + 1]
  std::array<TokenId, kShortPiece> results;
  std::size_t count = symbols.size();
  std::copy(symbols.begin(), symbols.end(), parts.begin());

  const auto look_up = [&](std::size_t i) {
    const auto rule = merges.find(parts[i], parts[i + 1]);
    ranks[i] = rule ? rule->rank : kNoRule;
    if (rule) results[i] = rule->result;
  };
  for (std::size_t i = 0; i + 1 < count; ++i) look_up(i);

  while (count > 1) {
    std::size_t best = 0;
    for (std::size_t i = 1; i + 1 < count; ++i) {
      if (ranks[i] < ranks[best]) best = i;  // strict: the leftmost wins a tie
    }
    if (ranks[best] == kNoRule) break;
    parts[best] = results[best];
    std::copy(parts.begin() + best + 2, parts.begin() + count,
              parts.begin() + best + 1);
    if (best + 2 < count - 1) {  // pairs after the next one shift down; none
                                 // when it was the last
      std::copy(ranks.begin() + best + 2, ranks.begin() + count - 1,
                ranks.begin() + best + 1);
      std::copy(results.begin() + best + 2, results.begin() + count - 1,
                results.begin() + best + 1);
    }
    --count;
    if (best > 0) look_up(best - 1);
    if (best + 1 < count) look_up(best);
  }
  out.insert(out.end(), parts.begin(), parts.begin() + count);
}

// One symbol of the piece, linked to its neighbours by position. A merge
// gives the left symbol the merged token and unlinks the right one, which no
// pair ever reaches again.
struct Symbol {
  TokenId token;
  std::uint32_t prev;
  std::uint32_t next;
};

// An adjacent pair that has a rule, as it stood when it was found. Merges
// around it may since have changed it, so it is checked again before it
// merges.
struct Candidate {
  std::uint32_t rank;
  std::uint32_t
      left;  // positions; the left one also breaks ties, leftmost first
  std::uint32_t right;
  TokenId left_token;
  TokenId right_token;
  TokenId result;

  // For a min-heap: the lowest rank first, then the leftmost.
  bool operator>(const Candidate& other) const noexcept {
    return rank != other.rank ? rank > other.rank : left > other.left;
  }
};

using Queue =
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>>;

// Queues the pair that starts at `left`, if it has a rule.
void consider(const std::vector<Symbol>& piece, std::uint32_t left,
              const MergeTable& merges, Queue& queue) {
  const std::uint32_t right = piece[left].next;
  if (right == kNone) return;
  if (const auto rule = merges.find(piece[left].token, piece[right].token)) {
    queue.push({rule->rank, left, right, piece[left].token, piece[right].token,
                rule->result});
  }
}

// True when the pair still stands as it did when it was queued: both symbols
// linked to each other and holding the same tokens. An unlinked symbol has no
// next, so a pair naming one never stands.
bool still_stands(const std::vector<Symbol>& piece, const Candidate& c) {
  return piece[c.left].next == c.right && piece[c.left].token == c.left_token &&
         piece[c.right].token == c.right_token;
}

// A long piece: the chain and the heap.
void merge_long(std::span<const TokenId> symbols, const MergeTable& merges,
                std::vector<TokenId>& out) {
  const auto count = static_cast<std::uint32_t>(symbols.size());

  std::vector<Symbol> piece(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    piece[i] = {symbols[i], i == 0 ? kNone : i - 1,
                i + 1 == count ? kNone : i + 1};
  }

  Queue queue;
  for (std::uint32_t i = 0; i + 1 < count; ++i)
    consider(piece, i, merges, queue);

  while (!queue.empty()) {
    const Candidate c = queue.top();
    queue.pop();
    if (!still_stands(piece, c)) continue;

    // The left symbol becomes the merged token and takes over the right
    // one's place in the chain.
    Symbol& left = piece[c.left];
    Symbol& right = piece[c.right];
    left.token = c.result;
    left.next = right.next;
    if (right.next != kNone) piece[right.next].prev = c.left;
    right.prev = right.next = kNone;

    // Only the two pairs that include the new token are new.
    if (left.prev != kNone) consider(piece, left.prev, merges, queue);
    consider(piece, c.left, merges, queue);
  }

  // The first symbol is never the right of a pair, so the chain starts at 0.
  for (std::uint32_t i = 0; i != kNone; i = piece[i].next)
    out.push_back(piece[i].token);
}

}  // namespace

void merge(std::span<const TokenId> symbols, const MergeTable& merges,
           std::vector<TokenId>& out) {
  if (symbols.empty()) return;
  if (symbols.size() <= kShortPiece) {
    merge_short(symbols, merges, out);
  } else {
    merge_long(symbols, merges, out);
  }
}

}  // namespace frizzle::tokenizer::bpe
