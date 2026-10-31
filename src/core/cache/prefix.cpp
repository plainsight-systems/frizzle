// Axis G: changes with a new cache or context mechanism.
//
// Contract, algorithm and cost: prefix.h. This file defines what that header
// states.

#include "core/cache/prefix.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <tuple>
#include <vector>

namespace frizzle::cache {
namespace {

// One range of one neuron, with what it does there, compared by bits.
struct Segment {
  std::uint32_t layer;
  std::uint32_t neuron;
  edit::EditKind kind;
  std::uint32_t bits;
  std::uint32_t from;
  std::uint32_t until;
};

bool same_neuron(const Segment& a, const Segment& b) {
  return a.layer == b.layer && a.neuron == b.neuron;
}

bool same_edit(const Segment& a, const Segment& b) {
  return a.kind == b.kind && a.bits == b.bits;
}

// The set sorted by (layer, neuron, from), with ranges of one neuron that
// meet and do the same thing joined: one form for every split or order of
// the same edits in force.
std::vector<Segment> normal(std::span<const edit::Edit> edits) {
  std::vector<Segment> s;
  s.reserve(edits.size());
  for (const edit::Edit& e : edits) {
    s.push_back({e.layer, e.neuron, e.kind,
                 std::bit_cast<std::uint32_t>(e.value), e.from, e.until});
  }
  std::ranges::sort(s, [](const Segment& a, const Segment& b) {
    return std::tie(a.layer, a.neuron, a.from) <
           std::tie(b.layer, b.neuron, b.from);
  });
  std::vector<Segment> joined;
  joined.reserve(s.size());
  for (const Segment& x : s) {
    if (!joined.empty()) {
      Segment& last = joined.back();
      if (same_neuron(last, x) && same_edit(last, x) && last.until == x.from) {
        last.until = x.until;
        continue;
      }
    }
    joined.push_back(x);
  }
  return joined;
}

// End of the group of one neuron that begins at `at`.
std::size_t group_end(const std::vector<Segment>& s, std::size_t at) {
  std::size_t end = at;
  while (end < s.size() && same_neuron(s[end], s[at])) ++end;
  return end;
}

// The first position where two normal lists of one neuron differ, or
// kToTheEnd. Each list's ranges are apart and none meets one doing the same
// thing, so once two ranges start together and do the same thing, they
// differ where the shorter stops.
std::uint32_t neuron_difference(const std::vector<Segment>& a, std::size_t i,
                                std::size_t i_end,
                                const std::vector<Segment>& b, std::size_t j,
                                std::size_t j_end) {
  while (i < i_end && j < j_end) {
    const Segment& x = a[i];
    const Segment& y = b[j];
    if (x.from != y.from) return std::min(x.from, y.from);
    if (!same_edit(x, y)) return x.from;
    if (x.until != y.until) return std::min(x.until, y.until);
    ++i;
    ++j;
  }
  if (i < i_end) return a[i].from;
  if (j < j_end) return b[j].from;
  return edit::kToTheEnd;
}

}  // namespace

std::size_t longest_common_prefix(
    std::span<const tokenizer::TokenId> cached,
    std::span<const tokenizer::TokenId> incoming) noexcept {
  const auto [end, ignored] = std::mismatch(cached.begin(), cached.end(),
                                            incoming.begin(), incoming.end());
  return static_cast<std::size_t>(end - cached.begin());
}

// Symmetric: the two sets swapped give the same position, so arguments
// passed in either order are correct.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::uint32_t first_edit_difference(std::span<const edit::Edit> cached,
                                    std::span<const edit::Edit> incoming) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  const std::vector<Segment> a = normal(cached);
  const std::vector<Segment> b = normal(incoming);
  std::uint32_t least = edit::kToTheEnd;
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() || j < b.size()) {
    // A neuron in one set alone differs at its first range's start.
    const bool a_first =
        j == b.size() ||
        (i < a.size() &&
         std::tie(a[i].layer, a[i].neuron) < std::tie(b[j].layer, b[j].neuron));
    const bool b_first =
        i == a.size() ||
        (j < b.size() &&
         std::tie(b[j].layer, b[j].neuron) < std::tie(a[i].layer, a[i].neuron));
    if (a_first) {
      least = std::min(least, a[i].from);
      i = group_end(a, i);
    } else if (b_first) {
      least = std::min(least, b[j].from);
      j = group_end(b, j);
    } else {
      const std::size_t i_end = group_end(a, i);
      const std::size_t j_end = group_end(b, j);
      least = std::min(least, neuron_difference(a, i, i_end, b, j, j_end));
      i = i_end;
      j = j_end;
    }
  }
  return least;
}

}  // namespace frizzle::cache
