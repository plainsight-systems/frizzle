#include "core/tokenizer/bpe/sentencepiece_merges.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace frizzle::tokenizer::bpe {
namespace {

// Whether a split of UTF-8 text before this byte falls between two characters.
bool starts_character(char byte) noexcept {
  return (static_cast<unsigned char>(byte) & 0xC0) != 0x80;
}

}  // namespace

LoadResult implied_merges(const Vocabulary& vocabulary,
                          std::span<const float> scores, MergeTable& out) {
  if (scores.size() != vocabulary.size())
    return {LoadError::CountMismatch, "scores"};
  const auto score = [&](TokenId t) {
    return scores[static_cast<std::size_t>(t)];
  };
  const auto normal = [&](TokenId t) {
    return vocabulary.type(t) == TokenType::Normal;
  };

  std::vector<TokenId> by_score;
  for (std::size_t id = 0; id < vocabulary.size(); ++id) {
    const auto token = static_cast<TokenId>(id);
    if (!normal(token)) continue;
    if (std::isnan(score(token)))
      return {LoadError::InvalidScore, std::string(vocabulary.text(token))};
    by_score.push_back(token);
  }
  // Highest score first; the rank rises only where the score falls.
  std::sort(by_score.begin(), by_score.end(),
            [&](TokenId a, TokenId b) { return score(a) > score(b); });

  std::vector<RankedRule> rules;
  std::uint32_t rank = 0;
  for (std::size_t i = 0; i < by_score.size(); ++i) {
    const TokenId token = by_score[i];
    if (i > 0 && score(token) != score(by_score[i - 1])) ++rank;
    const std::string_view text = vocabulary.text(token);
    for (std::size_t at = 1; at < text.size(); ++at) {
      if (!starts_character(text[at])) continue;
      const auto left = vocabulary.find(text.substr(0, at));
      if (!left || !normal(*left)) continue;
      const auto right = vocabulary.find(text.substr(at));
      if (!right || !normal(*right)) continue;
      rules.push_back({{*left, *right, token}, rank});
    }
  }
  // Each pair joins to one text, and each text is one token, so no pair
  // repeats.
  auto table = MergeTable::from_ranked_rules(rules);
  if (!table) return {LoadError::DuplicateMerge, "implied merges"};
  out = std::move(*table);
  return {};
}

}  // namespace frizzle::tokenizer::bpe
