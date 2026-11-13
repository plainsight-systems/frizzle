// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Contract, algorithm, cost and corpus guidance: detokenizer.h. This file
// defines what that header states.

#include "core/tokenizer/detokenizer.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer {

Detokenizer::Detokenizer(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {
  const Vocabulary& vocabulary = tokenizer.vocabulary();
  for (std::size_t id = 0; id < vocabulary.size(); ++id) {
    longest_ =
        std::max(longest_, vocabulary.text(static_cast<TokenId>(id)).size());
  }
  bytes_.reserve(longest_);
  // At the worst each of a token's bytes is one the stream cannot place,
  // written as U+FFFD's three bytes; the up to three held before them leave
  // as themselves, completed, or as one U+FFFD for all of them: 3 more.
  text_.reserve(3 * longest_ + 3);
}

std::string_view Detokenizer::push(TokenId token) {
  bytes_.clear();
  tokenizer_->decode(token, bytes_);
  text_.clear();
  stream_.push(bytes_, text_);
  return text_;
}

std::string_view Detokenizer::finish() {
  text_.clear();
  (void)stream_.finish(text_);
  return text_;
}

void texts(const Tokenizer& tokenizer, std::span<const TokenId> tokens,
           std::string& pool, std::vector<std::uint32_t>& offsets) {
  offsets.clear();
  offsets.reserve(tokens.size() + 1);
  offsets.push_back(static_cast<std::uint32_t>(pool.size()));
  Utf8Stream stream;
  std::string bytes;
  for (const TokenId token : tokens) {
    bytes.clear();
    tokenizer.decode(token, bytes);
    stream.push(bytes, pool);
    offsets.push_back(static_cast<std::uint32_t>(pool.size()));
  }
  // What finishing gives (U+FFFD for pending bytes) is the last token's.
  (void)stream.finish(pool);
  offsets.back() = static_cast<std::uint32_t>(pool.size());
}

}  // namespace frizzle::tokenizer
