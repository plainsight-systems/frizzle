// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Contract, algorithm, cost and corpus guidance: vocabulary.h. This file
// defines what that header states.

#include "core/tokenizer/vocabulary.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "core/tokenizer/load.h"

namespace frizzle::tokenizer {
namespace {

// FNV-1a, 64-bit: short, and well spread for keys of a few bytes, as token
// texts are. The slot comes from its top bits after mixing, the tag from its
// low bits as they are.
constexpr std::uint64_t kFnvOffsetBasis = 0xCBF29CE484222325;  // FNV-1a's start
constexpr std::uint64_t kFnvPrime = 0x100000001B3;  // FNV-1a's 64-bit prime

std::uint64_t text_hash(std::string_view text) noexcept {
  std::uint64_t hash = kFnvOffsetBasis;
  for (const char c : text) {
    hash = (hash ^ static_cast<unsigned char>(c)) * kFnvPrime;
  }
  return hash;
}

// The fewest slots a table has (vocabulary.h).
constexpr std::size_t kMinSlots = 16;

}  // namespace

std::optional<TokenId> Vocabulary::find(std::string_view text) const noexcept {
  if (slots_.empty()) return std::nullopt;
  const std::uint64_t hash = text_hash(text);
  const auto tag = static_cast<std::uint32_t>(hash);
  const std::size_t mask = slots_.size() - 1;
  for (std::size_t at = first_slot(hash);; at = (at + 1) & mask) {
    const Slot& slot = slots_[at];
    if (slot.token == kFree) return std::nullopt;
    const auto token = static_cast<TokenId>(slot.token);
    if (slot.tag == tag && this->text(token) == text) return token;
  }
}

LoadResult load_vocabulary(gguf::ByteSource& source,
                           const gguf::TensorIndex& index, Vocabulary& out) {
  constexpr std::string_view kTokens = "tokenizer.ggml.tokens";
  constexpr std::string_view kTypes = "tokenizer.ggml.token_type";

  Vocabulary vocabulary;
  if (auto r = read_string_array(source, index, kTokens, vocabulary.tokens_);
      !r.ok()) {
    return r;
  }
  std::vector<std::int32_t> types;
  if (auto r = read_int32_array(source, index, kTypes, types); !r.ok()) {
    return r;
  }
  if (types.size() != vocabulary.tokens_.size()) {
    return {LoadError::CountMismatch, std::string(kTypes)};
  }

  vocabulary.types_.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (types[i] < 0 || types[i] > static_cast<std::int32_t>(TokenType::Byte)) {
      return {LoadError::UnknownTokenType, "token " + std::to_string(i)};
    }
    vocabulary.types_.push_back(static_cast<TokenType>(types[i]));
  }

  // The smallest power of two that keeps the table at most three quarters full.
  std::size_t slots = kMinSlots;
  unsigned bits = std::countr_zero(kMinSlots);
  while (slots * 3 < types.size() * 4) {
    slots *= 2;
    ++bits;
  }
  vocabulary.slots_.assign(slots, Vocabulary::Slot{Vocabulary::kFree, 0});
  vocabulary.shift_ = std::numeric_limits<std::uint64_t>::digits - bits;
  const std::size_t mask = slots - 1;
  for (std::size_t i = 0; i < types.size(); ++i) {
    const std::string_view text = vocabulary.text(static_cast<TokenId>(i));
    const std::uint64_t hash = text_hash(text);
    const auto tag = static_cast<std::uint32_t>(hash);
    std::size_t at = vocabulary.first_slot(hash);
    for (; vocabulary.slots_[at].token != Vocabulary::kFree;
         at = (at + 1) & mask) {
      const Vocabulary::Slot& slot = vocabulary.slots_[at];
      if (slot.tag == tag &&
          vocabulary.text(static_cast<TokenId>(slot.token)) == text) {
        return {LoadError::DuplicateToken, std::string(text)};
      }
    }
    vocabulary.slots_[at] = {static_cast<std::uint32_t>(i), tag};
  }

  out = std::move(vocabulary);
  return {};
}

}  // namespace frizzle::tokenizer
