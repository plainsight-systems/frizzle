#pragma once

#include <span>

#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// The merge rules a SentencePiece vocabulary implies. Its file lists none,
// only a score for each token, and llama.cpp merges any two adjacent symbols
// whose text joined is a token, the highest score first and the leftmost on
// a tie. These are that search's rules, found once at load rather than at
// every step: each split of a normal token's text, between two characters,
// into two normal tokens is a rule that makes it, ranked by its score. Tokens
// of equal score share a rank, so the leftmost wins their tie as it does there.
//
// Only normal tokens take part. The symbols merged are characters' tokens,
// byte tokens where a character has none, and the tokens merges make; special
// tokens are matched before merging and never among them, and a byte token's
// text ("<0x41>") is not the byte it stands for.
//
// `scores` holds a score for each token of `vocabulary`. It fails with
// CountMismatch if it holds another number, and with InvalidScore if a normal
// token's score is NaN, which cannot be ordered; the subject names the token.
// `out` is left untouched unless it succeeds.
[[nodiscard]] LoadResult implied_merges(const Vocabulary& vocabulary,
                                        std::span<const float> scores,
                                        MergeTable& out);

}  // namespace frizzle::tokenizer::bpe
