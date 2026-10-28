#pragma once

#include <span>
#include <vector>

#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// The merge step of BPE, for one run of symbols: in byte-level BPE a piece of
// split text, one token for each of its bytes; in SentencePiece BPE a stretch
// of text between special tokens, one token for each of its characters.
//
// Merging repeatedly finds the adjacent pair whose rule ranks lowest and
// replaces the two with the token the rule makes. When pairs tie — the same
// pair more than once, or two pairs whose rules share a rank — the leftmost
// merges first, and a pair that shares a token with one already merged no
// longer exists. Merging stops when no adjacent pair has a rule. The tokens
// left are appended to `out`, after whatever it already holds.
//
// A run can be long: a word of 10,000 letters is one piece, and SentencePiece
// merges a whole paragraph at once. The cost must not grow with the square of
// the run's length.
//
// Optimization (practice): two strategies by length, as tiktoken and Hugging
// Face tokenizers both choose. A short piece is rescanned, its pair ranks in
// fixed arrays on the stack; a long one goes through a min-heap over a chain
// of symbols, O(n log n), which bounds the cost of a long or hostile piece.
void merge(std::span<const TokenId> symbols, const MergeTable& merges,
           std::vector<TokenId>& out);

}  // namespace frizzle::tokenizer::bpe
