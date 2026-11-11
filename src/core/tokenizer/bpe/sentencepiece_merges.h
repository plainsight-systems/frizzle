#pragma once

#include <span>

#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// The merge rules a SentencePiece vocabulary implies. Its file lists none,
// only a score for each token, and SentencePiece BPE merges any two
// adjacent symbols whose text joined is a token, the highest score first
// and the leftmost on a tie (sentencepiece_bpe.h). These are that search's
// rules, found once at load rather than at every step, for the ordinary
// merge step (merge.h).
//
// Contract
//   - implied_merges(vocabulary, scores, out): Ok with `out` holding a rule
//     for each split of a normal token's text, between two characters, into
//     two normal tokens; the rule joins those two into the token, at the
//     rank of its score. Tokens of equal score share a rank, so the leftmost
//     pair wins their tie in the merge step.
//   - Refusals, `out` then untouched: CountMismatch, subject "scores", when
//     `scores` holds other than one score a token; InvalidScore, naming the
//     token, when a normal token's score is NaN, which cannot be ordered;
//     DuplicateMerge, subject "implied merges", if two rules joined the same
//     pair, which cannot happen (below).
//   - Only normal tokens take part. The symbols merged are characters'
//     tokens, byte tokens where a character has none, and the tokens merges
//     make; special tokens are matched before merging and never among them,
//     and a byte token's text ("<0x41>") is not the byte it stands for.
//
// Algorithm, for a vocabulary of V tokens
//   1. Collect the normal tokens, refusing a NaN score; sort them by score,
//      highest first. The rank starts at 0 and rises by one wherever the
//      score falls from the token before.
//   2. For each token in that order, and each byte position after its first
//      byte that begins a UTF-8 character: look up the text before and the
//      text after. When both are normal tokens, record the rule (left, right,
//      token) at the token's rank.
//   3. Build the merge table from the rules (merge_table.h). Each pair joins
//      to one text, and each text is one token, so no pair repeats.
//
// For Gemma 3 1B's pinned file: 255,474 normal tokens of 262,144, the longest
// 48 bytes; 513,511 implied merges.
//
// Design and cost
//   - Counted, once a load: one pass over the 262,144 types; a sort of the
//     255,474 normal tokens, 4 bytes each, 1 MiB; at most one split a byte
//     of each normal token, two vocabulary lookups each (vocabulary.h); the
//     513,511 rules, 16 bytes each, 7.8 MiB, before the table copies them
//     into its 16 MiB of slots (merge_table.h). The rules and the sorted
//     tokens are freed when it returns.
//   - The scores are compared as read; equal scores tie, so the ranks follow
//     the file's scores exactly, with no rounding.
//
// Verification
//   The tokenizer suite loads the pinned tokenizer, whose encodes run on
//   these rules, and checks every reference text's IDs. No test reaches a
//   refusal.
//
// Corpus guidance
//   C++ Core Guidelines
//     E.27     If you can't throw exceptions, use error codes systematically:
//              every refusal is a LoadResult naming its subject.
//     SL.io.2  When reading, always consider ill-formed input: the scores'
//              count and every normal token's score are checked before any
//              rule is made.
//     ES.45    Avoid "magic constants"; use symbolic constants: a byte
//              begins a character unless it lies in the continuation range,
//              named in unicode.h.
//   C++ performance guidelines
//     CDSA.32  Transform static data once into the layout its consumer
//              reads: the scores become rules once, at load, in the table
//              the merge step reads, never during an encode.

[[nodiscard]] LoadResult implied_merges(const Vocabulary& vocabulary,
                                        std::span<const float> scores,
                                        MergeTable& out);

}  // namespace frizzle::tokenizer::bpe
