#pragma once

#include <span>
#include <vector>

#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer::bpe {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// The merge step of BPE, for one run of symbols: a stretch of text between
// special tokens, cut before a ▁ where no merge could cross it
// (sentencepiece_bpe.h), one token for each of its characters, or one byte
// token for each byte of a character with no token.
//
// Contract
//   - merge(symbols, merges, out): appends to `out`, after whatever it
//     already holds, the tokens the run merges to. Nothing is appended for
//     an empty run.
//   - Merging repeatedly finds the adjacent pair whose rule ranks lowest and
//     replaces the two with the token the rule makes. When pairs tie, the
//     same pair more than once or two pairs whose rules share a rank, the
//     leftmost merges first, and a pair that shares a token with one already
//     merged no longer exists. Merging stops when no adjacent pair has a
//     rule.
//   - Both strategies below give the same tokens for the same run; which
//     one runs is not visible in the result.
//   - Preconditions, not checked: the run has fewer than 2^32 − 1 symbols;
//     encoding admits at most kMaxEncodeBytes, 512 KiB, of normalized text
//     (tokenizer.h), so a run holds at most 524,288.
//   - Allocates only for a long run, and `out`'s growth; no failure is
//     reported.
//
// Algorithm, for n symbols
//   Short, n <= 64:
//     1. Copy the symbols into a fixed array; look up the rule of each of
//        the n − 1 adjacent pairs, its rank (2^32 − 1 for none) and result.
//     2. Find the lowest rank, the first on a tie; if it is 2^32 − 1, stop.
//     3. The left symbol becomes the result; the symbols, ranks and results
//        after the pair move down one place, closing the gap; look up again
//        only the pairs on either side of the new token. Back to 2.
//     4. Append the symbols left.
//   Long, n > 64:
//     1. Link the symbols in a chain by position, each to its neighbours.
//     2. Push every adjacent pair with a rule onto a min-heap of candidates,
//        ordered by rank, then by left position.
//     3. Pop the top. If its two symbols are no longer linked to each other
//        or no longer hold the tokens they held when it was pushed, drop it.
//        Otherwise the left symbol takes the result and the right one's
//        place in the chain, the right one is unlinked, and the two pairs
//        that include the new token are pushed if they have rules. Repeat
//        until the heap is empty.
//     4. Walk the chain from the first symbol, which is never the right of a
//        pair, and append its tokens.
//
// Design and cost
//   - Optimization (practice): two strategies by length. For a few symbols
//     a scan of adjacent ranks beats a heap, and a short run allocates
//     nothing: its three arrays of 64 entries, 4 bytes each, 768 bytes, are
//     on the stack, zeroed on each call (ES.20). Rescanning costs at most
//     n − 2 rank comparisons a merge and n − 1 merges, under 64 × 64 for a
//     short run. A
//     long run costs O(n log n): at most n − 1 merges, each a pop and at
//     most two pushes, which bounds the cost of a long or hostile run. The
//     threshold of 64 rests on a timing made outside Frizzle: the rescan
//     about 20% faster than the heap from 3 to 32 symbols, still faster at
//     64, slower beyond; an estimate, not measured in Frizzle.
//   - A run is long when its text holds no ▁ a merge cannot cross for more
//     than 64 characters, such as 10,000 letters with no space, or a stretch
//     of characters with no token, each spelled in byte tokens. Gemma 3's
//     runs are cut a word at a time where they can be (sentencepiece_bpe.h),
//     so a long one needs a word of more than 64 symbols; how often prose
//     has one is not counted.
//   - Counted, a long run of n symbols: the chain, 12 bytes a symbol (a
//     token and two positions); the heap, 24 bytes a candidate, at most
//     n − 1 pushed at the start and two a merge after; the worst case over
//     an encode's whole merging stage, the run's symbols and the tokens
//     included as well as the chain and heap, is counted in tokenizer.h, at
//     most 88 MiB for 512 KiB of normalized text.
//
// Verification
//   The tokenizer suite encodes every reference text, which runs the merge
//   on Gemma 3's runs; no test sets the two strategies side by side or
//   gives a run longer than 64 symbols on purpose.
//
// Corpus guidance
//   C++ performance guidelines
//     CDSA.16  Default priority queue: an array-backed heap without
//              decrease-key: std::priority_queue over a vector, and a pair
//              changed by a merge is left in the heap and dropped when
//              popped, rather than updated in place.
//     MEM.9    Allocate at init, not in steady state: a short run, most of
//              Gemma 3's, allocates nothing; a long run allocates its chain
//              and heap for that encode, once a turn.
//     EMB.5    Budget the stack: the short run's 768 bytes of locals are
//              above the rule's guide of about 256 bytes, a stated
//              exception: they are about 1% of the browser module's stack,
//              Emscripten's default of 64 KiB as the build sets none, and
//              the merge does not recurse.
//   C++ Core Guidelines
//     ES.20    Always initialize an object: the short run's arrays are
//              zeroed, though a result is read only where a rank was found.
//     SL.con.3 Avoid bounds errors: the arrays are reached through spans
//              and subspans whose bounds follow from n <= 64.
//     CDSA.21  Put a measured, named, injectable threshold on every algorithm
//              switch: kShortRun is named, and both strategies give the same
//              tokens. Departures: it was timed outside Frizzle (above), it
//              is a constant, not passed in, and no test forces each side.

void merge(std::span<const TokenId> symbols, const MergeTable& merges,
           std::vector<TokenId>& out);

}  // namespace frizzle::tokenizer::bpe
