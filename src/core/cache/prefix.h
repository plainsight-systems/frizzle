#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/edit/edit.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::cache {

// Axis G: changes with a new cache or context mechanism.
//
// The diff that keys the KV cache on the token sequence. JavaScript sends the
// whole rendered conversation every turn and holds no cache state; this
// function finds how much of it the cache already holds.
//
// A pure function over two token sequences: no GPU, no model, no file. It is
// the most correctness-critical logic in the chat loop, and it is kept apart
// from the cache's storage so it is tested directly.
//
// The runtime truncates the cache to the returned length and prefills the
// rest. The cache may keep less than the common prefix: a rollback deeper
// than the rollback reserve empties it, and the runtime prefills from the
// first token. When the whole incoming sequence is cached, the runtime still
// recomputes its last token: generation needs that token's logits, and the
// cache holds keys and values, not logits.
//
// Attention is causal, so a token's keys and values depend only on the tokens
// before it and on the edits in force at those positions: everything before
// the first difference in either is exactly reusable, and everything from it
// on is recomputed, matching or not.
//
// Contract
//   - longest_common_prefix(cached, incoming): how many leading tokens the
//     two sequences share.
//   - first_edit_difference(cached, incoming): the least position at which
//     the edits in force differ, or edit::kToTheEnd when they never do. Both
//     are edit sets placed at positions (edit::placed): `cached`, the edits
//     the run that wrote the cache applied; `incoming`, the next run's. The
//     edits in force at a position are those whose range holds it, each
//     with its layer, neuron, kind and value's bits; two sets differ at a
//     position when those differ there, whatever order or split their
//     ranges have, so [3, 5) and [5, 9) of one edit are [3, 9) of it.
//   - The runtime reuses min(longest_common_prefix, first_edit_difference)
//     tokens, less one when that is the whole prompt (runtime.h); the
//     engine keeps the placed edits of the run that wrote the cache
//     (runtime/engine.h).
//
// Algorithm, first_edit_difference
//   Each set, sorted by (layer, neuron, first position): per neuron, a list
//   of ranges, each with its kind and value. Walk the two lists of each
//   neuron in step: the first position where one holds an edit and the
//   other does not, or both hold different kinds or value bits, is that
//   neuron's difference. The least over every neuron in either set is the
//   answer. A neuron in one set alone differs at its first range's start.
//
// What it costs, once per turn: first_edit_difference sorts a copy of each
// set, at most kMaxEdits records, and walks them once (CDSA.10): below a
// millisecond, an estimate. longest_common_prefix, one pass over at most the
// shorter sequence,
// comparing 4-byte identifiers — at most the context offered, 32,768 for
// Gemma 3 1B, 128 KiB read from each side. At one comparison a cycle that is
// about 12 µs, an estimate, not a measurement. What it saves grows with the
// tokens it reuses: each reused token is a row that every layer's
// projections, attention and feed-forward no longer compute — about 1.4
// GFLOP for Gemma 3 1B, counted from its projections and feed-forward, before
// attention, which grows with the context — and each 512 reused tokens a
// prefill pass that no longer reads the weights. Both sequences live in the
// module's memory when compared. The page sends text and, for a branch, the
// identifiers of the reply tokens it keeps (runtime/engine.h): a few kilobytes
// once a run, which the page holds because it holds the conversation, as it
// holds the edit set (logical-overview.md, invariant 4). WASM.2 is kept in its
// intent: one crossing a run, never one a token.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     ES.1   Prefer the standard library to handcrafted code — std::mismatch.
//     Per.2  Don't optimize prematurely — the diff's 8 bytes a compared
//            token are far below the work one reused token saves, so it is a
//            plain comparison, not a vectorized or hashed one.
//   C++ performance guidelines
//     GDSA.6 Account the bytes a stage moves — the count above.
//     WASM.2 Batch work across the JS boundary — the kept identifiers cross
//            once a run, in the run's own call.

[[nodiscard]] std::size_t longest_common_prefix(
    std::span<const tokenizer::TokenId> cached,
    std::span<const tokenizer::TokenId> incoming) noexcept;

[[nodiscard]] std::uint32_t first_edit_difference(
    std::span<const edit::Edit> cached, std::span<const edit::Edit> incoming);

}  // namespace frizzle::cache
