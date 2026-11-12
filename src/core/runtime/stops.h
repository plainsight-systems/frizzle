#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// The stop set: the tokens that end a turn when drawn. Made once at load
// from the file and the load policy; read once a draw, when the runtime
// learns the drawn token (runtime.h).
//
// Contract
//   - resolve_stops(index, vocabulary, policy_stops, out): Ok with `out`
//     holding every stop token, each once, in the order found; or the first
//     refusal, named in the result's subject, `out` untouched.
//   - From the file: tokenizer.ggml.eos_token_id, tokenizer.ggml.eot_token_id
//     and tokenizer.ggml.eom_token_id, each where present. Refused: a key
//     present but not an unsigned 32-bit integer (WrongType); an identifier
//     at or past the vocabulary (OutOfVocabulary).
//   - From the load policy (policy/policy.h, LoadPolicy::stop): further
//     stop tokens, by their text. Each must be exactly one token of the
//     vocabulary (else NotAToken) and a control token (else NotControl),
//     since a stop on an ordinary word would end replies at that word. A
//     text is checked against the vocabulary at load, so a wrong one is
//     refused by name rather than stopping on an unrelated identifier.
//   - Refused too: more than kMaxStops distinct tokens (TooMany); and none
//     at all (None), since a turn would then end only at its limit or the
//     context. A token named twice is held once.
//   - StopSet(tokens): precondition, not checked: the tokens are distinct
//     and at most kMaxStops; resolve_stops ensures both. A StopSet is a
//     value with no member that changes it; the default one is empty.
//   - contains(token): whether `token` is in the set. tokens(): the set, in
//     the order found, valid as long as the StopSet.
//   Precondition of resolve_stops: `vocabulary` was loaded from `index`'s
//   file.
//
// Algorithm, resolve_stops
//   1. For each of the three keys in order: absent, skip it; not an
//      unsigned 32-bit integer, WrongType; an identifier at or past the
//      vocabulary's size, OutOfVocabulary; otherwise add it unless already
//      found.
//   2. For each policy text in order: find its token, else NotAToken; not a
//      control token, NotControl; otherwise add it unless already found.
//   3. None found, None; more than kMaxStops, TooMany; otherwise `out` is
//      the set.
//   contains: a linear search of the held tokens, in order.
//
// For Gemma 3 1B's pinned file: tokenizer.ggml.eos_token_id is 106,
// <end_of_turn>, a control token, the token the chat template closes each
// turn with; the file has no eot or eom key. The load policy lists no stop
// (LoadPolicy's default, which every host passes), so the set holds one
// token. <eos>, identifier 1, is also a control token, and a policy could
// add it by its text.
//
// Design and cost
//   - Load, counted: three metadata reads, a vocabulary lookup for each
//     policy text, and a vector of the found tokens, freed when it returns.
//   - A draw, counted: at most kMaxStops comparisons of 4-byte identifiers,
//     one for Gemma 3 1B; nothing allocated (MEM.9).
//   - The set is held inline, kMaxStops identifiers and a count: 40 bytes
//     natively, 36 in the 32-bit browser build. 8 is a ceiling on what a
//     file and a policy name, stated, not measured across models (EMB.2).
//     So few keys are searched by a linear scan (CDSA.9).
//
// Verification
//   The replay scenario's turns run with the stop set made from the pinned
//   file (tests/README.md). No test checks resolve_stops on its own, reaches
//   one of its refusals, or passes a policy stop.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.2      Use class if the class has an invariant: StopSet's tokens are
//              distinct and at most kMaxStops.
//     E.27     If you can't throw exceptions, use error codes
//              systematically: StopsError, with the subject named.
//     SL.con.3 Avoid bounds errors: contains searches a span of the held
//              tokens, never past the count (stops.cpp).
//     Enum.7   Specify an enumeration's underlying type only when
//              necessary: StopsError names one byte, to save space.
//   C++ performance guidelines
//     EMB.2    Prefer fixed-capacity containers: the set, inline, its
//              capacity stated.
//     MEM.9    Allocate at init: made at load; a draw's test allocates
//              nothing.
//     CDSA.9   Pick the sorted-search layout by cache level: up to a few
//              dozen keys, a linear scan; here at most 8 identifiers.

inline constexpr std::size_t kMaxStops = 8;

class StopSet {
 public:
  StopSet() = default;  // empty: what resolve_stops fills
  // Precondition: the tokens are distinct and at most kMaxStops.
  explicit StopSet(std::span<const tokenizer::TokenId> tokens) noexcept;

  [[nodiscard]] bool contains(tokenizer::TokenId token) const noexcept;
  [[nodiscard]] std::span<const tokenizer::TokenId> tokens() const noexcept;

 private:
  std::array<tokenizer::TokenId, kMaxStops> tokens_{};
  std::size_t size_ = 0;
};

enum class StopsError : std::uint8_t {
  Ok,
  WrongType,        // a file's key present, but not an unsigned 32-bit integer
  OutOfVocabulary,  // a file's identifier at or past the vocabulary
  NotAToken,        // a policy text that is not exactly one token
  NotControl,       // a policy text naming a token that is not a control token
  TooMany,          // more than kMaxStops distinct tokens
  None,             // neither the file nor the policy names one
};

struct StopsResult {
  StopsError error = StopsError::Ok;
  std::string subject;  // the key or text refused, and why
};

// The stop set of `index`'s file and `policy_stops`, into `out`; `out` is
// left untouched on failure. Precondition: `vocabulary` was loaded from
// `index`'s file.
[[nodiscard]] StopsResult resolve_stops(
    const gguf::TensorIndex& index, const tokenizer::Vocabulary& vocabulary,
    std::span<const std::string> policy_stops, StopSet& out);

}  // namespace frizzle::runtime
