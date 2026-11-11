#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/policy/policy.h"
#include "core/runtime/runtime.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// A loaded model as the engine uses it: the rendered conversation in, the
// reply's text out. The runtime (runtime.h) works on identifiers alone; this
// puts the model's tokenizer (tokenizer.h) on either side of it, so the
// engine (engine.h) and the hosts above it handle text only.
//
// Contract
//   - A turn, in two calls, so the engine can place a run's edits, which
//     needs the text's length in tokens, before the turn starts:
//       prepare(text, kept): the text, UTF-8, is encoded whole — the chat
//         template wrote the special tokens as text, and they encode as
//         their identifiers — into tokens held for the generator's life,
//         reserved at the context offered plus one; then `kept`, reply
//         token identifiers an earlier turn emitted, is appended as given.
//         Returns the text's tokens and the whole input's. An identifier
//         at or past the vocabulary is refused, naming its index in
//         `kept`; nothing else of `kept` is checked: the model reads what
//         it is given.
//       start(policy, on_token, on_end): the runtime's turn over the
//         prepared input. Refused as EmptyPrompt, "no turn is prepared",
//         without one prepared since the last start: there is no prompt to
//         run. A refused prepare leaves none prepared.
//   - A token covers at most the tokenizer's longest_cover(), L bytes, of
//     normalized text, and normalizing never shrinks the text
//     (tokenizer.h), so text longer than the context × L bytes makes more
//     tokens than the context holds. The check allows 7/2 of that, looser
//     than it needs and still sound: text longer than 7/2 × the context × L
//     bytes is refused before it is encoded, as PromptTooLong, naming its
//     bytes and a least count of tokens; shorter text too long to fit is
//     refused by the encode, below.
//   - Shorter text is encoded with the context offered as its limit
//     (tokenizer/tokenizer.h, encoding is bounded): text that makes more
//     tokens at least, its specials and segments counted after normalizing,
//     is refused before any piece is merged, as PromptTooLong with that
//     count, as the runtime refuses one, so the page shortens the
//     conversation alike; text past kMaxEncodeBytes once normalized, as the
//     encode's TooLong, the result carrying the tokenizer's EncodeResult,
//     its normalized bytes with it, which the boundary sends as
//     "prompt-too-large" with the limit; and the memory an encode takes stays
//     within what tokenizer.h counts for kMaxEncodeBytes, whatever the
//     text. Admitted text that still makes more tokens than the context
//     grows the tokens once, and the runtime refuses it, naming both counts.
//   - Refused at once, without a callback: text the tokenizer cannot encode,
//     named; text over those bounds; and whatever the runtime refuses
//     (runtime.h).
//   - Each token the runtime emits is decoded into bytes and pushed through
//     a UTF-8 stream (tokenizer/detokenizer.h), and the token callback is
//     called once for it, with its identifier and the characters it
//     completes: empty for a token that completes none — the first bytes
//     of a character another token finishes, which owns that character's
//     text. So a host counts tokens by callbacks and maps text to them.
//     Stop tokens are never emitted, so never decoded; any other control
//     token's text, as written, is, since a reader may take the reply's
//     structure from it. The kept tokens are input, never emitted, and pass
//     no callback: their text is the host's already.
//   - When the turn finishes, the stream is finished: bytes still pending,
//     the turn having stopped inside a character, are passed as U+FFFD in
//     the result's `tail`, owned by the last token emitted, which a host
//     appends to that token's text, rather than dropped unseen; otherwise
//     the tail is empty. Then the end callback, with the runtime's result
//     and the tail. So the token callbacks are exactly the emitted tokens,
//     one each, and a host counts them by callbacks.
//   - It owns the tokenizer and the runtime, and its callbacks carry its
//     state, held while a turn runs, as the runtime's are: destroying a
//     generator with a turn running destroys the runtime, which finishes the
//     turn Cancelled (runtime.h), and its end callback is still called.
//   - rewind(tokens): the runtime's rewind (runtime.h), between turns;
//     refused, false, while a turn runs. The end callback runs after the
//     turn has ended, so it may rewind.
//
// Algorithm, a turn
//   1. prepare: refuse text too long to fit, by its bytes; encode it, with
//      the context as the limit; check and append the kept identifiers.
//   2. start: the runtime's turn over the tokens.
//   3. Each emitted token: decode it, push its bytes through the stream,
//      and pass its identifier and the characters it completes to the
//      token callback.
//   4. The end: finish the stream; the end callback with the runtime's
//      result, what was pending as U+FFFD its tail.
//
// Design and cost
//   - A turn, once: the text encoded, at the cost SentencePiece BPE counts
//     (bpe/sentencepiece_bpe.h), then the runtime's turn. The encode
//     allocates its working lists afresh each time: the escaped copy, the
//     segments, and for each long run its merge's symbols and heap
//     (bpe/merge.h) — allocations a turn in proportion to the conversation,
//     beside the turn's prefill, none a token. Its time is not measured.
//   - A token: one decode, an indirect call that appends the token's bytes
//     to a string reserved at load for the vocabulary's longest token; the
//     stream's push over those bytes; and one token callback — the
//     boundary's one crossing a token (WASM.2).
//     Nothing is allocated a token.
//
//   - Verification: the tokenizer suite (tests/contracts/tokenizer/) locks
//     the encoding, and the replay and edits scenario
//     (tests/integration/replay/, tests/README.md) runs turns through the
//     engine and compares their text and identifiers.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer: the tokenizer and the
//            runtime owned, by unique_ptr.
//     E.27   Use error codes systematically: GenerateResult.
//   C++ performance guidelines
//     WASM.2 Batch work across the JS boundary: the conversation crosses
//            once a turn, as bytes; text, once a token that completes some.
//     WASM.4 Reduce indirect dispatch in hot paths: one decode a token.
//     MEM.9  Allocate at init: the tokens, the decode buffer and the stream,
//            at load; a token's path allocates nothing.

// Each emitted token, in order: its identifier and the whole characters it
// completes, possibly none. Valid only during the call.
using ReplyTokenCallback = void (*)(tokenizer::TokenId token,
                                    std::string_view text, void* userdata);
// Once a turn, after its last text.
using EndCallback = void (*)(const TurnResult& result, void* userdata);

// Why a turn did not start: the text, or the runtime.
// A turn's start: the encode's refusal, if the tokenizer refused the text —
// never TooManyTokens, which is the runtime's PromptTooLong, in `start` — or
// the runtime's start.
struct GenerateResult {
  tokenizer::EncodeResult encode;
  StartResult start;
  // The index in `kept` of the first identifier at or past the vocabulary.
  std::optional<std::size_t> kept_out_of_vocabulary;
  [[nodiscard]] bool ok() const noexcept {
    return encode.error == tokenizer::EncodeError::Ok &&
           start.error == StartError::Ok && !kept_out_of_vocabulary;
  }
};

// What prepare made: the text's tokens, and the input's, the kept tokens
// after them.
struct Prepared {
  std::uint32_t text_tokens = 0;
  std::uint32_t input_tokens = 0;
};

class Generator {
 public:
  // Preconditions: `tokenizer` is the model's, and `runtime` was made over
  // its program and cache (runtime.h).
  Generator(std::unique_ptr<tokenizer::Tokenizer> tokenizer,
            std::unique_ptr<Runtime> runtime);

  // Prepares a turn over `text`, the whole rendered conversation, then
  // `kept`. On ok(), what it made; otherwise the refusal, and nothing is
  // prepared. Neither is read after the call. Refused while a turn runs.
  [[nodiscard]] GenerateResult prepare(std::string_view text,
                                       std::span<const tokenizer::TokenId> kept,
                                       Prepared& made);

  // Starts the prepared turn. On ok(), `on_token` is called for each
  // emitted token and `on_end` once, both with `userdata`; otherwise
  // neither is.
  [[nodiscard]] GenerateResult start(const policy::TurnPolicy& policy,
                                     ReplyTokenCallback on_token,
                                     EndCallback on_end, void* userdata);

  // Ends the running turn at its next report (runtime.h); with none,
  // nothing.
  void cancel() noexcept;

  // The runtime's rewind (runtime.h): between turns, keeps at most the
  // first `tokens` of the cache. False, changing nothing, while a turn runs.
  [[nodiscard]] bool rewind(std::uint32_t tokens) noexcept;

  // The positions the cache holds (runtime.h); 0 once destroyed.
  [[nodiscard]] std::uint32_t cached() const noexcept;

#if FRIZZLE_DIAGNOSTICS_ENABLED
  // The runtime's step observer (runtime.h): false, changing nothing,
  // while a turn runs.
  bool observe_steps(StepObserver observer, void* userdata) noexcept;
#endif

  ~Generator();
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;
  Generator(Generator&&) = delete;
  Generator& operator=(Generator&&) = delete;

  // What the generator and its in-flight callbacks share; defined only
  // where the generator is.
  struct State;

 private:
  std::shared_ptr<State> state_;  // shared with in-flight callbacks
};

}  // namespace frizzle::runtime
