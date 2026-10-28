#pragma once

#include <memory>
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
//   - A turn: the text, UTF-8, is encoded whole — the chat template wrote
//     the special tokens as text, and they encode as their identifiers —
//     into tokens held for the generator's life, reserved at the context
//     offered plus one. Then the runtime's turn starts over them.
//   - A token covers at most the tokenizer's longest_cover(), L bytes, of
//     normalized text, and normalizing shrinks UTF-8 at most 7/2-fold — a
//     character NFC writes stands for at most its longest canonical
//     equivalent, at most 7/2 times its bytes in Unicode 16.0 — so text
//     longer than 7/2 × the context × L bytes makes more tokens than the
//     context holds: it is refused before it is encoded, as PromptTooLong,
//     naming its bytes and the least tokens they make.
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
//     a UTF-8 stream (tokenizer/detokenizer.h), and the characters it
//     completes are passed to the text callback; a token that completes
//     none — the first bytes of a character another token finishes — passes
//     nothing. Stop tokens are never emitted, so never decoded; any other
//     control token's text, as written, is, since a reader may take the
//     reply's structure from it.
//   - When the turn finishes, the stream is finished: bytes still pending,
//     the turn having stopped inside a character, are passed as U+FFFD
//     rather than dropped unseen. Then the end callback, with the runtime's
//     result.
//   - It owns the tokenizer and the runtime, and its callbacks carry its
//     state, held while a turn runs, as the runtime's are: destroying a
//     generator with a turn running destroys the runtime, which finishes the
//     turn Cancelled (runtime.h), and its end callback is still called.
//   - rewind(tokens): the runtime's rewind (runtime.h), between turns;
//     refused, false, while a turn runs. The end callback runs after the
//     turn has ended, so it may rewind.
//
// Algorithm, a turn
//   1. Refuse text too long to fit, by its bytes; encode it, with the
//      context as the limit; start the runtime's turn over the tokens.
//   2. Each emitted token: decode it, push its bytes through the stream,
//      and pass the characters it completes to the text callback.
//   3. The end: finish the stream, passing what is pending as U+FFFD; then
//      the end callback with the runtime's result.
//
// Design and cost
//   - A turn, once: the text's bytes encoded, 28.8 to 30.4 ns a byte for
//     byte-level BPE with its piece cache warm, 43.3 for SentencePiece (their
//     headers) — 4.6 to 5.6 ms for a conversation of a listed model's whole
//     context — then the runtime's turn. The encoders allocate their working
//     lists afresh each encode: for each span between special tokens its
//     normalized text and its pieces, and for each piece too long for the
//     piece cache its symbols and their merge queue — allocations a turn in
//     proportion to the conversation's spans and long pieces, beside the
//     turn's prefill, none a token.
//   - A token: one decode, an indirect call that appends the token's bytes
//     to a string reserved at load for the vocabulary's longest token; the
//     stream's push over those bytes; and, when they complete a character,
//     one text callback — the boundary's one crossing a token (WASM.2).
//     Nothing is allocated a token.
//
//   - Verification: Charlotte's generator tests were not ported. In Frizzle
//     the tokenizer suite (tests/contracts/tokenizer/) locks the encoding,
//     and the replay and edits suite, planned (tests/README.md), runs turns
//     through the engine; until it is built, only the native tool exercises
//     the streaming.
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

// Each piece of the reply's text, whole characters, in order. Valid only
// during the call.
using TextCallback = void (*)(std::string_view text, void* userdata);
// Once a turn, after its last text.
using EndCallback = void (*)(const TurnResult& result, void* userdata);

// Why a turn did not start: the text, or the runtime.
// A turn's start: the encode's refusal, if the tokenizer refused the text —
// never TooManyTokens, which is the runtime's PromptTooLong, in `start` — or
// the runtime's start.
struct GenerateResult {
  tokenizer::EncodeResult encode;
  StartResult start;
  [[nodiscard]] bool ok() const noexcept {
    return encode.error == tokenizer::EncodeError::Ok &&
           start.error == StartError::Ok;
  }
};

class Generator {
 public:
  // Preconditions: `tokenizer` is the model's, and `runtime` was made over
  // its program and cache (runtime.h).
  Generator(std::unique_ptr<tokenizer::Tokenizer> tokenizer,
            std::unique_ptr<Runtime> runtime);

  // Starts a turn over `text`, the whole rendered conversation. On ok(),
  // `on_text` is called for each piece of the reply and `on_end` once,
  // both with `userdata`; otherwise neither is. `text` is not read after
  // the call.
  [[nodiscard]] GenerateResult start(std::string_view text,
                                     const policy::TurnPolicy& policy,
                                     TextCallback on_text, EndCallback on_end,
                                     void* userdata);

  // Ends the running turn at its next report (runtime.h); with none,
  // nothing.
  void cancel() noexcept;

  // The runtime's rewind (runtime.h): between turns, keeps at most the
  // first `tokens` of the cache. False, changing nothing, while a turn runs.
  [[nodiscard]] bool rewind(std::uint32_t tokens) noexcept;

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
