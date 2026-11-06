#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/cache/kv.h"
#include "core/kernels/program.h"
#include "core/policy/policy.h"
#include "core/runtime/stops.h"
#include "core/runtime/turn.h"
#include "core/tokenizer/tokenizer.h"

namespace frizzle::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// The turn loop. A model loaded is a program (kernels/program.h), its cache
// (cache/kv.h) and its stop set (stops.h); a turn takes the whole rendered
// conversation as tokens and streams the reply's tokens back. Tokens in,
// tokens out: encoding the prompt and decoding the reply into text are the
// boundary's, with the tokenizer (tokenizer/tokenizer.h), so the loop is
// tested on identifiers alone.
//
// Contract
//   - start(prompt, policy, on_token, on_turn): a turn over the prompt, the
//     whole rendered conversation. Refused at once, without a callback, when
//     it cannot run (StartError): a turn still running; a lost device; an
//     empty prompt; a prompt longer than the context offered, naming both
//     counts so the page can drop its oldest messages and render again
//     (logical-overview.md); or settings check refuses (sampler/sampler.h).
//     Otherwise each emitted token is passed to on_token and the turn's
//     result to on_turn, once. A max_tokens of 0 inspects: the prompt is
//     run and kept, nothing is emitted, and the turn ends Limit (turn.h).
//   - Invariant, between turns: the tokens held are exactly the cache's
//     length, those whose keys and values it holds. During a turn they also
//     hold the prompt's uncached tokens and the accepted draws.
//   - rewind(tokens), between turns: keeps at most the first `tokens` of the
//     cache and the held tokens together, through the cache's truncate, so a
//     rollback past the ring's reserve empties both (cache/kv.h). It never
//     lengthens them. Refused, false and changing nothing, while a turn
//     runs; the turn callback runs after the turn has ended, so it may
//     rewind. The engine rewinds before a run to the first position whose
//     edits differ from the cached run's (runtime/engine.h).
//   - Cancel ends the running turn at its next report; the steps already run
//     report first, and are discarded (turn.h). A cancel with no turn running
//     does nothing.
//   - Failure is visible, and leaves the cache consistent: a step's failure
//     empties it, so the next turn prefills from the first token; a
//     non-finite draw ends the turn NonFinite, the cache kept up to it; a lost
//     device refuses every later turn.
//   - Callbacks run inside the program's, on the thread that processes the
//     device's events. The token callback may cancel; the turn callback may
//     start the next turn, rewind, or destroy the runtime.
//   - It owns the program, and so outlives none of the program's callbacks:
//     they carry the runtime's state, held by those in flight as the
//     program's is, so destroying a runtime with a turn running finishes that
//     turn Cancelled, its callback called. The upload the program borrows
//     outlives the runtime (I.11).
//   - Diagnostic builds only (core/diagnostics.h, TLM.6): observe_steps sets
//     a step observer, or clears it, between turns. While one is set every
//     step runs profiled — kernels/program.h's run_profiled, through all its
//     launches — timed by the GPU at its one pass's beginning and end and
//     reading back its record as run() does, so the turn's tokens are the
//     same. The observer is called with the reporting step's kind, position
//     and tokens and its two GPU times, after the next steps are run. The
//     runtime reads no clock: a caller wanting the CPU's time of each report
//     reads its own clock in the observer, and the two clocks are never
//     subtracted (TLM.11). The clean build has none of it.
//
// Algorithm, a turn
//   1. The checks above.
//   2. The diff: the longest common prefix of the tokens the cache holds and
//      the prompt (cache/prefix.h), less one when it is the whole prompt —
//      the cache holds keys and values, not logits, and the first draw needs
//      the last prompt token's. The cache is truncated to it and keeps it, or
//      empties, for a ring rolled back past its reserve (cache/kv.h).
//   3. Steps, as the turn plans them (turn.h): the prompt's uncached tokens
//      in prefill blocks, then decode steps fed on the GPU. Each step run
//      advances the cache by its tokens, and copies a prefill block's
//      identifiers into the step; the turn's settings and seed are written
//      into the step once (sampler::apply).
//   4. As each step reports, in the order run: the turn decides its draw;
//      the next steps it plans are run; then an emitted draw's token is
//      passed to the token callback. The next step goes to the GPU before the
//      CPU's work on the token (GPU.7), and so a cancel from the token
//      callback finds that step queued — the case the cache must hold.
//   5. Finished: the cache is truncated to what the turn keeps, and the
//      held tokens to match; then the turn callback, once, with how it ended.
//
// Design and cost
//   - The turn's decisions are turn.h's pure functions; this sequences the
//     program and the cache around them (F.8).
//   - A decoded token, on the CPU: one report from the program, the turn's
//     constant work, at most kMaxStops comparisons, one run of the program —
//     17 calls into WebGPU and 2 or 3 a launch, and 48 bytes written
//     (program.h) — 4 bytes appended to the held tokens, and one token
//     callback. Nothing waits on the GPU: the token is read a step behind,
//     while the next step runs.
//   - A turn, once: the diff, one comparison a held token, at most the
//     context offered (cache/prefix.h); the cache's truncation, a visit to
//     each layer, twice; the prompt's uncached identifiers copied once into
//     the held tokens and once into the steps, 4 bytes each; and a prefill
//     step for each kPrefillBlock of them.
//   - A rewind: one truncation, a visit to each layer, between turns.
//   - Its first token waits on the prefill steps and the last one's map;
//     every later token arrives a step after it was drawn. A stop is
//     reported once any step queued behind it has run — at most one; none at
//     the limit or the context. A cancel waits for what remains of every
//     step already run, at most two, since the next step is run before the
//     token callback that may cancel.
//   - Profiling perturbs: each step's pass also writes two timestamps, which
//     the step resolves, copies, maps, reads and unmaps — five more calls
//     into WebGPU, 22 with run()'s 17 (kernels/program.h), and one more
//     callback, which the step's report waits on.
//   - Nothing is allocated after construction: the held tokens are reserved
//     at the context offered plus one, the step is the runtime's own, the
//     turn is a value, and a failure's message is reserved (MEM.9).
//   - What it asks of the other contracts: TurnPolicy's max_tokens, 0 an
//     inspection, bounded first by the context offered, and LoadPolicy's
//     stop texts (policy/policy.h, stops.h); in diagnostic builds the
//     program's launch count, the profiled step's bound; and the cache after
//     a step queued behind a turn's end keeps each position whose input
//     token is accepted (turn.h), where sampler/sampler.h said the queued
//     step's position is truncated.
//   - Verification: the replay and edits scenario (tests/integration/replay/,
//     tests/README.md) runs turns through the engine, edited and not,
//     against a fresh engine's; the turn suite (tests/contracts/turn/)
//     locks the inspection, a turn with a limit of 0. No test locks the
//     other ways a turn ends on their own.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.8    Prefer pure functions: the turn's decisions are turn.h's; this
//            sequences the program and the cache around them.
//     E.27   Use error codes systematically: StartError, and TurnFailure.
//     I.11   Never transfer ownership by a raw pointer: the program is
//            owned, by unique_ptr; the upload borrowed, and stated so.
//     I.5    State preconditions: on construction and on start.
//   C++ performance guidelines
//     GPU.7  Pipeline CPU and GPU work: two steps in flight; the next run
//            before the CPU's work on a token.
//     TLM.6  Label measurement modes: the step observer is diagnostic
//            builds' alone.
//     TLM.11 Never align clocks without a calibrated pair: GPU times out,
//            the caller's CPU clock its own.
//     GPU.1  Keep data on the device: the token is fed on the GPU, and only
//            its 16-byte record is read back.
//     MEM.9  Allocate at init: every buffer the turn touches, at
//            construction.
//     WASM.2 Batch work across the JS boundary: the prompt crosses once a
//            turn; a token, once.

enum class StartError : std::uint8_t {
  Ok,
  Busy,        // a turn is running
  DeviceLost,  // a step reported the device lost; no turn can run
  EmptyPrompt,
  PromptTooLong,  // the subject names the prompt's tokens and the context
                  // offered
  Settings,       // the subject names the setting and its range
};

struct StartResult {
  StartError error = StartError::Ok;
  std::string subject;
  // For PromptTooLong, as numbers, so the page can shorten the
  // conversation by the excess: the prompt's tokens — or, for text refused
  // before it is encoded, the fewest it makes — and the context offered.
  std::uint64_t prompt_tokens = 0;
  std::uint32_t context = 0;
};

struct TurnResult {
  TurnFailure failure;
  TurnEnd end;               // meaningful when failure is None
  std::string_view message;  // WebGPU's, for a step's failure
  std::uint32_t prompt;      // the prompt's tokens
  std::uint32_t reused;      // of those, the ones the cache already held
  std::uint32_t emitted;     // the reply's tokens passed to the token callback
  std::optional<tokenizer::TokenId>
      stop;  // the stop token drawn, for TurnEnd::Stop
  // Text the last emitted token ends with that its callback could not yet
  // pass: the generator's (generator.h); the runtime, which handles no
  // text, leaves it empty.
  std::string_view tail;
};

// Each emitted token, in order, a step after it was drawn.
using TokenCallback = void (*)(tokenizer::TokenId token, void* userdata);

#if FRIZZLE_DIAGNOSTICS_ENABLED
// A step as it reported, profiled (above): its kind, position and tokens,
// and its pass's GPU times. The two times of one step are on one conversion
// of the GPU's ticks to nanoseconds; two steps' may not be — on the target
// natively, Dawn changed the factor once during a process, by 1.07 × 10⁻⁴,
// so a step began 341.6 s "before" the step ahead of it. Differences across
// steps are valid only between steps converted alike.
struct StepTimes {
  bool prefill;
  std::uint32_t position;
  std::uint32_t tokens;
  std::uint64_t begin_ns;
  std::uint64_t end_ns;
};
using StepObserver = void (*)(const StepTimes& step, void* userdata);
#endif
// Once a turn, when it has finished. `result.message` is valid only during
// the call.
using TurnCallback = void (*)(const TurnResult& result, void* userdata);

class Runtime {
 public:
  // Preconditions: `program` was built for the model's graph, reading back
  // its draw's record (graph/graph.h); `cache` is empty and its capacity at
  // most the context the plan offers; `stops` is not empty.
  Runtime(std::unique_ptr<kernels::Program> program, cache::KvCache cache,
          StopSet stops);

  // Starts a turn over `prompt`, the whole rendered conversation. On Ok,
  // `on_token` is called for each token emitted and `on_turn` once, both
  // with `userdata`; otherwise neither is. Preconditions: every token below
  // the vocabulary; `prompt` is not read after the call.
  [[nodiscard]] StartResult start(std::span<const tokenizer::TokenId> prompt,
                                  const policy::TurnPolicy& policy,
                                  TokenCallback on_token, TurnCallback on_turn,
                                  void* userdata);

  // Ends the running turn at its next report; with none, nothing.
  void cancel() noexcept;

  // The context offered: the most tokens a prompt and its reply hold.
  [[nodiscard]] std::uint32_t capacity() const noexcept;

  // The positions the cache holds; between turns, the tokens held.
  [[nodiscard]] std::uint32_t cached() const noexcept;

  // Between turns, keeps at most the first `tokens` of the cache and the
  // tokens held: the cache's truncate (cache/kv.h), which empties it for a
  // rollback past its reserve. Never lengthens it. False, changing nothing,
  // while a turn runs.
  [[nodiscard]] bool rewind(std::uint32_t tokens) noexcept;

#if FRIZZLE_DIAGNOSTICS_ENABLED
  // Sets the step observer, or with null clears it; false, changing
  // nothing, while a turn runs or, setting one, when the program's device
  // granted no timestamp queries (kernels/program.h, can_profile).
  bool observe_steps(StepObserver observer, void* userdata) noexcept;
#endif

  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  // What the runtime and its in-flight callbacks share; defined only where
  // the runtime is.
  struct State;

 private:
  std::shared_ptr<State> state_;  // shared with in-flight callbacks
};

}  // namespace frizzle::runtime
