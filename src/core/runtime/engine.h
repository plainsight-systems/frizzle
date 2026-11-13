#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/capture/capture.h"
#include "core/edit/edit.h"
#include "core/gguf/byte_source.h"
#include "core/gpu/device.h"
#include "core/policy/policy.h"
#include "core/residency/mapping.h"
#include "core/runtime/generator.h"

namespace frizzle::runtime {

// Axis M: changes with an operation the engine offers, or the order of the
// stages inside one.
//
// The engine's operations (docs/architecture/logical-overview.md): load and
// generate. Every host calls these and nothing below them: the native tool
// now, the browser boundary (src/wasm/) in slice 3. It composes what is
// ported, unchanged: the GGUF reader, the capability table, the residency
// planner and upload, the graph and program, the turn loop and the
// generator. It shows what the model produces and computes nothing beside
// it.
//
// Contract
//   load, in three calls, because the file arrives in pieces
//     1. begin_load(front, max_chunk, policy, ready): reads the GGUF index
//        and the tokenizer from `front`, a byte source whose size() is the
//        file's and which holds at least the file's index and metadata
//        (natively the whole file; in the browser its first bytes).
//        Describes the architecture through the capability table, plans
//        residency within the device's limits and `policy`, loads the
//        tokenizer, resolves the stop tokens, and begins the upload.
//        Refused at once, named (LoadError), with no callback: a load or a
//        turn in progress; a file that is not a GGUF this engine reads; an
//        architecture or tokenizer it does not implement; stops it cannot
//        resolve; a plan that does not fit. `front` is not read after the
//        call. Otherwise `ready` is called once: Ok when the upload's
//        buffers exist, or the upload's failure. A model loaded before is
//        released first, so the two are never on the device together.
//     2. write(offset, chunk, accepted): the file's bytes, in order, at most
//        `max_chunk` a call, each accepted through its callback before the
//        next; `chunk` may be reused once accepted.
//     3. finish_load(done): checks the upload is complete, builds the graph
//        and its program (pipelines compile asynchronously), makes the
//        capture store's reader (capture/reader.h), then makes the cache and
//        the turn loop. `done` reports Ok or a named failure; a reader
//        WebGPU refuses is Capture, with its message.
//     write and finish_load are refused at once, NotLoading, outside a load,
//     and Busy while the previous call's callback is pending.
//   generate(text, kept, edits, policy, on_token, on_end)
//     - `text` is exactly what the model reads: Gemma's chat template already
//       applied. The engine renders no template (the worker does; T).
//     - `kept`: reply token identifiers from an earlier run, which the
//       model reads after `text` as if it had just drawn them: a branch
//       that keeps the first k tokens of that reply, so k = 0 is a run from
//       the reply's start. The run's input is text, then kept; its steps
//       count from the text's last token (edit/edit.h): step k is the
//       position whose input is the last kept token, which draws reply
//       token k, the first this run emits.
//     - Precondition, the caller's: `text` is the template's rendering,
//       which begins with <bos> and wraps each message in its turn markers.
//       The renderer owns that and its tests lock it; the engine checks
//       nothing of the template. Text wrapped otherwise still runs, and the
//       reply is what the model makes of it.
//     - Runs the model over the input and streams the reply, under
//       `policy`: its sampling settings, seed and token limit: each emitted
//       token with its identifier and text (generator.h); then calls back
//       once, `on_end`, with the turn's result (runtime.h), its tail the
//       text the last token ends with, if any (generator.h), and the run's
//       captures (Captured below). The sampler draws by seed and position
//       (sampler/sampler.h), so a branch draws each position as the run it
//       keeps from did, and only edits make the two differ.
//     - A max_tokens of 0 inspects (runtime/turn.h): step k is computed,
//       with the edits in force there, and captured; nothing is drawn; the
//       turn ends Limit with no tokens. So a host sees the captures at any
//       point of a reply without generating from it.
//     - Captured: every layer's captures at step k, the input's last
//       position, with the edits in force there (capture/capture.h): its
//       neurons after the edits, the residual coming into it, what its
//       attention and feed-forward blocks added, and its attention weights
//       and their statistics. With them, the store's layout and the
//       position, which the weights count their keys back from. Or why
//       there are none:
//         - Taken: the turn ended Stop, Limit or Context, so its step that
//           ends the prompt ran, and the store was read back whole; the
//           words are this run's, valid only during the call.
//         - NotTaken: the turn was Cancelled or failed. The step that ends
//           the prompt may not have run, so the store may still hold an
//           earlier run's captures; it is not read, and nothing is reported.
//         - Unread: the turn ended as Taken says, but the read back failed;
//           `mapped` says how (residency/mapping.h). The turn's own result
//           stands. An engine destroyed while the store is read reports
//           Unread, Cancelled.
//       The turn's result is reported as it was in every case; a capture
//       never turns a reply into a failure, nor a failure into a reply.
//     - `edits` (edit/edit.h), each over its range of steps, placed at the
//       text's last token (origin = the text's tokens less one). A run is a
//       function of its text, kept tokens, edits and policy: an edit applies
//       at exactly the positions its range holds, kept or generated, whether
//       the cache holds them or they are computed again. So:
//         - No edits, or Scale × 1 edits, give the run without edits, bit
//           for bit, from the same seed.
//         - A branch keeping k tokens of a reply, with that reply's own
//           edits, continues it bit for bit: the continuation is the
//           reply's tokens from k on, with its ending.
//         - A reply generated with edits from step b, kept whole as the
//           next branch's history, is recomputed with those edits at its
//           positions from b on, as generated, if the cache no longer holds
//           them.
//     - Refused at once, named (OperationError): not loaded; a turn running;
//       an invalid edit set (Edits, with edit::check's result); a kept
//       identifier at or past the vocabulary (Kept, naming its index); and
//       what the generator refuses (generator.h): text the tokenizer cannot
//       encode, an empty or too long input. `text`, `kept` and `edits` are
//       not read after the call.
//     - Invariant, between runs: the cache holds entries computed with the
//       edits the run that wrote them applied, at their positions, and the
//       engine holds those edits, placed (edit::placed), with it. They are
//       dropped together: a run that empties the cache (a step's failure, a
//       rollback past the ring's reserve, cache/kv.h) or a new load. A run
//       reuses the cache only up to the first position where its tokens or
//       its edits in force differ (cache/prefix.h), so a later run never
//       reads an entry computed under edits it does not apply there.
//   chat_template(), once begin_load has accepted a file: the chat template
//     the file carries (tokenizer.chat_template) and its BOS and EOS tokens'
//     text, read at begin_load, for the host that renders conversations (T,
//     the worker); none when the file carries no template, which the host
//     refuses by name. The engine itself never renders it.
//   cancel(): ends the running turn at its next report; on_end is then
//     called once, Cancelled (runtime.h). With no turn running, nothing. The
//     turn's steps already run report first and are discarded (turn.h); the
//     cache keeps what they wrote, under the run's edits, which the engine
//     then holds.
//   A begin_load refused as Index because `front` ends before the index
//     does carries bytes_needed: the least front that could hold it, so a
//     host holding only the file's first bytes reads that many and begins
//     again, rather than parsing the subject.
//   - One operation at a time. Callbacks run on the thread that processes
//     the device's events, and a callback may run before the call that
//     started it returns: the upload refuses some files and accepts a first
//     chunk at once, and the program refuses some graphs at once. Every
//     failure is reported, named; none is turned into an empty reply.
//   - Borrows the device, which outlives it (I.11); the upload takes a
//     reference of its own. Owns everything it makes. Destroying it with an
//     operation in flight reports that operation Cancelled, as the runtime
//     does, so each callback's userdata must stay valid until its callback
//     has run.
//
// Algorithm, generate
//   1. Check the edit set (edit::check); refuse an invalid one.
//   2. Prepare the turn on the generator: the text encoded, the kept
//      identifiers appended (generator.h); origin = the text's tokens less
//      one.
//   3. Place the edits at origin (edit::placed). Rewind the cache to the
//      first position where they differ from the held edits
//      (cache::first_edit_difference), when that is below its length; then
//      hold the placed edits in their place.
//   4. Write the table (edit::table, at origin) into the plan's
//      `edit_table` buffer (graph/graph.h) with one queue write on the
//      device's queue, which orders it before the turn's first step. No
//      turn is in flight, so no step reads the table while it changes.
//   5. Start the prepared turn with the caller's policy. The turn loop
//      reuses whatever prefix of the input the cache holds, after the
//      rewind, and recomputes the rest (runtime.h).
//   6. Stream each emitted token. When the turn ends with the cache
//      emptied, drop the held edits with it.
//   7. If it ended Stop, Limit or Context, read the capture store back
//      (Reader::read), after every step the turn submitted. Then call back
//      once with how it ended and what was captured. The engine stays
//      running until then, so no turn starts while the store is read, and
//      the turn's message and tail are copied to outlive the read.
//
// Design and cost
//   - The engine makes the composition once, as a class every host shares,
//     not inline in each host's bindings. It changes nothing ported but
//     where the ported header says (runtime.h's and generator.h's rewind,
//     graph.h's edit launch).
//   - The tokenizer and stops load before the upload begins, on the CPU,
//     so every refusal begin_load names comes at once. Overlapping them with
//     the device's buffer creation, about 20 to 56 ms (an estimate, not
//     measured in Frizzle), is not done until it is measured to matter.
//   - Settings: Gemma 3 has not been measured, so the load and turn
//     policies are policy.h's defaults for an unmeasured model; the tool
//     prints what it ran with. Measuring is J's, still open.
//   - State shared with in-flight callbacks (F.27), as the runtime's is: each
//     callback carries a record holding it, so destroying the engine leaves
//     a live state for the callback to land on, and the last holder frees it.
//   - Edits: each run writes its whole table, and the engine holds the
//     placed edits of the run that wrote the cache, the one state it keeps
//     beside the cache, kept and dropped with it, so a run is still a
//     function of its text, kept tokens, edits and policy. Until slice 4's
//     branch it held none, and rewound after every edited run, so the
//     cache held no edited entry; a branch from an accepted reply needs
//     exactly those entries, so they are now kept and compared instead.
//   - The cache: runs of one prompt share every position before the first
//     whose edits differ: an original and a run edited from step k share
//     the input up to step k, and the branch recomputes from there, as the
//     turn loop recomputes the last position. A branch back into a reply
//     rolls back to its point; deeper than the rollback reserve (policy.h,
//     4,096 tokens by default), it empties the cache (cache/kv.h) and the
//     run prefills from the first token, correct and slower. Scale × 1
//     edits change no bits but are compared as edits; telling them apart
//     is not worth a rule.
//   - Kept identifiers cross from the host, a few kilobytes a run: the page
//     holds the conversation (logical-overview.md), and the engine checks
//     only that each names a token.
//   - The table write cannot fail validation: the buffer is a scratch
//     buffer, made with CopyDst (residency/upload.cpp), planned at the
//     table's largest size, and the table is whole u32 words.
//   - A turn costs what the turn loop's does (runtime.h). The engine adds a
//     record a call, the edit set's check and table, at most 48 KiB, and one
//     queue write (GPU.1): nothing a token. The edit kernel and the capture
//     launch run in every scoring step (edit/edit_kernel.h,
//     capture/capture.h). A run that ends Stop, Limit or Context adds one
//     read back of the store, 3.2 MiB for Gemma 3 1B, before its end is
//     reported (capture/reader.h).
//   - Captures cross with the result, not in a call of their own: a host
//     gets them exactly once, with the run they belong to, and never asks
//     for "the last run's" (logical-overview.md, invariant 5).
//   - The engine's operations stay load and generate; a branch and an
//     inspection are generate's, with kept tokens and a limit of 0, rather
//     than operations of their own: one turn loop, one capture, one way a
//     run's input is made.
//   - on_end's type is RunEndCallback, not the generator's EndCallback: the
//     native tool takes the captures and reads nothing of them; the replay
//     scenario compares two runs' stores; the browser bindings send the
//     whole store to the page, with its layout and the position.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.1    Make interfaces explicit: the stages and the order of load, and
//            when callbacks may run, are stated above; the bytes a short
//            front needs cross as a number, not inside a sentence.
//     I.11   Never transfer ownership by a raw pointer or reference: the
//            device is borrowed; everything else is owned.
//     E.27   Use error codes systematically: LoadError and OperationError,
//            and the runtime's own, each named.
//     C.2    Use class if the class has an invariant: the engine is
//            unloaded, loading, or loaded, and each operation states which
//            it needs.
//     F.27   Use a shared_ptr to share ownership: the state, between the
//            engine and its callbacks in flight.
//     ES.79  Use default for common cases only: the switches over error
//            enums name every case and have no default, so a new case is a
//            compile warning, not a silent fallthrough.
//   C++ performance guidelines
//     MEM.9  Allocate at init: the cache and the program at load; a turn
//            allocates only the engine's record for it and its edit table.
//     GPU.1  Keep data on the device: the edit table is written once a run;
//            the capture store is read back once a run, at its end.

enum class LoadError : std::uint8_t {
  Ok,
  Busy,          // a load stage or a turn is in progress
  NotLoading,    // write or finish_load with no load begun
  Index,         // the bytes are not a GGUF index this engine reads
  Architecture,  // missing, not implemented, or not describing this file
  Fit,           // the plan does not fit the device's limits or the budget
  Tokenizer,     // missing, not implemented, or not loading from this file
  Stops,         // the stop tokens cannot be resolved
  Upload,        // the upload refused or failed; the subject names how
  Graph,         // the graph does not build over the plan
  Program,       // the kernels did not build; the subject is WebGPU's
  Capture,       // the capture store's readback was refused; WebGPU's words
  Cancelled,     // the engine was destroyed with the stage in flight
};

struct LoadResult {
  LoadError error = LoadError::Ok;
  std::string subject;  // what was refused, and why
  // For Index, when `front` ended before the index: the bytes it needs; 0
  // otherwise.
  std::uint64_t bytes_needed = 0;
  [[nodiscard]] bool ok() const noexcept { return error == LoadError::Ok; }
};

// The chat template a file carries, for the host that renders it.
struct ChatTemplate {
  std::string text;  // tokenizer.chat_template, as the file holds it
  std::string bos;   // the BOS token's text, "<bos>" for Gemma 3
  std::string eos;   // the EOS token's text
};

using LoadCallback = void (*)(const LoadResult& result, void* userdata);

enum class OperationError : std::uint8_t {
  Ok,
  NotLoaded,
  Busy,   // a turn is running
  Edits,  // the edit set is invalid; `edits` says why
  Kept,   // a kept identifier names no token; start.kept_out_of_vocabulary
          // says which
  Start,  // the generator refused; `start` says why
};

struct OperationStart {
  OperationError error = OperationError::Ok;
  edit::EditCheck edits;  // for Edits
  GenerateResult start;   // for Start
  [[nodiscard]] bool ok() const noexcept { return error == OperationError::Ok; }
};

// A run's captures (generate, above).
enum class CaptureStatus : std::uint8_t {
  Taken,
  NotTaken,  // the turn was Cancelled or failed
  Unread,    // the read back failed; `mapped` says how
};

struct Captured {
  CaptureStatus status = CaptureStatus::NotTaken;
  residency::Mapped mapped = residency::Mapped::Ok;  // for Unread
  // For Taken: the store's layout (the engine's, alive while it is), the
  // store's words, valid only during the callback, and the input's last
  // position, which each layer's weights count back from: key i of a head
  // is at position − min(position + 1, keys) + 1 + i (capture/weights.h).
  const capture::Layout* layout = nullptr;
  std::span<const float> words;
  std::uint32_t position = 0;
};

using RunEndCallback = void (*)(const TurnResult& result,
                                const Captured& captured, void* userdata);

class Engine {
 public:
  // Preconditions: `device` outlives the engine.
  explicit Engine(const gpu::Device& device);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  // Load, step 1. On ok(), `ready` is called once with `userdata`.
  // Preconditions: max_chunk > 0.
  [[nodiscard]] LoadResult begin_load(gguf::ByteSource& front,
                                      std::size_t max_chunk,
                                      const policy::LoadPolicy& policy,
                                      LoadCallback ready, void* userdata);
  // Load, step 2. On ok(), `accepted` is called once with `userdata`.
  [[nodiscard]] LoadResult write(std::uint64_t offset,
                                 std::span<const std::byte> chunk,
                                 LoadCallback accepted, void* userdata);
  // Load, step 3. On ok(), `done` is called once with `userdata`.
  [[nodiscard]] LoadResult finish_load(LoadCallback done, void* userdata);

  // On ok(), `on_token` is called for each emitted token and `on_end`
  // once, both with `userdata`. `text`, `kept` and `edits` are not read
  // after the call.
  [[nodiscard]] OperationStart generate(
      std::string_view text, std::span<const tokenizer::TokenId> kept,
      std::span<const edit::Edit> edits, const policy::TurnPolicy& policy,
      ReplyTokenCallback on_token, RunEndCallback on_end, void* userdata);

  // Ends the running turn at its next report (above); with none, nothing.
  void cancel() noexcept;

  // The context the loaded model offers; 0 when none is loaded.
  [[nodiscard]] std::uint32_t context() const noexcept;

  // The file's chat template, once begin_load has accepted it (above).
  [[nodiscard]] std::optional<ChatTemplate> chat_template() const;

  // What the engine and its callbacks in flight share; defined only where
  // the engine is.
  struct State;

 private:
  std::shared_ptr<State> state_;
};

}  // namespace frizzle::runtime
