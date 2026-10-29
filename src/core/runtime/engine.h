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
//   generate(text, edits, policy, on_text, on_end)
//     - `text` is exactly what the model reads: Gemma's chat template already
//       applied. The engine renders no template (the worker does; T).
//     - Precondition, the caller's: `text` is the template's rendering,
//       which begins with <bos> and wraps each message in its turn markers.
//       The renderer owns that and its tests lock it; the engine checks
//       nothing of the template. Text wrapped otherwise still runs, and the
//       reply is what the model makes of it.
//     - Runs the model over `text` and streams the reply as text, under
//       `policy`: its sampling settings, seed and token limit; then calls
//       back once, `on_end`, with the turn's result (runtime.h) and the
//       run's captures (Captured below).
//     - Captured: every layer's neurons at the last prompt position, after
//       this run's edits (capture/capture.h), with the store's layout, one
//       slot a layer; or why there are none:
//         - Taken: the turn ended Stop, Limit or Context, so its step that
//           ends the prompt ran, and the store was read back whole; the
//           words are this run's, valid only during the call.
//         - NotTaken: the turn was Cancelled or failed. The step that ends
//           the prompt may not have run, so the store may still hold an
//           earlier run's neurons; it is not read, and nothing is reported.
//         - Unread: the turn ended as Taken says, but the read back failed;
//           `mapped` says how (residency/mapping.h). The turn's own result
//           stands. An engine destroyed while the store is read reports
//           Unread, Cancelled.
//       The turn's result is reported as it was in every case; a capture
//       never turns a reply into a failure, nor a failure into a reply.
//     - `edits` (edit/edit.h) applies to this run only: at the last prompt
//       position and every generated token, never earlier. No edits, or
//       Scale × 1 edits, give the run without edits, bit for bit, from the
//       same seed.
//     - Refused at once, named (OperationError): not loaded; a turn running;
//       an invalid edit set (Edits, with edit::check's result); and what the
//       generator refuses (generator.h): text the tokenizer cannot encode, an
//       empty or too long prompt. `edits` is not read after the call.
//     - Invariant, between runs: the cache holds only entries computed
//       without edits. So a later prompt that contains an edited run's reply
//       (a conversation) never reuses that reply's edited entries.
//   chat_template(), once begin_load has accepted a file: the chat template
//     the file carries (tokenizer.chat_template) and its BOS and EOS tokens'
//     text, read at begin_load, for the host that renders conversations (T,
//     the worker); none when the file carries no template, which the host
//     refuses by name. The engine itself never renders it.
//   cancel(): ends the running turn at its next report; on_end is then
//     called once, Cancelled (runtime.h). With no turn running, nothing. The
//     turn's steps already run report first and are discarded (turn.h), and
//     a run with edits still rewinds the cache.
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
//   2. Write its table (edit::table) into the plan's `edit_table` buffer
//      (graph/graph.h) with one queue write on the device's queue, which
//      orders it before the turn's first step. No turn is in flight, so no
//      step reads the table while it changes.
//   3. Start a turn on the generator with the text and the caller's policy.
//      The turn loop reuses whatever prefix of the text the cache already
//      holds from the turn before, and recomputes the rest (runtime.h).
//   4. Stream its text. When it ends, however it ends, and the set had
//      edits: rewind the cache to just before the edits began, the turn's
//      prompt tokens less one (Generator::rewind, runtime.h).
//   5. If it ended Stop, Limit or Context, read the capture store back
//      (Reader::read), after every step the turn submitted. Then call back
//      once with how it ended and what was captured. The engine stays
//      running until then, so no turn starts while the store is read, and
//      the turn's message is copied to outlive the read.
//
// Design and cost
//   - The engine is the composition Charlotte's browser bindings made
//     inline, as a class every host shares. It changes nothing ported but
//     where the ported header says (runtime.h's and generator.h's rewind,
//     graph.h's edit launch).
//   - The tokenizer and stops load before the upload begins, on the CPU,
//     so every refusal begin_load names comes at once. Charlotte overlapped
//     them with the device's buffer creation (20 to 56 ms, its bindings);
//     not done here until it is measured to matter.
//   - Settings: Charlotte never measured Gemma 3, so the load and turn
//     policies are policy.h's defaults for an unmeasured model; the tool
//     prints what it ran with. Measuring is J's, still open.
//   - State shared with in-flight callbacks (F.27), as the runtime's is: each
//     callback carries a record holding it, so destroying the engine leaves
//     a live state for the callback to land on, and the last holder frees it.
//   - Edits: the engine keeps no edit state between runs. Each run writes
//     its whole table, and the rewind leaves the cache holding no edited
//     entry, so a run is a function of its text, edits and policy. Whether
//     a run had edits rides in its call's record.
//   - The cache: positions before the last prompt position are never
//     edited, so the original and the edited run of one prompt share them,
//     in either order; each recomputes only the last prompt position, as
//     the turn loop already does.
//   - A conversation after an edited reply prefills that reply again, and
//     the position before it: the reply's tokens plus one. A rewind deeper
//     than the cache's rollback reserve (policy.h, 4,096 tokens by default)
//     empties it instead (cache/kv.h), and the next run prefills from the
//     first token. Any set with edits rewinds, including Scale × 1, which
//     changes no bits; telling them apart is not worth a rule.
//   - The table write cannot fail validation: the buffer is a scratch
//     buffer, made with CopyDst (residency/upload.cpp), planned at the
//     table's largest size, and the table is whole u32 words.
//   - A turn costs what the turn loop's does (runtime.h). The engine adds a
//     record a call, the edit set's check and table, at most 48 KiB, and one
//     queue write (GPU.1): nothing a token. The edit kernel and the capture
//     launch run in every scoring step (edit/edit_kernel.h,
//     capture/capture.h). A run that ends Stop, Limit or Context adds one
//     read back of the store, 702 KiB for Gemma 3 1B, before its end is
//     reported (capture/reader.h).
//   - Captures cross with the result, not in a call of their own: a host
//     gets them exactly once, with the run they belong to, and never asks
//     for "the last run's" (logical-overview.md, invariant 5).
//   - on_end's type is RunEndCallback, not the generator's EndCallback: the
//     native tool and the replay scenario take the captures and read nothing
//     of them; the browser bindings send them to the page (slice 4, step 2).
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
  // For Taken: every layer's slot, in layer order, and the store's words,
  // valid only during the callback.
  std::span<const capture::Slot> slots;
  std::span<const float> neurons;
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

  // On ok(), `on_text` is called for each piece of the reply and `on_end`
  // once, both with `userdata`. `text` and `edits` are not read after the
  // call.
  [[nodiscard]] OperationStart generate(std::string_view text,
                                        std::span<const edit::Edit> edits,
                                        const policy::TurnPolicy& policy,
                                        TextCallback on_text,
                                        RunEndCallback on_end, void* userdata);

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
