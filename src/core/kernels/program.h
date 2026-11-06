#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/diagnostics.h"
#include "core/gpu/wgpu_handles.h"
#include "core/kernels/interface.h"
#include "core/residency/upload.h"

namespace frizzle::kernels {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Carries out a graph's launches (interface.h) on the device: built once at
// load, then run once a step. It holds the GPU objects the launch
// descriptions name — pipelines, bind groups, the constants and step
// uniforms, the readback slots — as upload holds the buffers the plan names.
// It decides nothing about the model: what runs, in what order, over which
// buffers, is the graph's (graph/graph.h); how many workgroups, the
// launch's geometry (interface.h, workgroups_for).
//
// Contract
//   - build(upload, launches, readback, done, userdata) calls `done` once,
//     with the program and Ok, or with null and Build (Cancelled where
//     WebGPU cancels a pipeline's creation). Refused, Build, named by the
//     launch's index, before any GPU call: constants over
//     kMaxLaunchConstants; a pack format with no pack; no invocations (a
//     workgroup size or invocations a row of 0); a tiled launch over the
//     last token, or a tile that is not a whole number of workgroups; a key
//     split with no window; an override named `workgroup_size` or
//     `last_token`, which are the program's, or one name given twice; a
//     binding to a buffer upload did not create. Then, before anything is
//     created, a readback range that is not a buffer upload created, a
//     whole number of words from 4 to kMaxReadback bytes. After those,
//     build runs inside out-of-memory,
//     validation and internal error scopes; any of them, or a pipeline that
//     fails to compile, fails the build with every message WebGPU gave,
//     joined by newlines: a shader that does not compile fails its pipeline
//     too, and the pipeline's message, which may land first, only says the
//     module was invalid.
//   - run(step, done, userdata) calls `done` once: when the queue has
//     finished the step, or its readback has mapped, or it has failed.
//     `readback` carries the step's copy of the readback range, valid only
//     during the call, empty for a step that read none or failed;
//     `message` is WebGPU's for a failed step, at most 1 KiB, empty on
//     success.
//   - Refused at once, unrun, Step, through the callback, before anything
//     is written: a third step while two are outstanding; a step of more
//     than kPrefillBlock tokens, since Step holds kPrefillBlock identifiers
//     (interface.h) and a longer step's write would run past them; in a
//     diagnostic build, a step beside a profiled one. Such a refusal is not
//     numbered among the steps and is never DeviceLost.
//   - Failed, Step, with nothing submitted, reported in run order once its
//     scopes pop. The step's head and identifiers are written to the step
//     uniform in every case. A step past kMaxPositions (position + tokens >
//     2^24), or with an input outside its binding's range or not of whole
//     words, fails before any input is written; a launch needing more
//     workgroups than the device's maxComputeWorkgroupsPerDimension is found
//     while the pass is encoded, after the inputs are written, and the pass
//     is not submitted. Refused, not truncated.
//   - A step that ran fails, Step, when its validation or internal scope
//     catches an error, or its work or mapping does not complete; the
//     first failure's message is kept. Any numbered step that fails, Step,
//     once the device has said it is lost is reported DeviceLost instead,
//     and every step reported after the program is destroyed, Cancelled.
//     WebGPU does not order the
//     lost callback before the failed mapping, so a failure that comes
//     first is reported Step, and a later one DeviceLost. A step reported
//     done is not proof the device ran it: a lost device resolves queued
//     work as done and scopes clean (gpu/device.h). What a step computed is
//     trusted only through a completed mapping, which a lost device refuses
//     — the sampler's readback of the step's drawn token, as upload's
//     witness is its.
//   - Order: at most two steps outstanding, one running and the next
//     queued. Callbacks come in the order the steps were run, whatever
//     order their mappings settle in (kernels/order.h).
//   - Preconditions of run(): step.tokens >= 1 (not checked: a step of 0
//     has no schedule, kernels/schedule.h); every identifier the step
//     carries below the vocabulary, which the runtime checks before it
//     writes a step (interface.h).
//   - run(step, inputs, done, userdata) is run() with `inputs` written
//     first, each into its range of one of the upload's buffers. The
//     engine passes none (runtime/runtime.cpp); the kernels suite's tests
//     use it to place activations and edit tables before a step.
//   - Ownership: the program borrows `upload`, and so its buffers, which
//     outlive it (I.11). GPU objects are RAII handles (R.1). Callbacks keep
//     the program's shared state alive on their own, as upload's do, so
//     destroying a program with a step in flight reports Cancelled.
//
// Algorithm
//   Build, at load:
//     1. Check every launch description and the readback range (Contract);
//        work out each launch's offset in the constants buffer, whole slots
//        of kLaunchConstantsAlignment, at least one, in launch order.
//     2. Inside the build scopes: create the constants buffer and write
//        every launch's constants in one write; create the step uniform,
//        sizeof(Step); where the graph names a readback range — the draw's
//        16-byte record (sampler/sampler.h) — create two mappable slots of
//        its size.
//     3. Find each distinct pipeline by its key: the kernel's text, its
//        entry point, the format it unpacks, the format it packs, its
//        override constants in name order. Compose each (compose, below)
//        and request every pipeline at once, with
//        createComputePipelineAsync.
//     4. Once every pipeline and the scopes have settled: ask each pipeline
//        once for its bind group layout, make a bind group for every launch
//        — binding 0 the step, 1 its constants' slots, 2 onward its
//        bindings, in order — and plan the schedules (kernels/schedule.h).
//        Once those scopes settle, hand the program over.
//   Run, once a step:
//     1. Refuse, or take the next sequence number and the slot it names.
//     2. Push the step's validation and internal scopes; write the step's
//        head and the identifier words it uses (interface.h) in one write.
//     3. Check the position bound and the inputs; write each input.
//     4. One command encoder, one compute pass: for each launch the step's
//        schedule lists for its token count and logits, its workgroups from
//        workgroups_for — skipped at 0, a combine the step does not split —
//        set the pipeline where it changes from the launch before, set the
//        bind group, dispatch. Launches run in the graph's order; WebGPU
//        orders a pass's dispatches by their buffer use, so a launch sees
//        what the one before it wrote.
//     5. For a step that asks for logits, where the graph reads back, copy
//        the range into the step's slot after the pass; submit; map the slot,
//        or, for a step that reads nothing back, wait on the queue.
//     6. Pop the scopes. The last of the step's callbacks to land settles
//        it, and the order reports every settled step whose turn it is.
//   WebGPU orders a queue's writes after the submits before them, so one
//   step uniform serves both outstanding steps, and a step's inputs never
//   overtake the last step's pass over the same buffer. It does not order
//   mappings of two buffers: hence the sequence numbers, and the two slots,
//   so a slot is never copied into while mapped (GPU.7).
//
// Profiling, in a diagnostic build alone (core/diagnostics.h; TLM.1): the
// clean build compiles none of it, and its module has no symbol of it
// (TLM.8). A profiled step runs as run() does — its launches in the graph's
// order, in its one compute pass, over the same pipelines and bind groups —
// but may stop after its first `launches` launches, and is timed by the GPU:
//   - a timestamp at the pass's beginning and one at its end, always — the
//     time the step, or its prefix, takes in the shape run() gives it;
//   - and, on a device that grants timestamps inside a pass, one after each
//     launch it dispatches, so each launch's time in that same pass. Only a
//     native Dawn build looks for that feature. The target's Metal adapter
//     does not offer it: there a launch's time is what adding it to a
//     prefix adds.
// No pass is split: a pass a launch would drain the GPU at every boundary,
// timing each launch in a shape run() never gives it, and add three calls
// into Dawn a launch.
// The timestamps are resolved into a buffer and copied to a mappable one
// with the step's readback, and reported with it; WebGPU gives them in
// nanoseconds, at the resolution of the GPU's counter. Each of the two slots
// a step may be outstanding in has its own query set, resolve buffer and
// readback buffer, made at the first profiled step — so profiled steps
// pipeline as run()'s do, and a run of them is timed as it runs. A profiled
// step runs only on a device that granted the timestamp-query feature
// (gpu/device.h's DiagnosticRequest), not beside an unprofiled one, and for
// at most the graph's launches; otherwise, and for run()'s refusals, it is
// refused at once, named. A whole profiled step's results are run()'s: no
// timestamp writes a buffer a kernel reads. A prefix leaves the step's later
// launches undone — the draw among them, so it reads nothing back, and the
// cache holds the step's positions for the layers it ran alone — so a step
// at those positions runs whole before any result is read from them.
//
// Design and cost
//   - Counted, a step of run(): calls into WebGPU are 1 writeBuffer of 64
//     bytes for a fed step and 64 + 16 × ceil(tokens / 4) for another, 1
//     createCommandEncoder, 1 beginComputePass, for each launch that runs 1
//     setBindGroup, 1 dispatchWorkgroups and 1 setPipeline where the
//     kernel changes, then end, finish, submit, two scopes' pushes and pops,
//     and the release of the three single-use objects WebGPU makes a step —
//     the command encoder, the pass encoder and the command buffer; then
//     onSubmittedWorkDone, or, for a step that reads back, a
//     copyBufferToBuffer, a mapAsync, a getConstMappedRange and an unmap:
//     14 calls out of the module, 17 reading back, and 2 or 3 a launch; and
//     three callbacks back in, the two scopes' and the queue's or the
//     mapping's. Each of a step's non-empty inputs adds one writeBuffer.
//   - Counted, a profiled step, in diagnostic builds: run()'s, and the
//     resolve of its timestamps, their copy, and their buffer's mapAsync,
//     getConstMappedRange and unmap — 22 calls a whole step, 19 a prefix,
//     which reads nothing back — a fourth callback, the timestamps'
//     mapping, and, where timestamps inside a pass are granted, a
//     writeTimestamp a launch; the device's capability is read once, at
//     build.
//   - Everything else — buffers, pipelines, bind groups — is made at load.
//     The GPU adds about 1.5 µs a dispatch, an estimate, not measured in
//     Frizzle (interface.h).
//   - Build, counted: a pipeline for each distinct composition and a bind
//     group layout for each pipeline — how many for Gemma 3 1B is not
//     counted here — one key a launch to find them, a bind group a launch,
//     and one write of every launch's constants.
//   - The two readback slots are kMaxReadback, 16 bytes, each; every
//     outstanding step's message buffer is reserved at build, 1 KiB, and a
//     longer message is cut there. A step allocates nothing on the heap.
//   Optimization (browser): every launch's constants in one buffer and one
//   write at load, and a step's parameters and identifiers in one write
//   (WASM.2, GPU.9).
//   Optimization (practice): each distinct kernel is composed and compiled
//   once, and every pipeline is requested at once, so the browser compiles
//   them together (WASM.7).
//   Optimization (practice): one pass and one submit a step, and a
//   pipeline set only where it changes (GPU.6).
//   - Nothing tests the program alone. The kernels suite's tests run their
//     kernels through it (tests/support/scratch_program.h), and the oracle and
//     replay run every step through it on the pinned model. No test reaches the
//     refusal of a step of more than kPrefillBlock tokens, the kMaxPositions
//     bound, or the workgroup limit.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer: the upload is
//            borrowed, and stated so.
//     R.1    Manage resources with resource handles: every GPU object is a
//            handle that releases it.
//     E.27   Use error codes systematically: ProgramError, through every
//            callback, with WebGPU's message.
//     I.5    State preconditions: run()'s, above; the token bound, which a
//            longer step would break by writing past Step, is a refusal.
//     C.21   Define or delete all of copy, move and destructor: Program
//            declares its destructor and deletes copy and move; it lives
//            behind the unique_ptr build() hands over.
//     Enum.7 Specify the underlying type only when necessary: ProgramError
//            is one byte; it travels only through in-process callbacks.
//     ES.42  Keep use of pointers simple: a step's record, readback slot and
//            timer are found with at() on two-entry arrays, a step's number
//            modulo 2 — one comparison each a step, against the step's
//            hundreds of dispatches — and the timestamps are read through a
//            span, with no pointer arithmetic.
//     ES.46  Avoid narrowing conversions: an override's earlier names are a
//            span's first entries, not an iterator offset by a size.
//     F.16   Pass "in" parameters cheaply: a build failure's message by
//            reference to const.
//     C.48   Prefer default member initializers: the build's callback and
//            userdata start null.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work: one pass, one submit, a step.
//     GPU.7  Pipeline CPU and GPU work with multi-buffered resources: two
//            steps outstanding, a readback slot each.
//     GPU.9  Suballocate: one constants buffer for every launch.
//     MEM.11 Plan offsets once from known sizes: each launch's constants
//            offset, at build.
//     WASM.2 Batch work across the JS boundary: one write a step.
//     WASM.7 Budget startup: pipelines compile together, asynchronously.
//     MEM.9  Allocate at init: buffers, pipelines and bind groups at load;
//            a step makes only the single-use encoders WebGPU requires, and
//            its callbacks carry the program's own state, so it allocates
//            nothing on the heap.
//     GPU.10 Profile with GPU timelines before optimizing: a profiled step,
//            timed by the GPU's own timestamps in the shape run() gives it.
//     TLM.1  Compile telemetry out by default: the profiled step is the
//            diagnostic build's alone.
//     TLM.8  Validate clean builds by artifact scan: the clean module has
//            no symbol of the profiled step.

// One byte: it travels only through in-process callbacks, never into a GPU
// buffer, a file or a message.
enum class ProgramError : std::uint8_t {
  Ok,
  // A launch description or the readback range was refused, named; or a
  // pipeline failed to compile, or building the buffers or bind groups
  // failed, with WebGPU's messages.
  Build,
  // A step was refused, or failed without running, or failed validation,
  // or the device reported an internal error.
  Step,
  // A numbered step failed once the device had said it was lost.
  DeviceLost,
  // The program was destroyed with work in flight, or WebGPU cancelled a
  // callback.
  Cancelled,
};

class Program;

// Bytes a run writes into a range of one of its upload's buffers before its
// pass: in the kernels suite, a test's activation or edit table. One
// wgpuQueueWriteBuffer, which copies the bytes before it returns, so
// `bytes` need live only for the call.
struct RunInput {
  Binding to;
  std::span<const std::byte> bytes;
};

using BuildCallback = void (*)(std::unique_ptr<Program> program,
                               ProgramError error, std::string_view message,
                               void* userdata);
// `message` is WebGPU's for a failed step, at most 1 KiB, and empty on
// success; `readback` the step's copy of the readback range, empty for a
// step that read none or failed. Both are valid only during the call.
using StepCallback = void (*)(ProgramError error, std::string_view message,
                              std::span<const std::byte> readback,
                              void* userdata);

// The most bytes a step reads back: the draw's record, 16. Each of the two
// slots a step may be outstanding in holds this much in the program's state,
// made at build.
inline constexpr std::uint64_t kMaxReadback = 16;

// A kernel as build() compiles it, for a launch: the step's declaration
// (step.wgsl), then the kernel, then the unpack of the format it reads and
// the pack of the format it writes, with its entry point and its override
// constants in name order — workgroup_size always, last_token for a launch
// of the last token, then the launch's own. Two launches with the same
// composition share a pipeline. Exposed so a tool can hand a browser
// exactly what the program compiles, to find which kernel its shader
// compiler refuses.
//
// Workaround (browser): WebKit fails a pipeline given a constant for an
// override its entry point does not use ("Compute library failed
// creation"), though WebGPU allows any override the module declares — and a
// module's entry points share its overrides, and the program sets
// last_token on every launch of the last token. So the entry point's body
// begins with a phony assignment of each constant the pipeline is given,
// `_ = name;`, which uses it and computes nothing. An entry point not found
// is left as it is, for the pipeline to refuse by name.
struct Composed {
  std::string source;
  std::string_view entry_point;
  std::vector<std::pair<std::string_view, double>> constants;
};
[[nodiscard]] Composed compose(const Launch& launch);

class Program {
 public:
  // Builds the program for `launches`, in the graph's order, over
  // `upload`'s buffers, reading back `readback` from each step that asks
  // for logits, where given. `done` is called once, with the program or a
  // failure. Precondition: `upload` has finished and outlives the program.
  // A description or a readback range the header's Contract does not admit
  // is refused, Build, named.
  static void build(const residency::Upload& upload,
                    std::vector<Launch> launches,
                    std::optional<Binding> readback, BuildCallback done,
                    void* userdata);

#if FRIZZLE_DIAGNOSTICS_ENABLED
  // A profiled step's GPU timestamps, in nanoseconds: its pass's beginning
  // and end, and, where the device grants timestamps inside a pass, the
  // end of each launch dispatched, by the launch's index in the graph.
  struct Timestamps {
    std::uint64_t begin_ns;
    std::uint64_t end_ns;
    std::span<const std::pair<std::uint32_t, std::uint64_t>> launches;
  };
  // `times` is valid only during the call; zero for a step refused or
  // failed.
  using ProfileCallback = void (*)(ProgramError error, std::string_view message,
                                   std::span<const std::byte> readback,
                                   const Timestamps& times, void* userdata);

  // Runs one step as run() does, but only its first `launches` launches
  // in the graph's order — all of them for a whole step — timed (above).
  // `done` is called once, in run order with other profiled steps.
  // Precondition: run()'s. Refused at once, Step, named: on a device that
  // granted no timestamp queries, beside an unprofiled step, for more
  // launches than the graph's, and for run()'s refusals.
  void run_profiled(const Step& step, std::uint32_t launches,
                    ProfileCallback done, void* userdata);

  // The graph's launches: a whole step's bound for run_profiled.
  [[nodiscard]] std::uint32_t launch_count() const noexcept;

  // Whether the device granted timestamp queries, read once at build:
  // without them run_profiled refuses every step.
  [[nodiscard]] bool can_profile() const noexcept;
#endif

  // Runs one step. `done` is called once: when the queue has finished it,
  // or its readback has mapped, or it has failed. Refused at once, Step,
  // unrun: a third step while two are outstanding, and a step of more than
  // kPrefillBlock tokens. A step past kMaxPositions fails without running.
  // Preconditions: step.tokens >= 1; every identifier below the vocabulary.
  void run(const Step& step, StepCallback done, void* userdata);

  // Runs one step as run() does, first writing `inputs` into their buffers.
  // An input outside its range, or not of whole words, fails the step,
  // Step, without running it. Preconditions: run()'s; every input's buffer
  // is the program's upload's.
  void run(const Step& step, std::span<const RunInput> inputs,
           StepCallback done, void* userdata);

  ~Program();
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&&) = delete;
  Program& operator=(Program&&) = delete;

  // What the program and its in-flight callbacks share; defined only where
  // the program is.
  struct State;

 private:
  Program() = default;
  std::shared_ptr<State> state_;  // shared with in-flight callbacks
};

}  // namespace frizzle::kernels
