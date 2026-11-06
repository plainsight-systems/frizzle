#pragma once

#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/mapping.h"
#include "core/residency/piece_writer.h"
#include "core/residency/plan.h"
#include "core/residency/routes.h"

namespace frizzle::residency {

// Axis D: changes with the WebGPU surface used to create and fill buffers, or
// how a device reports their failure.
//
// Carries out a residency plan on a device: creates every planned buffer,
// fills the weights from the file as it streams in, shows that the device
// holds them, and owns the buffers for the life of the loaded model. The KV
// cache and the kernels find them by their index in the plan. What goes where
// is decided without a device, by routes.h and piece_writer.h; this is the
// part that touches the GPU.
//
// Contract
//   - begin(device, index, plan, file_size, find_format, max_chunk, ready,
//     userdata): plans the routes (routes.h); a refusal calls `ready` with
//     null, its value and subject, before begin returns. Otherwise creates
//     every planned buffer and a 4-byte witness buffer inside out-of-memory,
//     validation and internal error
//     scopes, and calls `ready` once they pop: the first scope, in that
//     order, that caught an error gives its value and message; else a
//     creation that returned null is OutOfMemory naming the buffer; else the
//     Upload, Ok. `subject` is valid only during the call. Ready reports what
//     the scopes can show and no more: a lost device pops its scopes clean.
//   - Usage: weights STORAGE | COPY_DST, and COPY_SRC in a diagnostic build,
//     to be read back (upload_check.h); the cache STORAGE; working buffers
//     STORAGE | COPY_SRC | COPY_DST, read back (the sampled token, the
//     capture store) and written (the edit table, runtime/engine.cpp).
//     WebGPU zeroes a new buffer, so padding needs no write of its own.
//   - write(file_offset, chunk, accepted, userdata): queues the writes the
//     chunk completes (piece_writer.h) inside validation and internal scopes
//     pushed and popped within the call, and calls `accepted` when the page
//     may send the next chunk: at once for the first, and for chunk n once
//     the queue has finished chunk n − 1. A chunk sent before its
//     predecessor was accepted is OutOfOrder, and the chunk still waiting
//     hears it too; a write after finish is OutOfOrder and changes nothing.
//   - finish(done, userdata): called once, after the last chunk. `done` is
//     Ok only when the writer reached the file's end with every route
//     filled, every chunk's scopes popped clean, and the witness mapped with
//     the bytes written; otherwise the first failure. A second call is
//     OutOfOrder and changes nothing.
//   - Failure is a value, one for each cause, mapped from webgpu.h:
//       - the witness, by witness_result: its mapping read by mapping_result
//         (mapping.h), then its bytes. Ok with the expected four bytes is Ok,
//         the only Ok finish gives; Ok with other bytes is WitnessMismatch;
//         Cancelled, Internal and DeviceLost are themselves; Unexplained is
//         Unconfirmed.
//       - a popped scope's error type: OutOfMemory is OutOfMemory,
//         Validation is Validation, Internal, Unknown and any later type are
//         Internal.
//       - a scope pop that fails: CallbackCancelled (the instance went away)
//         is Cancelled; Error (no scope to pop, a defect here) is Internal.
//       - queued work done: CallbackCancelled is Cancelled; any other
//         failure is Internal.
//       - a chunk out of order, too large, or the file ending short, and
//         routes' refusals: their own values.
//     After a failure, every later call reports it and writes nothing; the
//     buffers are released with the Upload.
//   - Lifetime: what the callbacks need lives apart from the Upload, shared
//     between it and each callback in flight, with its own references to the
//     device and the instance. The destructor marks that state cancelled
//     before any member is released; each pending callback still runs
//     exactly once, reports Cancelled, and touches only that state. A
//     caller's userdata must stay valid until its callback has run, after
//     the Upload is destroyed if work was pending.
//   - The Upload, and the state its callbacks share, each hold a counted
//     reference to the device and its instance (gpu::retain,
//     wgpu_handles.h). WebGPU's callbacks are the instance's, and dropping
//     an instance's last reference cancels them, so the Upload and every
//     callback it queued outlive whoever else lets either go.
//   - Completion is reported through callbacks and never waited for: the
//     build does not use ASYNCIFY, and the worker stays responsive.
//   - Host memory the Upload allocates (its routes, the writer's staging) is
//     not an UploadError: in the wasm build a failed allocation aborts the
//     module, and the page reports that the engine's worker stopped
//     (app/src/protocol/start.ts). The load is never Ok. OutOfMemory is the
//     device's: a buffer it would not create.
//
// Algorithm
//   1. begin: routes and the carried plan; the Upload, its writer over its
//      own routes, and each routed tensor's name; push the three scopes;
//      create the buffers and the witness buffer; pop the scopes, innermost
//      first, keeping each result by the order pushed, and report.
//   2. write: refuse after finish, while a chunk waits, or after a failure;
//      the writer's writes for the chunk; push two scopes; one
//      wgpuQueueWriteBuffer each; pop the scopes (the first failure is kept);
//      ask the queue for work done; accept now if chunk n − 1 is done, else
//      keep the callback until it is.
//   3. finish: the writer's finish; once every chunk's scopes have popped,
//      report the first failure, or write the witness, four nonzero bytes
//      from the file's size, map it, and report witness_result. The queue
//      runs in order and no write was rejected, so a witness read back means
//      every write ran on a live device.
//
// Design and cost
//   - Success is shown, never assumed. A lost device still reports queued
//     work as done and error scopes as clean, and WebGPU does not order its
//     lost callback before them, so neither a clean status nor the absence
//     of a loss report is evidence. A completed mapping is: a lost device
//     refuses one. It costs one 4-byte write and one mapping a load, about
//     half a millisecond (an estimate, not measured in Frizzle).
//   - A chunk's scopes open and close within write: scopes are one stack per
//     device, so none is left open while the page reads, and nothing else
//     using the device between chunks pops the upload's scopes or has its
//     errors counted as the upload's. A rejected write never reaches the
//     queue, so only a scope shows it, for the chunk as a whole.
//   - Each write is one wgpuQueueWriteBuffer from the chunk or the writer's
//     staging in the heap; writeBuffer copies the bytes before it returns,
//     so both are free for the next chunk at once.
//   - Optimization: chunk n is accepted once chunk n − 1 is done, not chunk
//     n, so the page can read the next chunk while the GPU copies this one,
//     and the browser stages at most two chunks' bytes whatever happens.
//     Whether the two overlap is the browser's to decide; it is not measured
//     in Frizzle.
//   - Cost, counted for Gemma 3 1B in 16 MiB chunks (wasm/bindings.cpp): 44
//     chunks, each with two scopes pushed and popped and one work-done
//     callback; the writes as piece_writer.h counts them; one witness round
//     trip. The heap holds a chunk and the writer's staging, about the
//     chunk's size again (piece_writer.h); the plan, routes and names are
//     small beside them.
//   - Verification: begin and finish run on Dawn in the kernels suite, for
//     uploads with no weights (tests/support/scratch_program.h), and the
//     whole path in the oracle and replay on the pinned model. witness_result
//     and mapping_result are not tested branch by branch, and the failure
//     paths (a rejected write, a lost device, destruction with work pending)
//     are not tested.
//
// Corpus guidance
//   C++ Core Guidelines
//     R.1    RAII: every buffer is a gpu::Buffer and the device reference a
//            gpu::DeviceHandle, released with the Upload.
//     R.3    A raw pointer is non-owning: the device is held by a counted
//            handle, not a raw WGPUDevice whose lifetime is someone else's.
//     I.11, R.20  Never transfer ownership by a raw pointer; unique_ptr or
//            shared_ptr represent it: begin hands the Upload to `ready` by
//            unique_ptr.
//     R.21   unique_ptr unless ownership is shared: the callback state is the
//            one shared owner, since the Upload or a callback in flight may
//            end first.
//     C.21   Define or delete all of copy, move and destructor: its
//            destructor cancels pending callbacks, so copy and move are each
//            deleted; begin hands it out by unique_ptr.
//     E.27   Error codes systematically: UploadError, one value a cause, and
//            every call after a failure reports it.
//     E.26   Fail fast when recovery is impossible: a failed host allocation
//            aborts the module, and the page reports the worker stopped.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            UploadError takes one byte, the smallest that holds it.
//     SL.con.3 Avoid bounds errors: a scope's slot is read with at().
//     ES.45  Avoid magic constants: the witness's bytes are taken a byte,
//            CHAR_BIT bits, at a time.
//   C++ performance guidelines
//     GPU.9  Suballocate from large heaps: a few large buffers, as the plan
//            packs them.
//     GPU.7  Pipeline CPU and GPU work with bounded in-flight resources: two
//            chunks in flight, acknowledged on queue completion; its overlap
//            is claimed only once a timeline shows it, and none has.
//     WASM.3 Do not buy Asyncify to keep a blocking loop: completion arrives
//            by callback, and the worker returns to its event loop.
//     GPU.1  Budget every round trip: four bytes read back once a load, as
//            the proof of success.

// One value for each way upload can fail, routes' refusals among them, so a
// caller can tell a format this build lacks from a bad file (E.27). One byte,
// all its values need: the page is told its name, never its number.
enum class UploadError : std::uint8_t {
  Ok,
  Cancelled,          // the Upload was destroyed with this work pending
  UnsupportedFormat,  // routes.h: a tensor's format is not listed
  OutOfRange,         // routes.h: a piece reads past the file or its buffer
  RowNotSteppable,  // routes.h: rows no kernel can step through 32 weights at a
                    // time
  OutOfMemory,      // the out-of-memory scope caught an error
  Validation,   // the validation scope caught an error: a defect here, not in
                // the file
  Internal,     // an internal error, or a scope or queue error (see the mapping
                // above)
  DeviceLost,   // the witness was refused, and the device reports itself lost
  Unconfirmed,  // the witness was refused, and the device has not said why
  WitnessMismatch,  // the witness mapped, but held other bytes than were
                    // written
  OutOfOrder,  // a chunk did not start where the last one ended, or a call out
               // of turn: a chunk before the last was accepted, or after
               // finish, or finish twice
  ChunkTooLarge,
  Unfinished,  // the file ended before every weight was filled
};

// Each value's name, for the page to report.
[[nodiscard]] constexpr std::string_view to_string(UploadError error) noexcept {
  switch (error) {
    case UploadError::Ok:
      return "ok";
    case UploadError::Cancelled:
      return "cancelled";
    case UploadError::UnsupportedFormat:
      return "unsupported format";
    case UploadError::OutOfRange:
      return "out of range";
    case UploadError::RowNotSteppable:
      return "rows not a multiple of 32 weights";
    case UploadError::OutOfMemory:
      return "out of GPU memory";
    case UploadError::Validation:
      return "rejected by WebGPU validation";
    case UploadError::Internal:
      return "internal error";
    case UploadError::DeviceLost:
      return "GPU device lost";
    case UploadError::Unconfirmed:
      return "upload not confirmed by the device";
    case UploadError::WitnessMismatch:
      return "the device holds other bytes than were written";
    case UploadError::OutOfOrder:
      return "out of order";
    case UploadError::ChunkTooLarge:
      return "chunk too large";
    case UploadError::Unfinished:
      return "the file ended before every weight was filled";
  }
  return "unknown";
}

// What finish reports for its witness: the mapping's outcome, by
// mapping_result, then the bytes it gave back, compared with those written.
// Deterministic and device-free; not tested branch by branch (see above).
// `mapped` is consulted only when the mapping is Ok.
[[nodiscard]] UploadError witness_result(
    Mapped mapping, std::span<const std::byte> mapped,
    std::span<const std::byte, 4> expected) noexcept;

// Invoked exactly once per call that takes it: from the event loop in the
// browser, inside wgpuInstanceProcessEvents natively (gpu/callback_mode.h).
using UploadCallback = void (*)(UploadError error, void* userdata);

class Upload;

// Delivers the upload, or null and why, exactly once — the shape of
// gpu::Device::request. `subject` names the tensor or buffer at fault; it is
// valid only during the call.
using ReadyCallback = void (*)(std::unique_ptr<Upload> upload,
                               UploadError error, const char* subject,
                               void* userdata);

class Upload {
 public:
  Upload(const Upload&) = delete;
  Upload& operator=(const Upload&) = delete;
  Upload(Upload&&) = delete;
  Upload& operator=(Upload&&) = delete;

  // Plans the routes, creates the buffers, and calls `ready` once the
  // creation's error scopes have popped (see above). Ownership leaves by
  // unique_ptr, so nothing outlives the call by accident (I.11, R.20).
  // `max_chunk` is the largest chunk write will be given. The engine passes
  // capability::find_format as `find_format` (routes.h).
  // Takes its own reference to the device and shares its status.
  static void begin(const gpu::Device& device, const gguf::TensorIndex& index,
                    const ResidencyPlan& plan, std::uint64_t file_size,
                    FindFormat find_format, std::size_t max_chunk,
                    ReadyCallback ready, void* userdata);

  // Marks pending callbacks cancelled, then releases the buffers.
  ~Upload() noexcept;

  // Queues the writes the chunk at `file_offset` completes. `accepted` is
  // called when the page may send the next chunk: at once for the first,
  // and after the queue finishes the previous one for the rest.
  void write(std::uint64_t file_offset, std::span<const std::byte> chunk,
             UploadCallback accepted, void* userdata);

  // Called once, after the last chunk: `done` once every queued write has
  // completed, or with the error that stopped them. A second call, or a
  // write after it, is told OutOfOrder and changes nothing.
  void finish(UploadCallback done, void* userdata);

  // The plan as carried out.
  [[nodiscard]] const ResidencyPlan& plan() const noexcept { return plan_; }

  // The created buffers, by their index in the plan; null for an index the
  // plan does not have.
  [[nodiscard]] WGPUBuffer buffer(BufferIndex index) const noexcept;

  // The routes the upload carried out, in file order; upload_check.h
  // checks exactly these.
  [[nodiscard]] std::span<const Route> routes() const noexcept {
    return routes_;
  }

  [[nodiscard]] WGPUDevice device() const noexcept { return device_.get(); }

  // The instance the device came from. WebGPU has no way to find it from the
  // device, so a check of this upload takes its own reference from here.
  [[nodiscard]] WGPUInstance instance() const noexcept {
    return instance_.get();
  }

  // The device's status, to name the cause of a failure; never proof of
  // success (see above). Shared, as gpu::Device::status() is.
  [[nodiscard]] std::shared_ptr<const gpu::DeviceStatus> device_status()
      const noexcept {
    return device_status_;
  }

  // A routed tensor's name, kept from the index the routes came from, so a
  // report about a route needs no index beside the Upload.
  [[nodiscard]] std::string_view tensor_name(
      gguf::TensorId tensor) const noexcept;

 private:
  Upload(const gpu::Device& device, ResidencyPlan plan,
         std::vector<Route> routes, std::uint64_t file_size,
         std::size_t max_chunk);

  gpu::Instance
      instance_;  // a reference of its own: its callbacks are the instance's
  gpu::DeviceHandle device_;  // a reference of its own, taken in begin
  std::shared_ptr<const gpu::DeviceStatus> device_status_;
  std::vector<std::string>
      tensor_names_;  // by TensorId, for the routed tensors
  ResidencyPlan plan_;
  std::vector<Route> routes_;
  PieceWriter
      writer_;  // over routes_; the Upload never moves, so the span holds
  std::uint64_t file_size_;
  std::vector<gpu::Buffer> buffers_;
  std::vector<Write> writes_;  // reused across chunks
  std::uint64_t chunks_written_ = 0;
  bool finishing_ =
      false;  // finish was called; no write or second finish is taken
  // Shared with callbacks in flight: the queue, the witness buffer, the
  // first failure, the chunks' scopes still to report, and whether the
  // Upload is gone.
  std::shared_ptr<struct UploadState> pending_;
};

}  // namespace frizzle::residency
