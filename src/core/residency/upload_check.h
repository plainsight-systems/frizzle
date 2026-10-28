#pragma once

#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/gpu/wgpu_handles.h"
#include "core/residency/piece_writer.h"
#include "core/residency/upload.h"

namespace frizzle::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The Upload stage's own test, in a diagnostic build only: every byte upload
// wrote is read back from the device and compared, byte for byte, with what
// it should be. A sampled check passes while another chunk sits at the wrong
// offset; WASM.9 asks for every byte or no claim of integrity.
//
//   - What each range should hold is regenerated, not remembered: once upload
//     has finished, the page streams the cached file a second time, and a
//     fresh PieceWriter over the routes the Upload itself used — taken from
//     it, never passed in beside it, so the check cannot be pointed at other
//     routes and pass having compared nothing — turns it into the same writes
//     upload issued; it is deterministic (piece_writer.h). Nothing the size of
//     the model is kept to compare against.
//   - Every comparison rests on a completed mapping, which a lost device
//     refuses (upload.h), so a check on a lost device fails and never ends
//     with zero mismatches having compared nothing. A chunk's mapping is read
//     by mapping_result (mapping.h), the witness's own classifier: Ok is
//     compared; Cancelled, Internal and DeviceLost are themselves;
//     Unexplained is MapFailed. It holds its own references to the device
//     and the instance, as Upload does. Its callbacks keep
//     their state alive on their own, as Upload's do: its destructor marks
//     them cancelled, so destroying a check with a comparison pending reports
//     Cancelled instead of touching freed memory.
//   - For each chunk, every write's range is copied into one mappable staging
//     buffer, back to back, in one command buffer; the staging buffer is
//     mapped once, and each range is compared with the write's bytes. The
//     chunk is acknowledged only after that comparison, so the writer's
//     staging the writes point into is still intact when compared. The
//     readback is bounded by a chunk, as the upload was.
//   - The result names each piece that differs: the buffer, the offset of
//     its first differing byte, and the tensor the route belongs to, by the
//     name the Upload kept. A write may span pieces and the padding between
//     them (piece_writer.h); each piece in it is reported on its own, and a
//     byte of padding that differs is reported with no tensor. Zero
//     mismatches means every byte written is on the device where the plan
//     put it. Padding upload never wrote is not compared: WebGPU zeroes new
//     buffers, and nothing reads past a piece's file bytes but its unpack.
//
// Compiled only with diagnostics: CMake leaves its source out of every other
// build, so the shipped module holds none of it. A load timed
// in that build is a diagnostic figure, never a load-throughput one (TLM.6).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at run
//            time — where the bytes landed is checked on the device itself.
//     R.1    Manage resources automatically using RAII — the staging buffer
//            and the device reference are RAII handles.
//     E.27   Use error codes systematically — CheckError, and mismatches as
//            data, never a log line.
//     C.21   If you define or =delete any copy, move, or destructor
//            function, define or =delete them all — its destructor cancels
//            pending callbacks, so copy and move are each deleted, not left
//            implicit; its owner holds it in place.
//   C++ performance guidelines
//     COPY.4 Never declare only a destructor — the same four deletions.
//     WASM.9 Stream assets in bounded chunks; its Caveats: "A sampled
//            verification proves very little... Verify every byte, in a
//            diagnostic pass, or do not claim integrity." — every byte written
//            is compared, none sampled, a chunk at a time.
//     GPU.1  Budget every round trip — the whole model is read back once, in
//            this build only.
//     TLM.6  Diagnostic mode is not benchmark mode — what this build times
//            is labelled diagnostic and kept apart from the release figures.

enum class CheckError {
  Ok,
  Cancelled,   // the check was destroyed before the comparison ended
  DeviceLost,  // a mapping was aborted, and the device reports itself lost
  MapFailed,   // a mapping was aborted, and the device has not said why
  Internal,    // a mapping this check should not have asked for, or one with no
               // range
  OutOfOrder,  // a chunk did not start where the last one ended
  ChunkTooLarge,
  Unfinished,  // the file ended before every route was compared
};

// Each value's name, for the page to report.
[[nodiscard]] constexpr std::string_view to_string(CheckError error) noexcept {
  switch (error) {
    case CheckError::Ok:
      return "ok";
    case CheckError::Cancelled:
      return "cancelled";
    case CheckError::DeviceLost:
      return "GPU device lost";
    case CheckError::MapFailed:
      return "a readback could not be mapped";
    case CheckError::Internal:
      return "internal error";
    case CheckError::OutOfOrder:
      return "out of order";
    case CheckError::ChunkTooLarge:
      return "chunk too large";
    case CheckError::Unfinished:
      return "the file ended before every route was compared";
  }
  return "unknown";
}

struct Mismatch {
  BufferIndex buffer;
  std::uint64_t offset;  // the first differing byte, within the buffer
  std::string tensor;
};

using CheckCallback = void (*)(CheckError error, void* userdata);

class UploadCheck {
 public:
  UploadCheck(const UploadCheck&) = delete;
  UploadCheck& operator=(const UploadCheck&) = delete;
  UploadCheck(UploadCheck&&) = delete;
  UploadCheck& operator=(UploadCheck&&) = delete;

  // Checks the buffers `upload` filled, with the routes it used, naming
  // tensors by the names it kept. Takes its own references to upload's
  // device and instance (Upload::instance()) with gpu::retain, as do its
  // callbacks' shared state; tested on Dawn by releasing the caller's
  // references with a comparison pending, which still completes once,
  // never CallbackCancelled. Preconditions: `upload` has finished, and outlives
  // the check — it owns the buffers being checked, so it must anyway.
  // `max_chunk` as for Upload.
  UploadCheck(const Upload& upload, std::size_t max_chunk);

  // Marks pending callbacks cancelled, then releases the staging buffer.
  ~UploadCheck() noexcept;

  // Compares the ranges the chunk at `file_offset` covers; `accepted` once
  // they are compared and the page may send the next.
  void check(std::uint64_t file_offset, std::span<const std::byte> chunk,
             CheckCallback accepted, void* userdata);

  // Called after the last chunk: Unfinished unless every route was
  // compared. Like the writer's, Unfinished then refuses every chunk.
  [[nodiscard]] CheckError finish(std::uint64_t file_size);

  // Every mismatch found so far, in the order compared: chunk by chunk,
  // and within a chunk by route, then by stream.
  [[nodiscard]] std::span<const Mismatch> mismatches() const noexcept;

 private:
  gpu::Instance
      instance_;  // a reference of its own: its callbacks are the instance's
  gpu::DeviceHandle device_;
  const Upload& upload_;       // finished; outlives the check
  PieceWriter writer_;         // regenerates upload's writes
  std::vector<Write> writes_;  // a chunk's writes, handed to its comparison
  // Shared with callbacks in flight: the queue, the staging buffer
  // (MAP_READ | COPY_DST, sized to the most one chunk's writes can be),
  // the first failure, the mismatches, and whether the check is gone.
  std::shared_ptr<struct CheckState> pending_;
};

}  // namespace frizzle::residency
