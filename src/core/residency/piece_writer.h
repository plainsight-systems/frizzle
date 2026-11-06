#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/residency/routes.h"

namespace frizzle::residency {

// Axis D: changes with how the file's bytes reach the buffers the plan lays
// out.
//
// Turns the file, as it arrives in chunks, into the writes that fill the
// planned weight buffers, each piece in its format's device layout. A
// deterministic state machine with no I/O and no GPU object: it says what to
// write, not how. Upload issues each write as one wgpuQueueWriteBuffer
// (upload.h); the diagnostic check runs a second writer over the same routes
// to know what the device should hold (upload_check.h).
//
// Contract
//   - PieceWriter(routes, max_chunk): `routes` in file order, as plan_routes
//     gives them, outliving the writer; `max_chunk` the largest chunk accept
//     will be given. Allocates its staging once, staging_bound(routes,
//     max_chunk) bytes, and never grows it.
//   - accept(file_offset, chunk, out): appends to `out` the writes the chunk
//     completes. Chunks arrive in order, each starting where the last ended,
//     of any size up to max_chunk; a block, a row or a piece may span chunks.
//     A chunk out of order is OutOfOrder, one larger than max_chunk
//     ChunkTooLarge; on either, `out` is left as it was.
//   - finish(file_size): Ok only if the chunks reached `file_size` and every
//     route was filled; otherwise Unfinished.
//   - The first failure, Unfinished included, is final: every later accept
//     and finish returns it and writes nothing.
//   - A Write's bytes borrow the writer's staging: valid until the next call
//     to accept. Ownership never passes to the caller.
//   - Every write starts on a 4-byte boundary and is a multiple of 4 bytes
//     long, as writeBuffer requires, and lies inside its piece's bound range
//     or the plan's padding before it.
//   - Once every chunk is accepted, each piece's range holds, for each of its
//     layout's streams in order, that field of every block in block order,
//     the last stream's run padded with zeros to the word
//     (formats/device_layout.h). Bytes outside every route (the header, the
//     padding between tensors) are skipped.
//   - The same chunks, of the same sizes, give the same writes.
//
// Algorithm, one chunk
//   1. Check the chunk's size and that it starts at the next expected offset.
//   2. Walk the routes from the first unfilled one while bytes remain: skip
//      to the route's next byte (its first block, plus blocks done, plus any
//      held bytes); take the chunk's bytes up to the route's end.
//   3. If part of a block was held from the last chunk, complete it from the
//      front; if the chunk ends first, hold the bytes and go on.
//   4. Lay out the held block and the whole blocks that follow
//      (write_blocks):
//        a. if this piece follows one this call finished, in the same buffer,
//           at most kMaxJoinedPadding bytes on, with every buffer's pieces in
//           route order (checked once, at construction), stage that gap as
//           zeros and emit it;
//        b. for each stream s: stage its tail (bytes held back from its last
//           write), then field s of every block, at width w_s each;
//        c. emit each stream's run: when the piece is finished, padded with
//           zeros to the word; otherwise its whole words, holding back the
//           rest as the stream's tail. The device offset is the piece's offset
//           + Σ earlier streams' widths × the piece's blocks + blocks done ×
//           w_s − the tail's length.
//      emit joins a write to the last one when it continues it both in the
//      buffer and in staging.
//   5. Hold what is left, less than a block, for the next chunk.
//
// Design and cost
//   - What is held between chunks is at most one block (34 bytes, Q8_0's)
//     and up to 3 bytes for each stream: it does not grow with the file.
//   - Writes, not blocks, are the unit of cost: each crosses into the browser
//     and is validated there, about half a millisecond a call whatever its
//     length (an estimate, not measured in Frizzle). A chunk takes a write
//     for each buffer it reaches and one more for each stream of a piece it
//     cuts, rather than one for each stream of every piece: within a buffer,
//     a finished piece's streams are contiguous and the plan's padding to
//     the next piece, under 256 bytes, is written as zeros, so they join. The
//     zeros are what WebGPU created the buffer holding, so the buffer ends
//     byte for byte as it would without the join. A gap inside a piece,
//     which a later chunk fills, is never joined. Gemma 3 1B, counted: 342
//     pieces, 527 stream runs (two for each of 185 quantized pieces, one for
//     each of 157 F32 norms), streamed in 44 chunks of 16 MiB.
//   - Every run is copied into staging, even one already in device order
//     (F32): a layout whose one stream is its whole block is copied whole,
//     which costs far less than the write it would otherwise take on its own.
//   - Optimization: one staging area, allocated once and reused, so the heap
//     holds a chunk and its staging whatever the file's size. Its size is
//     proved from the routes: one chunk stages at most the chunk itself, plus
//     one block held over, plus, for each route the chunk reaches, up to 3
//     bytes held back and 3 of padding in each stream and the padding joined
//     before it. Every route could end inside one chunk (a file of adjacent
//     one-block tensors does), so the bound counts them all:
//       max_chunk + kMaxBlockBytes + routes × (kMaxStreams × 6 +
//       kMaxJoinedPadding),
//     91,690 bytes beyond the chunk for Gemma 3 1B's 342 routes, counted.
//     Taking more than the bound aborts (piece_writer.cpp, take): the proof
//     would be wrong.
//   - Verification: nothing tests the writer alone, the staging bound
//     included. The oracle and replay stream the pinned model through it and
//     check what the model then computes.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.2    A class with an invariant: the members below, each with the
//            invariant it keeps.
//     E.4    Error handling around invariants: after a failure the writer
//            accepts nothing, so no write follows a broken invariant.
//     E.27   Error codes systematically: WriteError.
//     R.14   Prefer span: chunks in, writes out, routes held, as spans.
//     I.11   Never transfer ownership by a raw pointer or reference: a
//            Write's span borrows staging the writer keeps.
//     SL.con.3 Avoid bounds errors: blocks and staging are reached through
//            spans, a stream's state with at(), never by pointer arithmetic.
//     ES.45  Avoid magic constants: the bytes held back and padded in each
//            stream, and the field widths copied at a fixed size, are named.
//     ES.46  Avoid narrowing conversions: offsets into staging are sizes.
//     ES.75  Avoid do-statements: the walk past empty routes is a while.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            WriteError takes one byte, the smallest that holds it.
//   C++ performance guidelines
//     WASM.1 Size the heap to the high-water mark: one chunk and its staging,
//            the staging allocated once at its proved bound.
//     WASM.2 Move a phase per crossing: writes joined across streams and
//            pieces, never one a block. Its caveat, that a batch blurs which
//            item failed, holds: a write the browser rejects is reported for
//            its chunk as a whole (upload.h).
//     WASM.9 Stream in bounded chunks: what is held is at most one block.
//     MEM.9  Allocate at init, not in steady state: the staging area.
//     CDSA.32 Transform static data once into the layout its consumer reads:
//            this is that transform, a chunk at a time, so source and result
//            never coexist whole. Its conversion is not tested against the
//            stored blocks alone; the oracle's logits cover it end to end.

// One write: `bytes` at `offset` in `buffer`. The span borrows, never owns:
// it points into the writer's staging, valid until the next call to accept.
// A caller that holds writes past the call — the diagnostic check does, until
// its comparison completes — makes no other call to accept until then (I.11:
// ownership stays with the caller).
struct Write {
  BufferIndex buffer;
  std::uint64_t offset;
  std::span<const std::byte> bytes;
};

// One byte, all its values need: returned, never stored or sent as a number.
enum class WriteError : std::uint8_t {
  Ok,
  OutOfOrder,     // a chunk did not start where the last one ended
  ChunkTooLarge,  // larger than the staging the writer was given
  Unfinished,     // the file ended before every route was filled
};

// The most padding a join writes: WebGPU's default storage-offset alignment,
// the largest a device may require, so the most bytes the plan leaves
// between two pieces of a buffer.
inline constexpr std::size_t kMaxJoinedPadding = 256;

class PieceWriter {
 public:
  // The most one chunk's writes can stage, by the bound above: what the
  // writer allocates, and what a reader of its writes must hold.
  [[nodiscard]] static constexpr std::size_t staging_bound(
      std::size_t routes, std::size_t max_chunk) noexcept {
    return max_chunk + formats::kMaxBlockBytes +
           routes * (formats::kMaxStreams * (kMaxTailBytes + kMaxRunPadding) +
                     kMaxJoinedPadding);
  }

  // `routes` in file order, as plan_routes gives them; `max_chunk` the
  // largest chunk accept will be given. Precondition: `routes` outlives the
  // writer.
  PieceWriter(std::span<const Route> routes, std::size_t max_chunk);

  // The writes the chunk at `file_offset` completes, appended to `out`. On
  // failure `out` is left as it was and the writer accepts nothing more.
  [[nodiscard]] WriteError accept(std::uint64_t file_offset,
                                  std::span<const std::byte> chunk,
                                  std::vector<Write>& out);

  // Called once the file has ended at `file_size`: Unfinished unless the
  // chunks reached it and every route was filled. Unfinished is a failure like
  // the others: from then on accept refuses every chunk.
  [[nodiscard]] WriteError finish(std::uint64_t file_size);

 private:
  // The most bytes of a stream's run held back, because it ended partway
  // through a word, and the most zeros that pad a finished run to the word:
  // each one less than a word's 4 bytes.
  static constexpr std::size_t kMaxTailBytes = 3;
  static constexpr std::size_t kMaxRunPadding = 3;

  // Bytes of one stream's run held back because it ended partway through a
  // word; they lead that stream's next write.
  struct Tail {
    std::array<std::byte, kMaxTailBytes> bytes{};
    std::uint8_t count = 0;
  };

  // Lays out route_'s next blocks — `held`, one block or none, then
  // `blocks` — as its streams, in staging_ past `staged`, which it
  // advances, and appends their runs to `out` through emit.
  void write_blocks(std::span<const std::byte> held,
                    std::span<const std::byte> blocks, std::size_t& staged,
                    std::vector<Write>& out);

  // Appends `w` to `out`, joined to the last write when it continues it in
  // both the buffer and staging.
  static void emit(const Write& w, std::vector<Write>& out);

  // The padding between the piece this call last finished, whose write
  // was the last, and route_'s, when the join rules above allow it to be
  // written; otherwise none.
  [[nodiscard]] std::optional<std::uint64_t> joinable_padding() const;

  std::span<const Route> routes_;
  std::size_t max_chunk_;
  std::size_t route_ = 0;          // the first route not yet filled
  std::uint64_t blocks_done_ = 0;  // of route_, the blocks already written
  std::uint64_t next_offset_ = 0;  // where the next chunk must start
  // A block of route_ cut off at the last chunk's end: fewer than its
  // layout's block_bytes, so never more than kMaxBlockBytes - 1.
  std::array<std::byte, formats::kMaxBlockBytes> held_{};
  std::size_t held_count_ = 0;
  std::array<Tail, formats::kMaxStreams>
      tails_{};  // one for each of route_'s streams
  // Whether every buffer's pieces lie in route order, each after the one
  // before it: padding is joined only if so. Set at construction.
  bool pieces_ascend_ = true;
  // The route this call finished last; route_ follows it.
  const Route* finished_ = nullptr;
  // Where rearranged bytes are written from: allocated once, at
  // construction, at the bound above, and never grown (MEM.9).
  std::vector<std::byte> staging_;
  // Set by the first failure; from then on accept refuses every chunk.
  WriteError failed_ = WriteError::Ok;
};

}  // namespace frizzle::residency
