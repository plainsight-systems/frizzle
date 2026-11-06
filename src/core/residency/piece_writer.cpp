// Axis D: changes with how the file's bytes reach the buffers the plan lays
// out.
//
// Contract, algorithm, cost and corpus guidance: piece_writer.h. This file
// defines what that header states.

#include "core/residency/piece_writer.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "core/gguf/checked.h"

namespace frizzle::residency {
namespace {

using formats::DeviceLayout;

constexpr std::uint64_t round_up4(std::uint64_t n) { return (n + 3) / 4 * 4; }

// Where stream `s` of a piece of `blocks` blocks starts, from the piece's
// start: after every earlier stream's fields. Each earlier stream is a
// multiple of 4 wide (device_layout.h), so every stream starts on a word.
std::uint64_t stream_start(const DeviceLayout& layout, std::size_t s,
                           std::uint64_t blocks) {
  std::uint64_t before = 0;
  for (std::size_t i = 0; i < s; ++i) before += layout.streams[i].width;
  return before * blocks;
}

// Copies field `stream`, Width bytes wide, of every block in `run` into
// `out`, back to back, and returns the bytes copied.
template <std::size_t Width>
std::size_t gather_fixed(std::span<const std::byte> run,
                         std::size_t block_bytes, formats::Stream stream,
                         std::span<std::byte> out) {
  std::size_t at = 0;
  for (std::size_t b = 0; b < run.size(); b += block_bytes, at += Width) {
    std::memcpy(out.subspan(at, Width).data(),
                run.subspan(b + stream.offset, Width).data(), Width);
  }
  return at;
}

// The field widths the listed layouts have (formats/device_layout.h): a
// scale; Q4_1's scale and minimum; Q4_0's and Q4_1's 32 four-bit codes;
// Q8_0's 32 one-byte codes.
constexpr std::size_t kScaleWidth = 2;
constexpr std::size_t kScaleAndMinimumWidth = 4;
constexpr std::size_t kNibbleCodesWidth = 16;
constexpr std::size_t kByteCodesWidth = 32;

// Optimization: every field a listed layout has is one of these widths, so
// its copy has a size the compiler knows and inlines as a few loads and
// stores. Copied at a width known only at run time, each field would be a
// call to memcpy: Gemma 3 1B's Q4_0 weights are 21,058,560 blocks, counted,
// so as many calls a stream. For about 360 MiB of Q4_0 blocks, natively,
// the fixed widths take this stage from about 83 ms to about 12 ms, about
// half the speed of memcpy over the same bytes: an estimate, not measured
// in Frizzle.
std::size_t gather(std::span<const std::byte> run, std::size_t block_bytes,
                   formats::Stream stream, std::span<std::byte> out) {
  // A stream that is the whole block (F32) is the run itself, in order.
  // std::copy, not memcpy: an empty run's data may be null.
  if (stream.offset == 0 && stream.width == block_bytes) {
    std::copy(run.begin(), run.end(), out.begin());
    return run.size();
  }
  switch (stream.width) {
    case kScaleWidth:
      return gather_fixed<kScaleWidth>(run, block_bytes, stream, out);
    case kScaleAndMinimumWidth:
      return gather_fixed<kScaleAndMinimumWidth>(run, block_bytes, stream, out);
    case kNibbleCodesWidth:
      return gather_fixed<kNibbleCodesWidth>(run, block_bytes, stream, out);
    case kByteCodesWidth:
      return gather_fixed<kByteCodesWidth>(run, block_bytes, stream, out);
    default:
      break;  // a width no listed layout has: copied as it comes
  }
  std::size_t at = 0;
  for (std::size_t b = 0; b < run.size();
       b += block_bytes, at += stream.width) {
    std::memcpy(out.subspan(at, stream.width).data(),
                run.subspan(b + stream.offset, stream.width).data(),
                stream.width);
  }
  return at;
}

// Takes `n` bytes of staging past `staged`. The bound the staging is sized at
// is proved in piece_writer.h; exceeding it would mean the proof is wrong,
// which no caller can recover from, so it stops here rather than write past
// the area (E.26).
std::span<std::byte> take(std::span<std::byte> staging, std::size_t& staged,
                          std::size_t n) {
  if (n > staging.size() - staged) std::abort();
  const auto area = staging.subspan(staged, n);
  staged += n;
  return area;
}

}  // namespace

PieceWriter::PieceWriter(std::span<const Route> routes, std::size_t max_chunk)
    : routes_(routes),
      max_chunk_(max_chunk),
      staging_(staging_bound(routes.size(), max_chunk)) {
  while (route_ < routes_.size() && routes_[route_].blocks == 0) ++route_;
  // Where each buffer's pieces have reached, walking the routes in order.
  std::vector<std::uint64_t> reached;
  for (const Route& r : routes_) {
    const auto b = static_cast<std::size_t>(r.buffer);
    if (b >= reached.size()) reached.resize(b + 1, 0);
    if (r.buffer_offset < reached[b]) pieces_ascend_ = false;
    reached[b] = std::max(reached[b], r.buffer_offset + r.length);
  }
}

WriteError PieceWriter::accept(std::uint64_t file_offset,
                               std::span<const std::byte> chunk,
                               std::vector<Write>& out) {
  if (failed_ != WriteError::Ok) return failed_;
  if (chunk.size() > max_chunk_) return failed_ = WriteError::ChunkTooLarge;
  std::uint64_t end = 0;
  if (file_offset != next_offset_ ||
      !gguf::checked_add(file_offset, chunk.size(), end)) {
    return failed_ = WriteError::OutOfOrder;
  }

  std::size_t staged = 0;
  finished_ = nullptr;
  std::uint64_t pos = file_offset;
  while (route_ < routes_.size() && pos < end) {
    const Route& route = routes_[route_];
    const std::uint64_t block_bytes = route.layout->block_bytes;
    // The route's next byte; routes are in file order and apart, so the
    // chunk never starts past it. Bytes before it are outside every route.
    const std::uint64_t next =
        route.file_offset + blocks_done_ * block_bytes + held_count_;
    if (pos < next) {
      pos = std::min(end, next);
      continue;
    }
    const std::uint64_t stop =
        std::min(end, route.file_offset + route.blocks * block_bytes);
    auto bytes = chunk.subspan(pos - file_offset, stop - pos);
    pos = stop;

    // A block cut off at the last chunk's end is completed first.
    std::span<const std::byte> held;
    if (held_count_ > 0) {
      const auto fill =
          std::min<std::size_t>(block_bytes - held_count_, bytes.size());
      std::copy_n(bytes.begin(), fill,
                  std::span(held_).subspan(held_count_).begin());
      held_count_ += fill;
      bytes = bytes.subspan(fill);
      if (held_count_ < block_bytes) continue;  // the chunk ended inside it
      held = std::span<const std::byte>(held_).first(block_bytes);
    }
    const auto whole = bytes.first(bytes.size() / block_bytes * block_bytes);
    if (!held.empty() || !whole.empty()) write_blocks(held, whole, staged, out);
    held_count_ = 0;

    // What is left is less than a block, cut off at the chunk's end.
    const auto rest = bytes.subspan(whole.size());
    std::copy(rest.begin(), rest.end(), held_.begin());
    held_count_ = rest.size();
  }
  next_offset_ = end;
  return WriteError::Ok;
}

void PieceWriter::write_blocks(std::span<const std::byte> held,
                               std::span<const std::byte> blocks,
                               std::size_t& staged, std::vector<Write>& out) {
  const Route& route = routes_[route_];
  const DeviceLayout& layout = *route.layout;
  const std::uint64_t block_bytes = layout.block_bytes;
  const std::uint64_t count =
      held.size() / block_bytes + blocks.size() / block_bytes;
  const bool finished = blocks_done_ + count == route.blocks;

  // The plan's padding before this piece, written as zeros so its first
  // run joins the piece this call finished before it.
  if (const auto padding = joinable_padding()) {
    const auto zeros = take(staging_, staged, *padding);
    std::fill(zeros.begin(), zeros.end(), std::byte{0});
    emit({route.buffer, route.buffer_offset - *padding, zeros}, out);
  }

  // Each stream's run: the bytes its tail held back, then its fields of
  // these blocks, in staging. The areas are taken in stream order, so a
  // stream's write can join the one before it.
  struct Run {
    std::span<std::byte> area;  // in staging
    std::uint64_t device = 0;   // where it starts in the piece's buffer
    std::size_t length = 0;     // its bytes, before padding to the word
  };
  const std::size_t streams = layout.streams.size();
  std::array<Run, formats::kMaxStreams> runs{};
  for (std::size_t s = 0; s < streams; ++s) {
    const formats::Stream stream = layout.streams[s];
    const Tail& tail = tails_.at(s);
    Run& run = runs.at(s);
    // The tail holds the run's bytes past its last whole word, so the
    // write starts on the word they begin.
    run.device = route.buffer_offset + stream_start(layout, s, route.blocks) +
                 blocks_done_ * stream.width - tail.count;
    run.length = tail.count + static_cast<std::size_t>(count * stream.width);
    run.area = take(staging_, staged, round_up4(run.length));
    std::copy_n(tail.bytes.begin(), tail.count, run.area.begin());
  }
  for (std::size_t s = 0; s < streams; ++s) {
    const Run& run = runs.at(s);
    std::size_t at = tails_.at(s).count;
    for (const auto part : {held, blocks}) {
      at += gather(part, block_bytes, layout.streams[s], run.area.subspan(at));
    }
  }

  for (std::size_t s = 0; s < streams; ++s) {
    Tail& tail = tails_.at(s);
    const Run& run = runs.at(s);
    // The run that ends the piece is written whole, padded to the word,
    // which the piece's bound length covers (device_layout.h); any other
    // only to its last whole word, the rest held back as the tail.
    const std::size_t written = finished ? run.area.size() : run.length / 4 * 4;
    if (finished) {
      std::ranges::fill(run.area.subspan(run.length), std::byte{0});
      tail.count = 0;
    } else {
      tail.count = static_cast<std::uint8_t>(run.length - written);
      std::copy_n(run.area.subspan(written).begin(), tail.count,
                  tail.bytes.begin());
    }
    if (written > 0) {
      emit({route.buffer, run.device, run.area.first(written)}, out);
    }
  }

  blocks_done_ += count;
  if (finished) {
    finished_ = &route;
    blocks_done_ = 0;
    tails_ = {};
    ++route_;
    while (route_ < routes_.size() && routes_[route_].blocks == 0) {
      ++route_;
    }
  }
}

void PieceWriter::emit(const Write& w, std::vector<Write>& out) {
  // Every write of a call is staged back to back, so a write that
  // continues the last on the device also continues it in staging; the
  // last test is what makes the joined span hold both writes' bytes.
  if (!out.empty()) {
    Write& last = out.back();
    if (last.buffer == w.buffer &&
        last.offset + last.bytes.size() == w.offset &&
        std::to_address(last.bytes.end()) == w.bytes.data()) {
      last.bytes =
          std::span(last.bytes.data(), last.bytes.size() + w.bytes.size());
      return;
    }
  }
  out.push_back(w);
}

std::optional<std::uint64_t> PieceWriter::joinable_padding() const {
  if (!pieces_ascend_ || finished_ == nullptr) return std::nullopt;
  const Route& route = routes_[route_];
  const std::uint64_t end = finished_->buffer_offset + finished_->length;
  // Pieces that ascend put route_'s at or past the one before it.
  if (finished_->buffer != route.buffer ||
      route.buffer_offset - end > kMaxJoinedPadding) {
    return std::nullopt;
  }
  return route.buffer_offset - end;
}

WriteError PieceWriter::finish(std::uint64_t file_size) {
  if (failed_ != WriteError::Ok) return failed_;
  if (route_ < routes_.size() || next_offset_ != file_size) {
    return failed_ = WriteError::Unfinished;
  }
  return WriteError::Ok;
}

}  // namespace frizzle::residency
