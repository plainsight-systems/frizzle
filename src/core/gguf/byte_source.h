#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace frizzle::gguf {

// Contract 1: the byte source. A random-access source of file bytes that
// knows how long the whole file is.
//
// The reader validates every offset and length against `size()` — the
// authoritative total — not against whatever part of the file is resident.
// A parser holding one span over the bytes would either require the entire
// file in memory, or be unable to validate a tensor region beyond the part it
// holds.
//
// A read inside the file of bytes that are not resident is not a failure of
// the file. The source says so, and the reader reports how much of the file
// it needs, so the caller can fetch that much and read again. Nothing waits.

enum class ReadStatus {
  Ok,
  OutsideFile,  // some of the range lies beyond the end of the file
  NotResident,  // the range is inside the file, but not supplied
};

class ByteSource {
 public:
  virtual ~ByteSource() = default;

  ByteSource() = default;
  ByteSource(const ByteSource&) = delete;
  ByteSource& operator=(const ByteSource&) = delete;
  ByteSource(ByteSource&&) = delete;
  ByteSource& operator=(ByteSource&&) = delete;

  // Authoritative length of the whole file, regardless of what is resident.
  [[nodiscard]] virtual std::uint64_t size() const noexcept = 0;

  // Fills `out` from `offset`. On anything but Ok, `out` holds nothing the
  // caller should trust. Implementations never read past `size()`.
  [[nodiscard]] virtual ReadStatus read(std::uint64_t offset,
                                        std::span<std::byte> out) noexcept = 0;
};

// A source over caller-owned bytes: the first `resident.size()` bytes of a
// file `file_size` long. Tests pass a whole file; the browser passes the
// prefix it has fetched and the size the server reported.
class MemoryByteSource final : public ByteSource {
 public:
  explicit MemoryByteSource(std::span<const std::byte> whole_file) noexcept
      : MemoryByteSource(whole_file, whole_file.size()) {}

  // Precondition: resident.size() <= file_size.
  MemoryByteSource(std::span<const std::byte> resident,
                   std::uint64_t file_size) noexcept
      : resident_(resident), file_size_(file_size) {}

  [[nodiscard]] std::uint64_t size() const noexcept override {
    return file_size_;
  }

  [[nodiscard]] ReadStatus read(std::uint64_t offset,
                                std::span<std::byte> out) noexcept override {
    // Written so the addition cannot wrap: offset + out.size() could
    // overflow for a hostile offset.
    const auto want = static_cast<std::uint64_t>(out.size());
    if (offset > file_size_ || want > file_size_ - offset) {
      return ReadStatus::OutsideFile;
    }
    const auto held = static_cast<std::uint64_t>(resident_.size());
    if (offset > held || want > held - offset) {
      return ReadStatus::NotResident;
    }
    if (want != 0) {
      std::memcpy(out.data(), resident_.data() + offset,
                  static_cast<std::size_t>(want));
    }
    return ReadStatus::Ok;
  }

 private:
  std::span<const std::byte> resident_;
  std::uint64_t file_size_;
};

}  // namespace frizzle::gguf
