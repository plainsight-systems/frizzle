#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace frizzle::gguf {

// Axis C: changes with the GGUF format.
//
// The byte source: how the reader (reader.h) and the array reads (arrays.h)
// reach a file's bytes, whether the caller holds all of the file or only its
// front.
//
// Contract
//   - A ByteSource knows the whole file's length, size(), whatever part of
//     it is resident. Every offset and length is checked against that
//     length, so a tensor region beyond the resident part can be validated.
//   - read(offset, out) fills `out` with the bytes from `offset` and returns
//     Ok; OutsideFile if any of the range lies past the end of the file;
//     NotResident if the range is inside the file but not supplied. On
//     anything but Ok, `out` holds nothing the caller should trust. A read
//     never touches memory past size() or past what is resident.
//   - NotResident is not a defect in the file: the reader reports how much
//     of the file it needs, and the caller supplies that much and reads
//     again. Nothing waits.
//   - size() and read() never throw. A ByteSource is used through a
//     reference; it is neither copied nor moved.
//   - MemoryByteSource reads caller-owned bytes: the first resident.size()
//     bytes of a file file_size long, precondition resident.size() <=
//     file_size. It holds a view and copies nothing at construction; the
//     bytes must outlive it. Tests pass a whole file; the browser passes the
//     front of the kept file and the kept file's size (src/wasm).
//
// Algorithm (MemoryByteSource::read)
//   1. If offset > size() or out.size() > size() - offset: OutsideFile.
//   2. If offset > resident.size() or out.size() > resident.size() - offset:
//      NotResident.
//   3. Copy out.size() bytes from resident[offset] into `out`; Ok.
//   Each test subtracts rather than adds, so a hostile offset cannot wrap.
//
// Design and cost
//   - One virtual call a read: the reader and the array reads hold a
//     ByteSource&, so every read is an indirect call. The reader reads
//     field by field, so a string array costs one call per element's
//     length: Gemma 3 1B's vocabulary is 262,144 calls, counted from its
//     element count. read_strings then reads the whole array in one call.
//   - A read copies its bytes once, into the caller's buffer, and allocates
//     nothing.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.35   A base class destructor is public and virtual, or protected:
//            public and virtual, so a source can be destroyed through its
//            interface.
//     C.67   A polymorphic class suppresses public copy and move: all four
//            are deleted, so a source cannot be sliced.
//     F.6    Declare noexcept what must not throw: size() and read(), in the
//            interface and so in every implementation.
//     ES.103 Don't overflow: the range tests subtract.
//     ES.42  Keep pointers simple, use span: the copy's source is a subspan
//            of the resident bytes, with no pointer arithmetic.
//     Enum.7 Specify an underlying type only when necessary: ReadStatus
//            names one byte, to save space.
//   C++ performance guidelines
//     WASM.9 Never assume the whole asset is addressable; validate every
//            offset and length against the authoritative total: size() is
//            that total, and residency is a separate answer.
//     WASM.4 Reduce indirect dispatch in hot paths: a read is one indirect
//            call, made once a field while the index is read and once a
//            chunk for an array, never once a weight or a value.

enum class ReadStatus : std::uint8_t {
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
// file `file_size` long.
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
      // Both fit a size_t: offset and want are within the resident span.
      const std::span<const std::byte> from = resident_.subspan(
          static_cast<std::size_t>(offset), static_cast<std::size_t>(want));
      std::memcpy(out.data(), from.data(), from.size());
    }
    return ReadStatus::Ok;
  }

 private:
  std::span<const std::byte> resident_;
  std::uint64_t file_size_;
};

}  // namespace frizzle::gguf
