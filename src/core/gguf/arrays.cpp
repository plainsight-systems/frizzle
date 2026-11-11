// Axis C: changes with the GGUF format.
//
// Whole-array reads of the arrays the index located (arrays.h states the
// contract): a string array into a StringTable, int32 and float32 arrays
// into vectors.
//
// Contract
//   - Defines read_strings, read_int32s and read_float32s as arrays.h
//     states: WrongElementType for another element type; ShortRead unless
//     the array's bytes hold exactly its elements; NeedMoreBytes, with the
//     bytes needed from the file's start, for bytes not resident;
//     OffsetOverflow for a sum or product that does not fit in 64 bits;
//     ArrayTooLong for a byte length past what a size_t holds; StringTooLong
//     for an element past 2^20 bytes. `out` is replaced only on Ok.
//   - Element bytes are copied as they lie in the file: GGUF stores numbers
//     little-endian, and the build refuses a big-endian target.
//
// Algorithm
//   read_int32s and read_float32s (read_scalars, T the element type):
//     1. The element type must match (WrongElementType); count × sizeof(T),
//        checked, must equal the array's byte length (ShortRead).
//     2. Read the bytes 1 MiB at a time into one chunk buffer, appending
//        each chunk's values to a new vector; stop at the first read that
//        is not Ok and return its result.
//     3. Move the vector into `out`.
//   read_strings:
//     1. The element type must be String (WrongElementType).
//     2. Read the array's whole byte range in one read.
//     3. The count must be at most bytes / 8, as each element costs at
//        least its 8-byte length (ShortRead), before the count sizes
//        anything.
//     4. For each element: an 8-byte length, at most 2^20 (StringTooLong)
//        and within what is left (ShortRead); its text is appended to the
//        table's one buffer and its end recorded.
//     5. Every byte must have been consumed (ShortRead); move the table
//        into `out`.
//
// Design and cost
//   - Optimization (browser): numbers are read 1 MiB at a time
//     (kChunkBytes), so a hostile element count never allocates before its
//     bytes exist.
//   - Gemma 3 1B's arrays, counted from the tokenizer fixture, whose arrays
//     are the pinned file's (tests/fixtures/tokenizer/): the vocabulary is
//     262,144 strings in 4,412,518 bytes, 2,315,366 of them text; the
//     scores and token types are 1,048,576 bytes each, one chunk each.
//   - read_strings holds the raw array and the table at once. The table's
//     text buffer is reserved at the raw array's length, so for Gemma 3 1B
//     that is about 8.8 MB at the peak, plus a size_t an element of ends,
//     and the table keeps the 4.4 MB capacity for 2.3 MB of text. Its text
//     is copied once, from the raw bytes into the table's buffer, which is
//     zero-filled as it grows into that capacity.
//   - read_scalars copies each value twice: from the source into the chunk,
//     then into the vector.
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.42    Keep pointers simple, use span: every offset into a buffer is
//              a subspan, with no pointer arithmetic.
//     ES.48    Avoid casts: text moves from std::byte to char storage by
//              memcpy, with no reinterpret_cast.
//     SL.str.5 Use std::byte for bytes that are not characters: the raw
//              array stays std::byte; only the table's text is char.
//     SL.con.3 Avoid bounds errors: each length is checked against what is
//              left before the subspan that reads it.
//     ES.103   Don't overflow: count × element size and offset + length
//              are checked.
//     E.27     Use error codes systematically: every refusal is a ReadError
//              in a ReadResult.
//   C++ performance guidelines
//     WASM.1   Size linear memory and resist growing it: the chunked read
//              bounds what a hostile count can make the heap allocate.
//     COPY.6   memcpy only between trivially copyable types: std::byte,
//              char, std::int32_t and float.

#include "core/gguf/arrays.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

#include "core/gguf/checked.h"

namespace frizzle::gguf {
namespace {

// GGUF stores values little-endian; these reads copy them as they lie.
static_assert(std::endian::native == std::endian::little);

// Scalars are read this much at a time, so a hostile element count is never an
// allocation before its bytes exist. A multiple of every scalar's size.
// Optimization (browser): the wasm heap is one contiguous block, and an
// allocation it cannot hold aborts the module (WASM.1).
constexpr std::uint64_t kChunkBytes = 1u << 20;

constexpr std::uint64_t kLengthBytes = 8;  // each string's length prefix

ReadResult failed(ReadError error) { return ReadResult{error, 0}; }

// Reads `length` bytes from `offset` into `out`, sized to hold them.
ReadResult read_range(ByteSource& source, std::uint64_t offset,
                      std::uint64_t length, std::vector<std::byte>& out) {
  std::uint64_t end = 0;
  if (!checked_add(offset, length, end)) {
    return failed(ReadError::OffsetOverflow);
  }
  if (length > std::numeric_limits<std::size_t>::max()) {
    return failed(ReadError::ArrayTooLong);
  }
  out.resize(static_cast<std::size_t>(length));
  switch (source.read(offset, out)) {
    case ReadStatus::Ok:
      return {};
    case ReadStatus::NotResident:
      return ReadResult{ReadError::NeedMoreBytes, end};
    case ReadStatus::OutsideFile:
      return failed(ReadError::ShortRead);
  }
  return failed(ReadError::ShortRead);
}

template <typename T>
ReadResult read_scalars(ByteSource& source, const ArrayLocation& array,
                        ValueType type, std::vector<T>& out) {
  static_assert(kChunkBytes % sizeof(T) == 0);
  if (array.element_type != type) return failed(ReadError::WrongElementType);
  std::uint64_t total = 0;
  if (!checked_mul(array.element_count, sizeof(T), total)) {
    return failed(ReadError::OffsetOverflow);
  }
  if (total != array.bytes.length) return failed(ReadError::ShortRead);

  std::vector<T> values;
  std::vector<std::byte> chunk;
  for (std::uint64_t done = 0; done < total;) {
    const std::uint64_t length = std::min(kChunkBytes, total - done);
    if (const auto r =
            read_range(source, array.bytes.offset + done, length, chunk);
        r.error != ReadError::Ok) {
      return r;
    }
    const std::size_t first = values.size();
    values.resize(first + chunk.size() / sizeof(T));
    std::memcpy(std::span(values).subspan(first).data(), chunk.data(),
                chunk.size());
    done += length;
  }
  out = std::move(values);
  return {};
}

}  // namespace

ReadResult read_strings(ByteSource& source, const ArrayLocation& array,
                        StringTable& out) {
  if (array.element_type != ValueType::String) {
    return failed(ReadError::WrongElementType);
  }
  // Read in one call: reading the index already took every element's
  // length, so the array's bytes are in hand up to its last element's text.
  std::vector<std::byte> raw;
  if (const auto r =
          read_range(source, array.bytes.offset, array.bytes.length, raw);
      r.error != ReadError::Ok) {
    return r;
  }
  // Every element costs at least its length prefix: checked before the
  // count sizes anything.
  if (array.element_count > raw.size() / kLengthBytes) {
    return failed(ReadError::ShortRead);
  }

  StringTable table;
  table.bytes_.reserve(raw.size());
  table.ends_.reserve(static_cast<std::size_t>(array.element_count));
  std::size_t at = 0;
  for (std::uint64_t i = 0; i < array.element_count; ++i) {
    if (raw.size() - at < kLengthBytes) return failed(ReadError::ShortRead);
    std::uint64_t length = 0;
    std::memcpy(&length, std::span(raw).subspan(at, kLengthBytes).data(),
                sizeof length);
    at += kLengthBytes;
    if (length > kMaxStringLength) return failed(ReadError::StringTooLong);
    if (length > raw.size() - at) return failed(ReadError::ShortRead);
    // Within raw, so it fits a size_t.
    const auto size = static_cast<std::size_t>(length);
    const std::size_t begin = table.bytes_.size();
    table.bytes_.resize(begin + size);
    std::memcpy(std::span(table.bytes_).subspan(begin).data(),
                std::span(raw).subspan(at, size).data(), size);
    table.ends_.push_back(table.bytes_.size());
    at += size;
  }
  if (at != raw.size()) return failed(ReadError::ShortRead);
  out = std::move(table);
  return {};
}

ReadResult read_int32s(ByteSource& source, const ArrayLocation& array,
                       std::vector<std::int32_t>& out) {
  return read_scalars(source, array, ValueType::Int32, out);
}

ReadResult read_float32s(ByteSource& source, const ArrayLocation& array,
                         std::vector<float>& out) {
  return read_scalars(source, array, ValueType::Float32, out);
}

}  // namespace frizzle::gguf
