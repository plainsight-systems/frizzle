#include "core/gguf/arrays.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
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
  if (!checked_add(offset, length, end))
    return failed(ReadError::OffsetOverflow);
  if (length > std::numeric_limits<std::size_t>::max())
    return failed(ReadError::ArrayTooLong);
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
  if (!checked_mul(array.element_count, sizeof(T), total))
    return failed(ReadError::OffsetOverflow);
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
    std::memcpy(values.data() + first, chunk.data(), chunk.size());
    done += length;
  }
  out = std::move(values);
  return {};
}

}  // namespace

ReadResult read_strings(ByteSource& source, const ArrayLocation& array,
                        StringTable& out) {
  if (array.element_type != ValueType::String)
    return failed(ReadError::WrongElementType);
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
  if (array.element_count > raw.size() / kLengthBytes)
    return failed(ReadError::ShortRead);

  StringTable table;
  table.bytes_.reserve(raw.size());
  table.ends_.reserve(static_cast<std::size_t>(array.element_count));
  std::size_t at = 0;
  for (std::uint64_t i = 0; i < array.element_count; ++i) {
    if (raw.size() - at < kLengthBytes) return failed(ReadError::ShortRead);
    std::uint64_t length = 0;
    std::memcpy(&length, raw.data() + at, sizeof length);
    at += kLengthBytes;
    if (length > kMaxStringLength) return failed(ReadError::StringTooLong);
    if (length > raw.size() - at) return failed(ReadError::ShortRead);
    const auto* text = reinterpret_cast<const char*>(raw.data() + at);
    table.bytes_.insert(table.bytes_.end(), text, text + length);
    table.ends_.push_back(table.bytes_.size());
    at += static_cast<std::size_t>(length);
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
