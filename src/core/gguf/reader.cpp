// Axis C: changes with the GGUF file format, or with this parse contract.
//
// Contract, algorithm, cost and corpus guidance: reader.h. This file
// defines what that header states.

#include "core/gguf/reader.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/gguf/checked.h"

namespace frizzle::gguf {
namespace {

// Little-endian decode. GGUF is little-endian on disk; doing this by hand
// rather than memcpy-ing into an integer keeps the code correct on a
// big-endian host without a conditional.
constexpr int kBitsPerByte = 8;

[[nodiscard]] std::uint64_t le(std::span<const std::byte> bytes) noexcept {
  std::uint64_t value = 0;
  for (auto it = bytes.rbegin(); it != bytes.rend(); ++it) {
    value = (value << kBitsPerByte) | std::to_integer<std::uint8_t>(*it);
  }
  return value;
}

// Fixed sizes for the scalar metadata types; zero for the variable-length
// ones, which are handled separately.
[[nodiscard]] std::uint64_t scalar_size(ValueType t) noexcept {
  switch (t) {
    case ValueType::UInt8:
    case ValueType::Int8:
    case ValueType::Bool:
      return sizeof(std::uint8_t);
    case ValueType::UInt16:
    case ValueType::Int16:
      return sizeof(std::uint16_t);
    case ValueType::UInt32:
    case ValueType::Int32:
    case ValueType::Float32:
      return sizeof(std::uint32_t);
    case ValueType::UInt64:
    case ValueType::Int64:
    case ValueType::Float64:
      return sizeof(std::uint64_t);
    case ValueType::String:
    case ValueType::Array:
      return 0;
  }
  return 0;
}

[[nodiscard]] bool known_value_type(std::uint32_t raw) noexcept {
  return raw <= static_cast<std::uint32_t>(ValueType::Float64);
}

// Sign-extends `raw`, read little-endian at the width of the signed type
// `t`, by narrowing to that width: the conversion is modular since C++20.
[[nodiscard]] std::int64_t sign_extend(std::uint64_t raw,
                                       ValueType t) noexcept {
  switch (t) {
    case ValueType::Int8:
      return static_cast<std::int8_t>(raw);
    case ValueType::Int16:
      return static_cast<std::int16_t>(raw);
    case ValueType::Int32:
      return static_cast<std::int32_t>(raw);
    default:
      return static_cast<std::int64_t>(raw);
  }
}

// True when some value appears twice. Sorts rather than comparing each value
// with every other: a file may declare up to 2^20 tensors or keys, and a
// quadratic check would let a hostile one stall the reader.
// Optimization (practice): work on untrusted input bounded at O(n log n).
[[nodiscard]] bool has_duplicate(std::vector<std::string_view> values) {
  std::sort(values.begin(), values.end());
  return std::adjacent_find(values.begin(), values.end()) != values.end();
}

template <typename Entry, typename Field>
[[nodiscard]] std::vector<std::string_view> views_of(
    const std::vector<Entry>& entries, Field field) {
  std::vector<std::string_view> views;
  views.reserve(entries.size());
  for (const Entry& e : entries) views.push_back(e.*field);
  return views;
}

// The alignment tensor data follows: general.alignment when the file declares
// it, the format's default when it does not. A declared value that is not a
// power-of-two uint32 makes the file malformed, not a reason to fall back to
// the default.
[[nodiscard]] ReadError resolve_alignment(
    const std::vector<MetadataEntry>& metadata, std::uint64_t& out) {
  out = kDefaultAlignment;
  for (const MetadataEntry& e : metadata) {
    if (e.key != "general.alignment") continue;
    if (e.type != ValueType::UInt32) return ReadError::BadAlignment;
    const std::uint64_t value = std::get<std::uint64_t>(e.value);
    if (value == 0 || (value & (value - 1)) != 0) {
      return ReadError::BadAlignment;
    }
    out = value;
  }
  return ReadError::Ok;
}

// True when two tensors' byte ranges intersect. Precondition: every range is
// inside the file, so no end wraps.
// Optimization (practice): sorted by offset, each range need only be checked
// against the one before it: O(n log n) on untrusted input, not O(n²).
[[nodiscard]] bool has_overlap(const std::vector<TensorEntry>& tensors) {
  std::vector<ByteRange> ranges;
  ranges.reserve(tensors.size());
  for (const TensorEntry& t : tensors) {
    ranges.push_back({t.data_offset, t.data_length});
  }
  std::sort(ranges.begin(), ranges.end(),
            [](const ByteRange& a, const ByteRange& b) {
              return a.offset < b.offset;
            });
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].offset < ranges[i - 1].offset + ranges[i - 1].length) {
      return true;
    }
  }
  return false;
}

// Walks the file front to back with a cursor. Each take_* advances the cursor
// only on success. A read that reaches bytes the source does not hold records
// how far into the file it needed to go, so the caller can supply that much.
class Reader {
 public:
  explicit Reader(ByteSource& source) noexcept : source_(&source) {}

  [[nodiscard]] ReadError parse();

  [[nodiscard]] std::uint64_t bytes_needed() const noexcept {
    return bytes_needed_;
  }
  [[nodiscard]] std::vector<TensorEntry> release_tensors() {
    return std::move(tensors_);
  }
  [[nodiscard]] std::vector<MetadataEntry> release_metadata() {
    return std::move(metadata_);
  }

 private:
  [[nodiscard]] ReadError take(std::span<std::byte> out);
  [[nodiscard]] ReadError take_uint(int bytes, std::uint64_t& out);
  [[nodiscard]] ReadError take_string(std::string& out);
  [[nodiscard]] ReadError skip(std::uint64_t count);
  [[nodiscard]] ReadError skip_string();
  [[nodiscard]] ReadError take_value(MetadataEntry& entry);
  [[nodiscard]] ReadError take_array(ArrayLocation& out);
  [[nodiscard]] ReadError take_tensor();
  [[nodiscard]] ReadError take_metadata(std::uint64_t count);
  [[nodiscard]] ReadError take_tensors(std::uint64_t count);
  [[nodiscard]] ReadError place_tensor_data(std::uint64_t alignment);

  // A pointer, not a reference member (C.12): never null, set once.
  ByteSource* source_;
  std::uint64_t cursor_ = 0;
  std::uint64_t bytes_needed_ = 0;
  std::vector<TensorEntry> tensors_;
  std::vector<MetadataEntry> metadata_;
};

ReadError Reader::take(std::span<std::byte> out) {
  const auto count = static_cast<std::uint64_t>(out.size());
  switch (source_->read(cursor_, out)) {
    case ReadStatus::Ok:
      cursor_ += count;
      return ReadError::Ok;
    case ReadStatus::NotResident:
      bytes_needed_ = cursor_ + count;  // inside the file, so no wrap
      return ReadError::NeedMoreBytes;
    case ReadStatus::OutsideFile:
      return ReadError::ShortRead;
  }
  return ReadError::ShortRead;
}

ReadError Reader::take_uint(int bytes, std::uint64_t& out) {
  std::array<std::byte, sizeof(std::uint64_t)> buf{};
  const std::span<std::byte> bytes_read =
      std::span(buf).first(static_cast<std::size_t>(bytes));
  if (const auto e = take(bytes_read); e != ReadError::Ok) return e;
  out = le(bytes_read);
  return ReadError::Ok;
}

ReadError Reader::take_string(std::string& out) {
  std::uint64_t length = 0;
  if (const auto e = take_uint(8, length); e != ReadError::Ok) return e;
  // Bounded before allocating: a hostile length must not be an allocation
  // request.
  if (length > kMaxStringLength) return ReadError::StringTooLong;
  if (!range_within(cursor_, length, source_->size())) {
    return ReadError::ShortRead;
  }
  out.resize(static_cast<std::size_t>(length));
  return take(std::as_writable_bytes(std::span(out.data(), out.size())));
}

// Advances past `count` bytes without reading them. Needs no bytes resident,
// only that the range lies inside the file.
ReadError Reader::skip(std::uint64_t count) {
  if (!range_within(cursor_, count, source_->size())) {
    return ReadError::ShortRead;
  }
  cursor_ += count;
  return ReadError::Ok;
}

ReadError Reader::skip_string() {
  std::uint64_t length = 0;
  if (const auto e = take_uint(8, length); e != ReadError::Ok) return e;
  if (length > kMaxStringLength) return ReadError::StringTooLong;
  return skip(length);
}

ReadError Reader::take_array(ArrayLocation& out) {
  std::uint64_t raw_type = 0;
  if (const auto e = take_uint(4, raw_type); e != ReadError::Ok) return e;
  if (!known_value_type(static_cast<std::uint32_t>(raw_type))) {
    return ReadError::UnknownValueType;
  }
  out.element_type = static_cast<ValueType>(raw_type);
  if (out.element_type == ValueType::Array) return ReadError::NestedArray;

  if (const auto e = take_uint(8, out.element_count); e != ReadError::Ok) {
    return e;
  }
  if (out.element_count > kMaxArrayLength) return ReadError::ArrayTooLong;

  out.bytes.offset = cursor_;
  if (out.element_type == ValueType::String) {
    // Variable-length elements must be walked to find the end: each
    // length is read, and the text it counts is skipped, not copied.
    for (std::uint64_t i = 0; i < out.element_count; ++i) {
      if (const auto e = skip_string(); e != ReadError::Ok) return e;
    }
  } else {
    std::uint64_t total = 0;
    if (!checked_mul(out.element_count, scalar_size(out.element_type), total)) {
      return ReadError::OffsetOverflow;
    }
    if (const auto e = skip(total); e != ReadError::Ok) return e;
  }
  out.bytes.length = cursor_ - out.bytes.offset;
  return ReadError::Ok;
}

ReadError Reader::take_value(MetadataEntry& entry) {
  switch (entry.type) {
    case ValueType::String: {
      std::string text;
      if (const auto e = take_string(text); e != ReadError::Ok) return e;
      entry.value = std::move(text);
      return ReadError::Ok;
    }
    case ValueType::Array: {
      ArrayLocation array{};
      if (const auto e = take_array(array); e != ReadError::Ok) return e;
      entry.value = array;
      return ReadError::Ok;
    }
    default:
      break;
  }
  const std::uint64_t size = scalar_size(entry.type);
  std::uint64_t raw = 0;
  if (const auto e = take_uint(static_cast<int>(size), raw);
      e != ReadError::Ok) {
    return e;
  }
  switch (entry.type) {
    case ValueType::Bool:
      entry.value = raw != 0;
      break;
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64:
      entry.value = sign_extend(raw, entry.type);
      break;
    case ValueType::Float32: {
      float f = 0;
      const auto bits = static_cast<std::uint32_t>(raw);
      std::memcpy(&f, &bits, sizeof f);
      entry.value = static_cast<double>(f);
      break;
    }
    case ValueType::Float64: {
      double d = 0;
      std::memcpy(&d, &raw, sizeof d);
      entry.value = d;
      break;
    }
    default:
      entry.value = raw;
      break;
  }
  return ReadError::Ok;
}

ReadError Reader::take_tensor() {
  // Fields are read into locals and the entry is constructed once, at the
  // end, complete. TensorEntry has no default constructor precisely so a
  // half-built entry cannot exist to be pushed by mistake.
  std::string name;
  if (const auto e = take_string(name); e != ReadError::Ok) return e;
  if (name.size() > kMaxTensorNameLength) return ReadError::TensorNameTooLong;

  TensorShape shape;
  std::uint64_t dimension_count = 0;
  if (const auto e = take_uint(4, dimension_count); e != ReadError::Ok) {
    return e;
  }
  if (dimension_count > kMaxDimensions) return ReadError::TooManyDimensions;
  shape.dimension_count = static_cast<std::uint32_t>(dimension_count);

  shape.element_count = 1;
  for (std::uint64_t& dimension :
       std::span(shape.dimensions).first(shape.dimension_count)) {
    std::uint64_t extent = 0;
    if (const auto e = take_uint(8, extent); e != ReadError::Ok) return e;
    // Stored as int64 on disk; a negative value arrives with the top bit
    // set and would otherwise become an enormous positive extent.
    if (static_cast<std::int64_t>(extent) < 0) {
      return ReadError::NegativeDimension;
    }
    dimension = extent;
    if (!checked_mul(shape.element_count, extent, shape.element_count)) {
      return ReadError::ElementCountOverflow;
    }
  }

  std::uint64_t raw_type = 0;
  if (const auto e = take_uint(4, raw_type); e != ReadError::Ok) return e;
  const auto type = static_cast<TensorType>(raw_type);
  const FormatLayout* layout = format_layout(type);
  if (layout == nullptr) return ReadError::UnknownTensorType;

  // Blocks never span rows, so every row must be whole blocks.
  if (shape.dimensions[0] % layout->block_elements != 0) {
    return ReadError::RowNotWholeBlocks;
  }

  ByteRange data;
  // Relative to the tensor data region; made absolute once the region's
  // start is known.
  if (const auto e = take_uint(8, data.offset); e != ReadError::Ok) return e;
  if (!checked_mul(shape.element_count / layout->block_elements,
                   layout->block_bytes, data.length)) {
    return ReadError::OffsetOverflow;
  }
  tensors_.emplace_back(std::move(name), type, shape, data);
  return ReadError::Ok;
}

ReadError Reader::take_metadata(std::uint64_t count) {
  metadata_.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    MetadataEntry entry{};
    if (const auto e = take_string(entry.key); e != ReadError::Ok) return e;
    if (entry.key.empty()) return ReadError::EmptyKey;
    std::uint64_t raw_type = 0;
    if (const auto e = take_uint(4, raw_type); e != ReadError::Ok) return e;
    if (!known_value_type(static_cast<std::uint32_t>(raw_type))) {
      return ReadError::UnknownValueType;
    }
    entry.type = static_cast<ValueType>(raw_type);
    if (const auto e = take_value(entry); e != ReadError::Ok) return e;
    metadata_.push_back(std::move(entry));
  }
  if (has_duplicate(views_of(metadata_, &MetadataEntry::key))) {
    return ReadError::DuplicateMetadataKey;
  }
  return ReadError::Ok;
}

ReadError Reader::take_tensors(std::uint64_t count) {
  tensors_.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    if (const auto e = take_tensor(); e != ReadError::Ok) return e;
  }
  if (has_duplicate(views_of(tensors_, &TensorEntry::name))) {
    return ReadError::DuplicateTensorName;
  }
  return ReadError::Ok;
}

// Tensor data begins at the next alignment boundary after the index. Each
// offset must be a multiple of the alignment (the format's rule), is made
// absolute, and must lie inside the file without overlapping another.
ReadError Reader::place_tensor_data(std::uint64_t alignment) {
  std::uint64_t data_start = 0;
  const std::uint64_t pad = (alignment - (cursor_ % alignment)) % alignment;
  if (!checked_add(cursor_, pad, data_start)) return ReadError::OffsetOverflow;
  if (data_start > source_->size()) return ReadError::TensorDataOutOfBounds;

  for (auto& t : tensors_) {
    if (t.data_offset % alignment != 0) return ReadError::MisalignedTensorData;
    std::uint64_t absolute = 0;
    if (!checked_add(data_start, t.data_offset, absolute)) {
      return ReadError::OffsetOverflow;
    }
    if (!range_within(absolute, t.data_length, source_->size())) {
      return ReadError::TensorDataOutOfBounds;
    }
    t.data_offset = absolute;
  }
  if (has_overlap(tensors_)) return ReadError::OverlappingTensorData;
  return ReadError::Ok;
}

ReadError Reader::parse() {
  std::array<std::byte, sizeof kMagic> magic{};
  if (const auto e = take(magic); e != ReadError::Ok) return e;
  if (!std::ranges::equal(magic, kMagic, {}, [](std::byte b) {
        return std::to_integer<char>(b);
      })) {
    return ReadError::BadMagic;
  }

  std::uint64_t version = 0;
  if (const auto e = take_uint(4, version); e != ReadError::Ok) return e;
  if (version != kSupportedVersion) return ReadError::UnsupportedVersion;

  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  if (const auto e = take_uint(8, tensor_count); e != ReadError::Ok) return e;
  if (const auto e = take_uint(8, metadata_count); e != ReadError::Ok) return e;
  // Bounded before reserving: a declared count is a claim, not a fact.
  if (tensor_count > kMaxTensorCount) return ReadError::CountTooLarge;
  if (metadata_count > kMaxMetadataCount) return ReadError::CountTooLarge;

  if (const auto e = take_metadata(metadata_count); e != ReadError::Ok) {
    return e;
  }
  std::uint64_t alignment = 0;
  if (const auto e = resolve_alignment(metadata_, alignment);
      e != ReadError::Ok) {
    return e;
  }
  if (const auto e = take_tensors(tensor_count); e != ReadError::Ok) return e;
  return place_tensor_data(alignment);
}

}  // namespace

ReadResult read_index(ByteSource& source, TensorIndex& out) {
  Reader reader{source};
  const ReadError error = reader.parse();
  if (error != ReadError::Ok) return ReadResult{error, reader.bytes_needed()};
  out.tensors_ = reader.release_tensors();
  out.metadata_ = reader.release_metadata();
  // Name order, for find: built once, O(T log T), the same order of cost as
  // the duplicate check's sort, which sorts its own copy of the names.
  out.by_name_.resize(out.tensors_.size());
  for (std::size_t i = 0; i < out.by_name_.size(); ++i) {
    out.by_name_[i] = static_cast<TensorId>(i);
  }
  std::sort(out.by_name_.begin(), out.by_name_.end(),
            [&](TensorId a, TensorId b) {
              return out.tensor(a).name < out.tensor(b).name;
            });
  return ReadResult{};
}

}  // namespace frizzle::gguf
