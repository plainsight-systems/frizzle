#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/types.h"

namespace frizzle::gguf {

// Axis C: changes with the GGUF file format, or what the index records of it.
//
// The tensor index: what the reader produces from a GGUF file's header
// (gguf/reader.h) and every later stage reads: each tensor's name, type,
// shape and bytes in the file, and the metadata. Describe, the plan, routes,
// the tokenizer and the engine read the header only through it, and what it
// locates through the byte source.
//
// Contract
//   - Immutable once read: read_index fills it, and every accessor is const.
//     A default-constructed index is empty.
//   - Complete: every tensor in the file, in the file's order, including
//     types no format here runs. Whether a type runs is the capability
//     table's question; the index records what the file says.
//   - tensor(id): precondition, `id` came from this index. find(name): the
//     tensor so named, or none.
//   - Scalar and string metadata is decoded during the read. Each typed
//     accessor gives Ok and sets `out`; MissingKey for a key the file lacks;
//     WrongType for a key it declares with another type: different errors,
//     because one is a file that lacks a field and the other a file that
//     misdeclares it. `out` is untouched unless the result is Ok.
//     read_string's `out` views storage the index owns and lives as long as
//     it.
//   - Arrays are located, not decoded: read_array gives the element type,
//     count and byte range, to be read from the byte source. An array of
//     numbers of at most kMaxDecodedArray elements is decoded too, each
//     element's bits widened to 64, signed ones sign-extended, so describe,
//     which has the index alone, reads a per-layer value as it reads a
//     scalar.
//   - read_u32_array: a decoded array of counts, UINT32 elements or INT32
//     ones none negative; WrongType for any other array, a longer one, or a
//     scalar. read_f32_array: a decoded array of FLOAT32 elements, WrongType
//     otherwise.
//
// Algorithm
//   - find: a binary search of every tensor's id sorted by name, built once
//     by read_index.
//   - A metadata accessor: find the entry by key, a scan of the metadata;
//     MissingKey if absent; WrongType if its declared type is not the one
//     asked for (and, for the array readers, if its elements are not, or
//     were not decoded); else convert the stored value to `out`.
//
// Design and cost
//   - Gemma 3 1B's pinned file, counted: 340 tensors, so a find is about
//     nine comparisons; 38 metadata keys, so a metadata lookup compares at
//     most 38 keys. Its vocabulary, 262,144 strings, is located and read by
//     the tokenizer from the byte source; its scores and token types, 262,144
//     numbers each, are past kMaxDecodedArray and only located.
//   - Optimization: the sorted ids are one contiguous vector, built once,
//     so describe's lookups each take a binary search rather than a
//     comparison with every tensor's name.
//   - Decoding a short array costs at most kMaxDecodedArray × 8 bytes, 8
//     KiB, at read.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.4    Precisely and strongly typed interfaces: TensorId is its own
//            type, so a tensor's position cannot be passed where a layer or a
//            token is meant.
//     Con.1  Immutable by default: filled once by read_index, then read
//            through const accessors only.
//     E.27   Error codes systematically: MetadataError, telling a missing key
//            from a misdeclared one.
//     E.12   noexcept where a throw is unacceptable: a scalar accessor reads
//            its value with get_if, and a value stored in another
//            alternative than its declared type's, a defect in the reader,
//            aborts.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            MetadataError takes one byte, the smallest that holds it.

// The longest array of numbers the read decodes: far past Gemma 3 1B's 26
// layers.
inline constexpr std::uint64_t kMaxDecodedArray = 1024;

// A tensor's position in the index. Its own type, so a tensor identifier
// cannot be passed where a layer or a token is meant (I.4).
enum class TensorId : std::uint32_t {};

// One byte, all its values need: returned, never stored or sent as a number.
enum class MetadataError : std::uint8_t {
  Ok,
  MissingKey,
  WrongType,
};

// Where an array value lives in the file: its elements, after the header
// that declares their type and count.
struct ArrayLocation {
  ValueType element_type;
  std::uint64_t element_count;
  ByteRange bytes;
  // For an array of numbers of at most kMaxDecodedArray elements, each
  // element's bits widened to 64, signed ones sign-extended; empty
  // otherwise.
  std::vector<std::uint64_t> decoded;
};

// One metadata value as the file stores it. Integers widen to 64 bits and
// floats to double; `type` keeps what the file declared, so an accessor can
// refuse a key of the wrong type.
struct MetadataEntry {
  std::string key;
  ValueType type;
  std::variant<std::uint64_t, std::int64_t, double, bool, std::string,
               ArrayLocation>
      value;
};

class TensorIndex {
 public:
  // An empty index. read_index fills one.
  TensorIndex() = default;

  [[nodiscard]] std::span<const TensorEntry> tensors() const noexcept {
    return tensors_;
  }

  // Precondition: `id` came from this index.
  [[nodiscard]] const TensorEntry& tensor(TensorId id) const noexcept {
    return tensors_[static_cast<std::size_t>(id)];
  }

  // The tensor named `name`: a binary search of the ids in name order, about
  // nine comparisons for Gemma 3 1B's 340 tensors (the header's
  // Optimization).
  [[nodiscard]] std::optional<TensorId> find(
      std::string_view name) const noexcept;

  [[nodiscard]] MetadataError read_u32(std::string_view key,
                                       std::uint32_t& out) const noexcept;
  [[nodiscard]] MetadataError read_f32(std::string_view key,
                                       float& out) const noexcept;
  [[nodiscard]] MetadataError read_bool(std::string_view key,
                                        bool& out) const noexcept;
  // `out` views storage owned by the index and lives as long as it does.
  [[nodiscard]] MetadataError read_string(std::string_view key,
                                          std::string_view& out) const noexcept;
  [[nodiscard]] MetadataError read_array(std::string_view key,
                                         ArrayLocation& out) const noexcept;
  // A short array of counts, one a layer: UINT32 elements, or INT32 ones
  // none negative. WrongType for any other array, a longer one, or a scalar.
  // `out` is left untouched unless it succeeds.
  [[nodiscard]] MetadataError read_u32_array(
      std::string_view key, std::vector<std::uint32_t>& out) const;
  // A short array of values: FLOAT32 elements, decoded as read_u32_array
  // decodes counts. `out` is left untouched unless it succeeds.
  [[nodiscard]] MetadataError read_f32_array(std::string_view key,
                                             std::vector<float>& out) const;

 private:
  friend ReadResult read_index(ByteSource& source, TensorIndex& out);

  [[nodiscard]] const MetadataEntry* entry(std::string_view key) const noexcept;

  std::vector<TensorEntry> tensors_;
  std::vector<TensorId> by_name_;  // every tensor's id, in name order
  std::vector<MetadataEntry> metadata_;
};

}  // namespace frizzle::gguf
