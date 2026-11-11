#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace frizzle::gguf {

// Axis C: changes with the GGUF format.
//
// The GGUF format's vocabulary, as the reader (reader.h), the index (index.h)
// and the formats use it: the header constants, the numbering of metadata
// value types and tensor formats, each format's block layout, the reader's
// bounds and errors, and the record of one tensor.
//
// Contract
//   - Format facts, as GGUF defines them: the magic "GGUF"; version 3; data
//     aligned to 32 bytes unless general.alignment says otherwise; at most 4
//     dimensions; a tensor name at most 64 bytes.
//   - Frizzle's bounds, not the format's: at most 2^20 tensors, 2^20
//     metadata entries, 2^20 bytes in a string and 2^24 elements in an
//     array. Each stops a hostile file from making the reader allocate, and
//     exceeding one is a named ReadError.
//   - ValueType and TensorType carry GGUF's own numbers. The file stores
//     each as a uint32; here each takes one byte, as every number GGUF
//     defines is below 256. The reader refuses a value-type number GGUF does
//     not define, and a format number above 255, before converting it, so a
//     file's number never wraps onto a defined one; a format number that
//     fits but is undefined has no layout and is refused then (reader.cpp).
//   - TensorType lists every format GGUF defines, gaps included: a gap is a
//     number GGUF no longer defines, and a file that names one is
//     UnknownTensorType. Listing a format is not running it; which formats
//     run is the capability table's question (capability/capability.h).
//   - kFormatLayouts holds one row per TensorType: `block_bytes` bytes hold
//     `block_elements` consecutive values of a row, as the format defines
//     its block. An unquantized format is a block of one value.
//     format_layout(type) returns the row for `type`, or null for a number
//     GGUF does not define; the row lives for the whole program.
//   - ReadError is the closed set of ways reading can fail, in one byte;
//     to_string names each in a short sentence. ReadResult carries the
//     error and, with NeedMoreBytes, the bytes needed from the file's start.
//   - TensorEntry records one tensor: name, format, shape, element count and
//     byte range, as offsets and lengths, never pointers. It has no default
//     constructor, so no entry exists without a format. Dimensions past
//     dimension_count are 1. is_quantized's precondition: `type` has a
//     layout, which holds for every entry the reader made.
//   - Nothing here allocates but TensorEntry's name, and nothing throws but
//     that allocation.
//
// Algorithm
//   1. format_layout scans kFormatLayouts in order and returns the first
//      row whose type equals the argument; with no such row, null.
//   2. to_string maps each ReadError to its sentence; a value outside the
//      set, which only a defect could make, gives "unrecognised error".
//
// Design and cost
//   - The tables and constants are constexpr: constant data, no work at
//     startup. kFormatLayouts's length follows its initializers, so a row
//     cannot be left out by a miscounted size.
//   - format_layout is a linear scan of 35 rows: at most 35 comparisons of
//     a one-byte value. The reader calls it once a tensor; for Gemma 3 1B's
//     340 tensors that is at most 11,900 comparisons a load, counted.
//   - The K-quant, IQ and TQ formats, all but IQ4_NL, store blocks of 256
//     values (kSuperBlock).
//   - TensorEntry takes its shape and byte range as two small structs of
//     distinct types, so its four leading arguments cannot be swapped
//     silently, where four std::uint64_t parameters could.
//   - Quantization travels with the tensor: no weight reference exists
//     without its format, so no consumer sees bytes it cannot interpret.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.con.1 Prefer std::array to a C array: the magic, the layout table
//              and a shape's dimensions.
//     Con.5    Use constexpr for values computed at compile time: the
//              constants and the layout table.
//     Enum.7   Specify an enumeration's underlying type only when necessary:
//              ValueType, TensorType and ReadError name one byte, to save
//              space, as the rule's own example does.
//     ES.46    Avoid narrowing conversions: the file's 32-bit numbers are
//              checked to fit before they become a one-byte enumeration.
//     E.27     Use error codes systematically: ReadError and ReadResult are
//              the reader's one vocabulary of failure.
//     C.41     A constructor creates a fully initialized object: a
//              TensorEntry needs its format to exist.
//     NR.5     Don't use two-phase initialization: there is no entry whose
//              format is assigned later.
//     F.6      Declare noexcept what must not throw: format_layout and
//              to_string.
//   C++ performance guidelines
//     EMB.6    Push computation to compile time: the layout table is
//              constant data, read with no startup work.

inline constexpr std::array<char, 4> kMagic = {'G', 'G', 'U', 'F'};
inline constexpr std::uint32_t kSupportedVersion = 3;
inline constexpr std::uint32_t kDefaultAlignment = 32;
inline constexpr std::uint32_t kMaxDimensions = 4;
// The format's own limit on a tensor name.
inline constexpr std::uint64_t kMaxTensorNameLength = 64;

// Bounds that exist to stop a hostile file from making us allocate. They are
// not format limits, they are our limits, and exceeding one is a named error
// rather than an attempt that happens to fail.
inline constexpr std::uint64_t kMaxTensorCount = 1u << 20;
inline constexpr std::uint64_t kMaxMetadataCount = 1u << 20;
inline constexpr std::uint64_t kMaxStringLength = 1u << 20;
inline constexpr std::uint64_t kMaxArrayLength = 1u << 24;

// GGUF metadata value types, matching the format's own numbering.
enum class ValueType : std::uint8_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

// The weight formats GGUF defines, numbered as the format numbers them. Gaps
// are numbers GGUF no longer defines; a file that names one is malformed.
//
// Enumerating a format is not implementing it. The reader needs every
// format's block size to validate the file and to name a tensor it cannot
// run; which formats run is the capability table's business.
enum class TensorType : std::uint8_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
  TQ1_0 = 34,
  TQ2_0 = 35,
  MXFP4 = 39,
  NVFP4 = 40,
  Q1_0 = 41,
  Q2_0 = 42,
};

// How a format stores its values: `block_bytes` bytes hold `block_elements`
// consecutive values of a row. An unquantized format is a block of one.
struct FormatLayout {
  TensorType type;
  std::string_view name;
  std::uint32_t block_elements;
  std::uint32_t block_bytes;
};

// The values in one block of GGUF's K-quant, IQ and TQ formats, IQ4_NL
// aside.
inline constexpr std::uint32_t kSuperBlock = 256;

// Every format GGUF defines, with its block as the format defines it.
inline constexpr auto kFormatLayouts = std::to_array<FormatLayout>({
    {TensorType::F32, "F32", 1, 4},
    {TensorType::F16, "F16", 1, 2},
    {TensorType::Q4_0, "Q4_0", 32, 18},
    {TensorType::Q4_1, "Q4_1", 32, 20},
    {TensorType::Q5_0, "Q5_0", 32, 22},
    {TensorType::Q5_1, "Q5_1", 32, 24},
    {TensorType::Q8_0, "Q8_0", 32, 34},
    {TensorType::Q8_1, "Q8_1", 32, 36},
    {TensorType::Q2_K, "Q2_K", kSuperBlock, 84},
    {TensorType::Q3_K, "Q3_K", kSuperBlock, 110},
    {TensorType::Q4_K, "Q4_K", kSuperBlock, 144},
    {TensorType::Q5_K, "Q5_K", kSuperBlock, 176},
    {TensorType::Q6_K, "Q6_K", kSuperBlock, 210},
    {TensorType::Q8_K, "Q8_K", kSuperBlock, 292},
    {TensorType::IQ2_XXS, "IQ2_XXS", kSuperBlock, 66},
    {TensorType::IQ2_XS, "IQ2_XS", kSuperBlock, 74},
    {TensorType::IQ3_XXS, "IQ3_XXS", kSuperBlock, 98},
    {TensorType::IQ1_S, "IQ1_S", kSuperBlock, 50},
    {TensorType::IQ4_NL, "IQ4_NL", 32, 18},
    {TensorType::IQ3_S, "IQ3_S", kSuperBlock, 110},
    {TensorType::IQ2_S, "IQ2_S", kSuperBlock, 82},
    {TensorType::IQ4_XS, "IQ4_XS", kSuperBlock, 136},
    {TensorType::I8, "I8", 1, 1},
    {TensorType::I16, "I16", 1, 2},
    {TensorType::I32, "I32", 1, 4},
    {TensorType::I64, "I64", 1, 8},
    {TensorType::F64, "F64", 1, 8},
    {TensorType::IQ1_M, "IQ1_M", kSuperBlock, 56},
    {TensorType::BF16, "BF16", 1, 2},
    {TensorType::TQ1_0, "TQ1_0", kSuperBlock, 54},
    {TensorType::TQ2_0, "TQ2_0", kSuperBlock, 66},
    {TensorType::MXFP4, "MXFP4", 32, 17},
    {TensorType::NVFP4, "NVFP4", 64, 36},
    {TensorType::Q1_0, "Q1_0", 128, 18},
    {TensorType::Q2_0, "Q2_0", 64, 18},
});

// The layout of `type`, or null for a number GGUF does not define.
[[nodiscard]] constexpr const FormatLayout* format_layout(
    TensorType type) noexcept {
  for (const FormatLayout& layout : kFormatLayouts) {
    if (layout.type == type) return &layout;
  }
  return nullptr;
}

// The closed set of ways reading can fail. One vocabulary for callers (E.27);
// the reader never throws and never reads out of bounds.
enum class ReadError : std::uint8_t {
  Ok = 0,
  // The bytes lie inside the file but have not been supplied yet. Not a
  // defect in the file: supply the bytes ReadResult asks for and read again.
  NeedMoreBytes,
  ShortRead,  // the file ends before this field
  BadMagic,
  UnsupportedVersion,
  CountTooLarge,  // tensor or metadata count beyond our bound
  StringTooLong,
  ArrayTooLong,
  UnknownValueType,
  UnknownTensorType,  // a format number GGUF does not define
  RowNotWholeBlocks,  // a row is not a whole number of its format's blocks
  TooManyDimensions,
  NegativeDimension,
  ElementCountOverflow,   // the shape product does not fit
  OffsetOverflow,         // offset + length wraps
  TensorDataOutOfBounds,  // the region is not inside the file
  DuplicateTensorName,
  TensorNameTooLong,      // longer than the format's 64 bytes
  MisalignedTensorData,   // an offset that is not a multiple of the alignment
  OverlappingTensorData,  // two tensors claim the same bytes
  BadAlignment,           // general.alignment is not a power-of-two uint32
  EmptyKey,
  DuplicateMetadataKey,
  NestedArray,       // arrays of arrays are not supported
  WrongElementType,  // an array read as another element type than it holds
};

[[nodiscard]] constexpr std::string_view to_string(ReadError e) noexcept {
  switch (e) {
    case ReadError::Ok:
      return "ok";
    case ReadError::NeedMoreBytes:
      return "more of the file is needed to read its index";
    case ReadError::ShortRead:
      return "short read: the file ends before this field";
    case ReadError::BadMagic:
      return "not a GGUF file: magic mismatch";
    case ReadError::UnsupportedVersion:
      return "unsupported GGUF version";
    case ReadError::CountTooLarge:
      return "declared count exceeds the reader's bound";
    case ReadError::StringTooLong:
      return "string length exceeds the reader's bound";
    case ReadError::ArrayTooLong:
      return "array length exceeds the reader's bound";
    case ReadError::UnknownValueType:
      return "unknown metadata value type";
    case ReadError::UnknownTensorType:
      return "tensor format not defined by GGUF";
    case ReadError::RowNotWholeBlocks:
      return "tensor row is not a whole number of blocks";
    case ReadError::TooManyDimensions:
      return "tensor declares more than 4 dimensions";
    case ReadError::NegativeDimension:
      return "tensor declares a negative dimension";
    case ReadError::ElementCountOverflow:
      return "tensor element count overflows";
    case ReadError::OffsetOverflow:
      return "offset plus length overflows";
    case ReadError::TensorDataOutOfBounds:
      return "tensor data lies outside the file";
    case ReadError::DuplicateTensorName:
      return "duplicate tensor name";
    case ReadError::TensorNameTooLong:
      return "tensor name is longer than 64 bytes";
    case ReadError::MisalignedTensorData:
      return "tensor data is not aligned as the file declares";
    case ReadError::OverlappingTensorData:
      return "two tensors claim the same bytes";
    case ReadError::BadAlignment:
      return "general.alignment is not a power-of-two uint32";
    case ReadError::EmptyKey:
      return "a metadata key is empty";
    case ReadError::DuplicateMetadataKey:
      return "duplicate metadata key";
    case ReadError::NestedArray:
      return "nested arrays are not supported";
    case ReadError::WrongElementType:
      return "the array holds another element type";
  }
  return "unrecognised error";
}

// The outcome of reading a file's index.
struct ReadResult {
  ReadError error = ReadError::Ok;
  // Set with NeedMoreBytes: how many bytes from the start of the file the
  // next read needs to be supplied with.
  std::uint64_t bytes_needed = 0;
};

// A tensor's extent, grouped so it cannot be passed to a constructor in
// pieces or misordered against the byte range.
struct TensorShape {
  std::uint32_t dimension_count = 0;
  std::array<std::uint64_t, kMaxDimensions> dimensions = {1, 1, 1, 1};
  std::uint64_t element_count = 0;
};

// A byte region within the file.
struct ByteRange {
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
};

// One tensor's entry in the index. Holds offsets and lengths, never pointers,
// so it cannot outlive a buffer into undefined behaviour.
//
// Quantization travels WITH the tensor by construction: there is no way to
// obtain a weight reference without also obtaining what it takes to interpret
// it. A design where the loader "handles quantization" and consumers see plain
// bytes is the design that silently produces uninterpretable values.
struct TensorEntry {
  // There is no default constructor, and `type` has no default value.
  //
  // A default-constructed entry would be a zero-length F32 tensor: a record
  // that looks valid, reports is_quantized() == false, and is wrong. The
  // type must be supplied to build the object at all, rather than assigned
  // afterwards by a step somebody can forget (C.41, NR.5).
  //
  // The parameters are four mutually non-confusable types. A flat
  // constructor would take four interchangeable std::uint64_t values, which
  // trades one silent-misinitialization defect for a worse one.
  TensorEntry() = delete;

  TensorEntry(std::string tensor_name, TensorType tensor_type,
              const TensorShape& shape, const ByteRange& data)
      : name(std::move(tensor_name)),
        type(tensor_type),
        dimension_count(shape.dimension_count),
        dimensions(shape.dimensions),
        element_count(shape.element_count),
        data_offset(data.offset),
        data_length(data.length) {}

  std::string name;
  TensorType type;
  std::uint32_t dimension_count;
  std::array<std::uint64_t, kMaxDimensions> dimensions;

  std::uint64_t element_count;
  // Byte range within the file. Relative to the tensor data region while the
  // index is being read; absolute — and validated as inside the file — once
  // read_index returns Ok.
  std::uint64_t data_offset;
  std::uint64_t data_length;

  // Quantized formats store blocks of several values; the rest store each
  // value on its own. A type the reader accepted always has a layout.
  [[nodiscard]] bool is_quantized() const noexcept {
    return format_layout(type)->block_elements > 1;
  }
};

// The invariant, asserted rather than described. If someone restores a default
// this fails at compile time, in this file, next to the reason.
static_assert(!std::is_default_constructible_v<TensorEntry>,
              "a TensorEntry without a type is the defect this type prevents");

}  // namespace frizzle::gguf
