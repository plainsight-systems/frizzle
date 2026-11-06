// Axis V: the model-file suite (tests/README.md).
//
// Contract locked: a malformed or hostile GGUF is refused by name, never
// read past its bounds or misinterpreted. The file arrives from a CDN, and
// in the browser only its front is held while the index is read.
//
// Inputs: tests/fixtures/gguf/, written by scripts/make_gguf_fixtures.py,
// each malformed file one deliberate change from a valid one, for the cases
// this contract names.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.con.3 Avoid bounds errors: every cut of a valid file is presented as
//            the whole file, so a read past the end is a wrong answer here.
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: each refusal is checked by its name.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/reader.h"
#include "core/gguf/types.h"
#include "support/files.h"

namespace {

using frizzle::gguf::MemoryByteSource;
using frizzle::gguf::ReadError;
using frizzle::gguf::ReadResult;
using frizzle::gguf::TensorIndex;

std::vector<std::byte> fixture(const std::string& name) {
  return frizzle::testing::read_file(std::string(FRIZZLE_FIXTURE_DIR) +
                                     "/gguf/" + name + ".gguf");
}

ReadError read_whole(std::span<const std::byte> bytes) {
  MemoryByteSource source{bytes};
  TensorIndex index;
  return read_index(source, index).error;
}

ReadError read_fixture(const std::string& name) {
  return read_whole(fixture(name));
}

}  // namespace

TEST_CASE("each malformed file is refused with its own named error") {
  CHECK(read_fixture("bad_magic") == ReadError::BadMagic);
  CHECK(read_fixture("bad_version") == ReadError::UnsupportedVersion);
  CHECK(read_fixture("truncated_header") == ReadError::ShortRead);
  CHECK(read_fixture("tensor_count_too_large") == ReadError::CountTooLarge);
  // Claims five keys; the zero padding after the header reads as an empty
  // key and is refused there, before the file runs out.
  CHECK(read_fixture("metadata_count_lies") == ReadError::EmptyKey);
  CHECK(read_fixture("negative_dimension") == ReadError::NegativeDimension);
  CHECK(read_fixture("unknown_tensor_type") == ReadError::UnknownTensorType);
  CHECK(read_fixture("not_block_aligned") == ReadError::RowNotWholeBlocks);
  CHECK(read_fixture("duplicate_tensor_name") ==
        ReadError::DuplicateTensorName);
  CHECK(read_fixture("duplicate_tensor_name_apart") ==
        ReadError::DuplicateTensorName);
  CHECK(read_fixture("nested_array") == ReadError::NestedArray);
  CHECK(read_fixture("unknown_value_type") == ReadError::UnknownValueType);
  CHECK(read_fixture("offset_past_eof") == ReadError::TensorDataOutOfBounds);
  CHECK(read_fixture("data_truncated") == ReadError::TensorDataOutOfBounds);
  // Offset arithmetic that would wrap is refused, not computed.
  CHECK(read_fixture("offset_overflow") == ReadError::OffsetOverflow);
}

TEST_CASE("the format's rules are enforced, not defaulted around") {
  CHECK(read_fixture("alignment_wrong_type") == ReadError::BadAlignment);
  CHECK(read_fixture("alignment_not_power_of_two") == ReadError::BadAlignment);
  CHECK(read_fixture("misaligned_tensor_offset") ==
        ReadError::MisalignedTensorData);
  CHECK(read_fixture("overlapping_tensor_data") ==
        ReadError::OverlappingTensorData);
  CHECK(read_fixture("empty_metadata_key") == ReadError::EmptyKey);
  CHECK(read_fixture("duplicate_metadata_key") ==
        ReadError::DuplicateMetadataKey);
  CHECK(read_fixture("tensor_name_65_bytes") == ReadError::TensorNameTooLong);
}

TEST_CASE("an empty file is refused without reading anything") {
  const std::vector<std::byte> empty;
  CHECK(read_whole(empty) == ReadError::ShortRead);
}

TEST_CASE("every truncation of a valid file is refused, none read past it") {
  // Each cut is presented as the whole file. Cuts at or past the last
  // declared tensor byte remove only padding no region claims, which the
  // reader rightly accepts.
  const std::vector<std::byte> full = fixture("valid");
  std::uint64_t last_declared_byte = 0;
  {
    MemoryByteSource whole{full};
    TensorIndex index;
    REQUIRE(read_index(whole, index).error == ReadError::Ok);
    for (const auto& t : index.tensors()) {
      last_declared_byte =
          std::max(last_declared_byte, t.data_offset + t.data_length);
    }
    REQUIRE(last_declared_byte < full.size());
  }
  std::size_t wrongly_accepted = 0;
  for (std::size_t cut = 0; cut < last_declared_byte; ++cut) {
    if (read_whole(std::span(full).first(cut)) == ReadError::Ok) {
      ++wrongly_accepted;
    }
  }
  CHECK(wrongly_accepted == 0);
}

TEST_CASE("a front of the file asks for the bytes it lacks, and nothing else") {
  // The browser holds the front of the file and knows its whole size. A
  // front either reads, or asks for more; it is never called malformed.
  const std::vector<std::byte> full = fixture("valid");
  for (std::size_t cut = 0; cut <= full.size(); ++cut) {
    CAPTURE(cut);
    MemoryByteSource source{std::span(full).first(cut), full.size()};
    TensorIndex index;
    const ReadResult result = read_index(source, index);
    if (result.error == ReadError::Ok) continue;
    REQUIRE(result.error == ReadError::NeedMoreBytes);
    CHECK(result.bytes_needed > cut);
    CHECK(result.bytes_needed <= full.size());
    CHECK(index.tensors().empty());  // untouched unless the read succeeds
  }
}

TEST_CASE("supplying the bytes asked for converges on the whole index") {
  const std::vector<std::byte> full = fixture("valid");
  std::uint64_t resident = 0;
  TensorIndex index;
  for (int attempt = 0;; ++attempt) {
    REQUIRE(attempt < 1000);
    MemoryByteSource source{
        std::span(full).first(static_cast<std::size_t>(resident)), full.size()};
    const ReadResult result = read_index(source, index);
    if (result.error == ReadError::Ok) break;
    REQUIRE(result.error == ReadError::NeedMoreBytes);
    resident = result.bytes_needed;
  }
  CHECK(index.tensors().size() == 2);
  // The index is whole with no tensor data resident.
  CHECK(resident <= index.tensors()[0].data_offset);
}
