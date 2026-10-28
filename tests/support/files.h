#pragma once

// Axis V: reading test inputs from disk. Test support only.
//
// Contract
//   - read_file(path): every byte of the file. A file that cannot be opened
//     or read fails the test, by its path, never yields an empty buffer: an
//     empty buffer would let a reader's test pass for the wrong reason.
//   - read_front(path): the front of a large file, as many bytes as reading
//     its GGUF index needs, and the whole file's size, so a byte source over
//     the front checks offsets against the real size. The model file is
//     722 MB; nothing here holds it whole.
//   - The returned bytes are the caller's; a MemoryByteSource over them must
//     not outlive them.
//
// Algorithm, read_front
//   1. Read the first 16 MiB (or the whole file, if shorter).
//   2. Read the index over them. While it asks for more bytes, read that
//      many and try again; any other failure fails the test, named.
//
// Corpus guidance
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: a missing or short file fails, by name.
//     SL.con.3 Avoid bounds errors: bytes are returned with their size, and
//            a source is always given the file's true size.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/reader.h"

namespace frizzle::testing {

// The first `count` bytes of the file at `path`.
inline std::vector<std::byte> read_bytes(const std::string& path,
                                         std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.is_open(), "cannot open " << path);
  std::vector<char> raw(count);
  in.read(raw.data(), static_cast<std::streamsize>(count));
  REQUIRE_MESSAGE(static_cast<std::size_t>(in.gcount()) == count,
                  "cannot read " << count << " bytes of " << path);
  std::vector<std::byte> bytes(count);
  for (std::size_t i = 0; i < count; ++i) {
    bytes[i] = static_cast<std::byte>(static_cast<unsigned char>(raw[i]));
  }
  return bytes;
}

inline std::uint64_t file_size(const std::string& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  REQUIRE_MESSAGE(!error, path << ": " << error.message());
  return size;
}

inline std::vector<std::byte> read_file(const std::string& path) {
  return read_bytes(path, static_cast<std::size_t>(file_size(path)));
}

struct Front {
  std::vector<std::byte> bytes;  // the file's first bytes
  std::uint64_t file_size = 0;   // the whole file's
  gguf::TensorIndex index;
};

inline Front read_front(const std::string& path) {
  constexpr std::uint64_t kFirst = std::uint64_t{16} << 20U;
  Front front;
  front.file_size = file_size(path);
  std::uint64_t want = front.file_size < kFirst ? front.file_size : kFirst;
  for (;;) {
    front.bytes = read_bytes(path, static_cast<std::size_t>(want));
    gguf::MemoryByteSource source{front.bytes, front.file_size};
    const gguf::ReadResult read = gguf::read_index(source, front.index);
    if (read.error == gguf::ReadError::Ok) return front;
    REQUIRE_MESSAGE(read.error == gguf::ReadError::NeedMoreBytes,
                    path << ": " << gguf::to_string(read.error));
    REQUIRE(read.bytes_needed > want);
    want = read.bytes_needed;
  }
}

}  // namespace frizzle::testing
