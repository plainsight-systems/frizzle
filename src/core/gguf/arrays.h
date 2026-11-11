#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/types.h"

namespace frizzle::gguf {

// Axis C: changes with the GGUF format.
//
// Whole-array reads, for a consumer that needs every element of an array the
// index located: the tokenizer's vocabulary, token types and scores. Each is
// one pass over the array, so its cost is linear in the array.
// Optimization (practice): Gemma 3's 262,144 token strings read element by
// element, each skipping from the array's start, would be about 34 billion
// skips, counted; one pass reads each once.
//
//   - Elements of another type than the one asked for are WrongElementType.
//   - The array's bytes must hold exactly its elements; anything else is
//     ShortRead.
//   - Bytes not resident give NeedMoreBytes, with how many to supply.
//   - Never throws, and leaves `out` untouched unless the read succeeds.

// A string array's elements, in one buffer. Only read_strings builds one, so
// every element it hands out lies within the buffer. The buffer is a vector,
// not a std::string, so moving a table never moves its bytes: views into it
// stay valid for as long as the table lives.
class StringTable {
 public:
  [[nodiscard]] std::size_t size() const noexcept { return ends_.size(); }

  // Precondition: element < size().
  [[nodiscard]] std::string_view operator[](
      std::size_t element) const noexcept {
    const std::size_t begin = element == 0 ? 0 : ends_[element - 1];
    const std::span<const char> text =
        std::span(bytes_).subspan(begin, ends_[element] - begin);
    return {text.data(), text.size()};
  }

 private:
  friend ReadResult read_strings(ByteSource& source, const ArrayLocation& array,
                                 StringTable& out);

  std::vector<char> bytes_;
  std::vector<std::size_t>
      ends_;  // element i is bytes_[ends_[i - 1], ends_[i])
};

[[nodiscard]] ReadResult read_strings(ByteSource& source,
                                      const ArrayLocation& array,
                                      StringTable& out);
[[nodiscard]] ReadResult read_int32s(ByteSource& source,
                                     const ArrayLocation& array,
                                     std::vector<std::int32_t>& out);
[[nodiscard]] ReadResult read_float32s(ByteSource& source,
                                       const ArrayLocation& array,
                                       std::vector<float>& out);

}  // namespace frizzle::gguf
