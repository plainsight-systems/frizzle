#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "core/gguf/arrays.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer {

// Reads a tokenizer array by its key, for the loaders in this module. Every
// failure names the key: missing, not an array, or unreadable, and why.

[[nodiscard]] LoadResult read_string_array(gguf::ByteSource& source,
                                           const gguf::TensorIndex& index,
                                           std::string_view key,
                                           gguf::StringTable& out);

[[nodiscard]] LoadResult read_int32_array(gguf::ByteSource& source,
                                          const gguf::TensorIndex& index,
                                          std::string_view key,
                                          std::vector<std::int32_t>& out);

[[nodiscard]] LoadResult read_float32_array(gguf::ByteSource& source,
                                            const gguf::TensorIndex& index,
                                            std::string_view key,
                                            std::vector<float>& out);

}  // namespace frizzle::tokenizer
