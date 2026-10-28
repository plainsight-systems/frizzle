#include "core/tokenizer/load.h"

#include <string>

namespace frizzle::tokenizer {
namespace {

LoadResult locate(const gguf::TensorIndex& index, std::string_view key,
                  gguf::ArrayLocation& out) {
  switch (index.read_array(key, out)) {
    case gguf::MetadataError::Ok:
      return {};
    case gguf::MetadataError::MissingKey:
      return {LoadError::MissingKey, std::string(key)};
    case gguf::MetadataError::WrongType:
      return {LoadError::WrongKeyType, std::string(key)};
  }
  return {LoadError::WrongKeyType, std::string(key)};
}

LoadResult checked(std::string_view key, const gguf::ReadResult& read) {
  if (read.error == gguf::ReadError::Ok) return {};
  return {LoadError::Unreadable,
          std::string(key) + ": " + std::string(gguf::to_string(read.error))};
}

}  // namespace

LoadResult read_string_array(gguf::ByteSource& source,
                             const gguf::TensorIndex& index,
                             std::string_view key, gguf::StringTable& out) {
  gguf::ArrayLocation array{};
  if (auto r = locate(index, key, array); !r.ok()) return r;
  return checked(key, gguf::read_strings(source, array, out));
}

LoadResult read_int32_array(gguf::ByteSource& source,
                            const gguf::TensorIndex& index,
                            std::string_view key,
                            std::vector<std::int32_t>& out) {
  gguf::ArrayLocation array{};
  if (auto r = locate(index, key, array); !r.ok()) return r;
  return checked(key, gguf::read_int32s(source, array, out));
}

LoadResult read_float32_array(gguf::ByteSource& source,
                              const gguf::TensorIndex& index,
                              std::string_view key, std::vector<float>& out) {
  gguf::ArrayLocation array{};
  if (auto r = locate(index, key, array); !r.ok()) return r;
  return checked(key, gguf::read_float32s(source, array, out));
}

}  // namespace frizzle::tokenizer
