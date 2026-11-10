// Axis C: changes with the GGUF file format, or what the index records of it.
//
// Contract, algorithm, cost and corpus guidance: index.h. This file
// defines what that header states.

#include "core/gguf/index.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>

namespace frizzle::gguf {
namespace {

// Reads the value of `entry`, stored as `Stored`, into `out` as `Out`,
// provided the file declared it as `declared`. Every typed accessor but
// read_string is this, with its own pair.
template <typename Stored, typename Out>
MetadataError read_as(const MetadataEntry* entry, ValueType declared,
                      Out& out) noexcept {
  if (entry == nullptr) return MetadataError::MissingKey;
  if (entry->type != declared) return MetadataError::WrongType;
  // The reader stores each declared type in its one alternative, so another
  // is a defect here, not in the file. It stops here, as std::get's throw
  // would have inside this noexcept function (E.12).
  const Stored* stored = std::get_if<Stored>(&entry->value);
  if (stored == nullptr) std::abort();
  out = static_cast<Out>(*stored);
  return MetadataError::Ok;
}

}  // namespace

std::optional<TensorId> TensorIndex::find(
    std::string_view name) const noexcept {
  const auto at =
      std::lower_bound(by_name_.begin(), by_name_.end(), name,
                       [&](TensorId id, std::string_view n) {
                         return std::string_view(tensor(id).name) < n;
                       });
  if (at == by_name_.end() || tensor(*at).name != name) return std::nullopt;
  return *at;
}

const MetadataEntry* TensorIndex::entry(std::string_view key) const noexcept {
  for (const MetadataEntry& e : metadata_) {
    if (e.key == key) return &e;
  }
  return nullptr;
}

MetadataError TensorIndex::read_u32(std::string_view key,
                                    std::uint32_t& out) const noexcept {
  return read_as<std::uint64_t>(entry(key), ValueType::UInt32, out);
}

MetadataError TensorIndex::read_f32(std::string_view key,
                                    float& out) const noexcept {
  return read_as<double>(entry(key), ValueType::Float32, out);
}

MetadataError TensorIndex::read_bool(std::string_view key,
                                     bool& out) const noexcept {
  return read_as<bool>(entry(key), ValueType::Bool, out);
}

MetadataError TensorIndex::read_string(std::string_view key,
                                       std::string_view& out) const noexcept {
  const MetadataEntry* e = entry(key);
  if (e == nullptr) return MetadataError::MissingKey;
  if (e->type != ValueType::String) return MetadataError::WrongType;
  out = std::get<std::string>(e->value);
  return MetadataError::Ok;
}

MetadataError TensorIndex::read_array(std::string_view key,
                                      ArrayLocation& out) const noexcept {
  return read_as<ArrayLocation>(entry(key), ValueType::Array, out);
}

}  // namespace frizzle::gguf
