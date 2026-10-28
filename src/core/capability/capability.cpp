// Axis I: the capability table. Changes when Frizzle gains or loses an
// implementation, and for no other reason.
//
// Contract
//   - Defines the three lookups declared in capability.h: an architecture by
//     general.architecture, a format by tensor type, a tokenizer algorithm by
//     tokenizer.ggml.model. Each returns a non-owning pointer to an immutable
//     static implementation, or null when this file has no row for the
//     identifier.
//   - An identifier is supported exactly when it has a row here. No
//     allocation, no I/O, no exceptions; read-only.
//
// Algorithm
//   1. A lookup scans its table in order and returns the implementation of
//      the first row whose identifier equals the argument.
//   2. If no row matches, it returns null.
//
// Design and cost
//   - Changed from Charlotte e003048: only the rows the pinned Gemma 3 file
//     needs remain (capability.h lists them), and the media-encoder and
//     pre-tokenizer lookups are gone.
//   - Adding an implementation adds its file and one row here, and nothing
//     else.
//   - The tables are constexpr arrays of an identifier and a pointer to a
//     constant: constant-initialized read-only data. A lookup is at most
//     five comparisons.
//   - A static_assert ties each format row to its Format's own type.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.60   Prefer T* over T& when "no argument" is a valid option: null
//            means Frizzle has no implementation.
//     F.6    If your function must not throw, declare it noexcept: lookup()
//            and the three lookups only compare and return.
//     P.5    Prefer compile-time checking: the static_assert on format rows.
//     Con.5  Use constexpr for compile-time values: the tables.
//     ES.71  Prefer a range-for: lookup() reads each row by const reference.
//   C++ performance guidelines
//     EMB.6   Push computation to compile time: constant tables, no startup
//             work.
//     CDSA.27 Scan flat before building an index: five rows at most.
//     CDSA.11 Pick a trie or a hash by query shape: point lookups favour a
//             hash at scale; not applied at five rows, per CDSA.27.
//     COPY.7  Eliminate hidden copies: rows read by const auto&; a
//             string_view or an enum passed by value.
//     LIFE.6  Static-initialization order: not at risk; the rows point at
//             constants and are constant-initialized.

#include "core/capability/capability.h"

#include <array>
#include <span>

#include "core/arch/gemma3/gemma3.h"
#include "core/formats/f16/f16.h"
#include "core/formats/f32/f32.h"
#include "core/formats/format.h"
#include "core/formats/q4_0/q4_0.h"
#include "core/formats/q4_1/q4_1.h"
#include "core/formats/q8_0/q8_0.h"
#include "core/tokenizer/bpe/sentencepiece_bpe.h"

namespace frizzle::capability {
namespace {

// One identifier a file can carry, and the implementation that runs it.
template <typename Identifier, typename Implementation>
struct Row {
  Identifier identifier;
  const Implementation* implementation;
};

template <typename Identifier, typename Implementation>
const Implementation* lookup(
    std::span<const Row<Identifier, Implementation>> table,
    Identifier identifier) noexcept {
  for (const auto& row : table) {
    if (row.identifier == identifier) {
      return row.implementation;
    }
  }
  return nullptr;
}

constexpr std::array kArchitectures{
    Row<std::string_view, arch::Architecture>{"gemma3", &arch::kGemma3},
};
constexpr std::array kFormats{
    Row<gguf::TensorType, formats::Format>{gguf::TensorType::F32,
                                           &formats::kF32},
    Row<gguf::TensorType, formats::Format>{gguf::TensorType::F16,
                                           &formats::kF16},
    Row<gguf::TensorType, formats::Format>{gguf::TensorType::Q4_0,
                                           &formats::kQ4_0},
    Row<gguf::TensorType, formats::Format>{gguf::TensorType::Q4_1,
                                           &formats::kQ4_1},
    Row<gguf::TensorType, formats::Format>{gguf::TensorType::Q8_0,
                                           &formats::kQ8_0},
};
// Every format row names its Format's own type (format.h).
static_assert([] {
  for (const auto& row : kFormats) {
    if (row.implementation->type() != row.identifier) {
      return false;
    }
  }
  return true;
}());
constexpr std::array kTokenizers{
    Row<std::string_view, tokenizer::Algorithm>{
        "llama", &tokenizer::bpe::kSentencePiece},
};

}  // namespace

const arch::Architecture* find_architecture(
    std::string_view general_architecture) noexcept {
  return lookup<std::string_view, arch::Architecture>(kArchitectures,
                                                      general_architecture);
}

const formats::Format* find_format(gguf::TensorType type) noexcept {
  return lookup<gguf::TensorType, formats::Format>(kFormats, type);
}

const tokenizer::Algorithm* find_tokenizer(
    std::string_view tokenizer_model) noexcept {
  return lookup<std::string_view, tokenizer::Algorithm>(kTokenizers,
                                                        tokenizer_model);
}

}  // namespace frizzle::capability
