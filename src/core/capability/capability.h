#pragma once

#include <string_view>

#include "core/gguf/types.h"

namespace frizzle::arch {
struct Architecture;
}  // namespace frizzle::arch
namespace frizzle::formats {
class Format;
}  // namespace frizzle::formats
namespace frizzle::tokenizer {
struct Algorithm;
}  // namespace frizzle::tokenizer

namespace frizzle::capability {

// Axis I: the capability table. Changes when Frizzle gains or loses an
// implementation, and for no other reason.
//
// Contract
//   - Three lookups, from an identifier a GGUF file carries to the
//     implementation that runs it: general.architecture to an Architecture,
//     a tensor's type to a Format, tokenizer.ggml.model to a tokenizer
//     Algorithm.
//   - Supported means present in the table. A lookup returns the
//     implementation, or null when Frizzle has none. Null is not an error
//     here: the caller decides what an unsupported identifier means, and
//     names it.
//   - A returned pointer is non-owning, to an immutable object of static
//     storage duration, valid for the life of the program.
//   - No allocation, no I/O, no exceptions. Read-only, so safe from any
//     thread.
//   - What the table holds: the architecture "gemma3"; the formats F32, F16,
//     Q4_0, Q4_1 and Q8_0; the tokenizer model "llama", SentencePiece BPE,
//     which is what Gemma 3's file names. No weight in the pinned file is
//     F16; the F16 row is for the KV cache, whose format whoever builds the
//     graph looks up by the plan's cache type, F16 by default
//     (runtime/engine.h).
//
// Algorithm
//   1. A lookup scans its table in order and returns the implementation of
//      the first row whose identifier equals the argument.
//   2. If no row matches, it returns null.
//
// Design and cost
//   - The tables hold a row only for an architecture, format or tokenizer
//     the pinned model uses.
//   - The tables are constexpr arrays of an identifier and a pointer to a
//     constant: constant-initialized read-only data, with no work at startup
//     and no dependence on another translation unit's initialization order.
//   - A lookup is a linear scan of at most five rows: at most five
//     comparisons of a short string or an enum, no allocation.
//   - Every format row is checked at compile time to name its Format's own
//     type, so the table and the formats cannot disagree.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.60   Prefer T* over T& when "no argument" is a valid option: each
//            lookup returns a pointer, null for "Frizzle has none".
//     F.6    If your function must not throw, declare it noexcept: the
//            lookups only compare and return.
//     P.5    Prefer compile-time checking to run-time checking: the format
//            rows' types are checked by a static_assert.
//     Con.5  Use constexpr for values that can be computed at compile time:
//            the tables.
//     ES.71  Prefer a range-for: the scan reads each row once, by const
//            reference.
//   C++ performance guidelines
//     EMB.6   Push computation to compile time: the tables are constant data,
//             with no startup work.
//     CDSA.27 Scan flat before building an index: five rows at most.
//     CDSA.11 Pick a trie or a hash by query shape: these are point lookups,
//             which favour a hash at scale; not applied at five rows, per
//             CDSA.27.
//     COPY.7  Eliminate hidden copies: rows are read by const auto&, and the
//             identifiers (a string_view, an enum) are passed by value.
//     LIFE.6  The static-initialization-order fiasco: not at risk, since the
//             rows point at constants and are constant-initialized.

[[nodiscard]] const arch::Architecture* find_architecture(
    std::string_view general_architecture) noexcept;

[[nodiscard]] const formats::Format* find_format(
    gguf::TensorType type) noexcept;

[[nodiscard]] const tokenizer::Algorithm* find_tokenizer(
    std::string_view tokenizer_model) noexcept;

}  // namespace frizzle::capability
