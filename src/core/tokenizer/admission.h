#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "core/tokenizer/tokenizer.h"

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Whether an encode goes on to merge (tokenizer.h, encoding is bounded):
// pure functions of the text's length and of counts, which SentencePiece BPE
// calls (bpe/sentencepiece_bpe.h), so the bound is decided in one place.
//
// Contract
//   - admit_raw(raw, max_tokens, longest_special): TooLong when the raw text
//     is longer than 7/2 × kMaxEncodeBytes plus max_tokens × longest_special
//     bytes, before it is segmented or normalized; otherwise Ok. Either way
//     the result names the raw bytes and no other count.
//   - Sound: text an encode could admit has at most max_tokens special
//     tokens, each at most longest_special bytes as the text spells them,
//     and ordinary bytes that normalize to at most kMaxEncodeBytes.
//     Normalizing never shrinks this tokenizer's text, as escaping a space
//     as ▁ makes 1 byte 3, so those ordinary bytes are at most
//     kMaxEncodeBytes before it, within the allowance; admit_raw never
//     refuses text that would be admitted.
//   - The 7/2 is looser than this tokenizer needs: a ratio of 1 would bound
//     it too. A looser early refusal is still sound: text it lets through
//     that is too long is refused by admit, after normalizing.
//   - admit(specials, segment_bytes, cover, max_tokens): the least tokens
//     the text makes are its specials, one each, and each ordinary segment's
//     ceil(n / cover) for its n normalized bytes, since no token spans a
//     special or two segments. Over max_tokens, TooManyTokens with that
//     count; then normalized bytes past kMaxEncodeBytes, TooLong; otherwise
//     Ok. The result names both counts and no raw bytes, which the caller
//     adds. The count is a lower bound, so it never refuses text that would
//     fit. Precondition: cover >= 1.
//   - Sums are in 64 bits and cannot wrap: a segment is at most the text,
//     at most 4 GiB of the module's memory, and there are at most as many
//     segments as its bytes. admit_raw's doubled allowance, 7 × 2^19 +
//     2 × max_tokens × longest_special, fits while a special token's
//     spelling is under 2^31 bytes; Gemma 3's longest is 93.
//
// Algorithm
//   admit_raw, in integers, both sides doubled:
//     1. allowed = 7 × kMaxEncodeBytes + 2 × max_tokens × longest_special.
//     2. TooLong when 2 × raw.size() > allowed; otherwise Ok.
//   admit:
//     1. least = specials; for each segment of n bytes, least grows by
//        n / cover, plus 1 when n % cover is not 0, and bytes by n.
//     2. TooManyTokens when least > max_tokens; else TooLong when bytes >
//        kMaxEncodeBytes; else Ok.
//
// For Gemma 3 1B's pinned file: the longest special token is spelled in 93
// bytes once spaces are ▁ (a run of 31 spaces); the longest token covers 48
// normalized bytes. At a context of 32,768 tokens, admit_raw refuses raw
// text past 7/2 × 512 KiB + 32,768 × 93 bytes, 4,882,432 bytes, 4.66 MiB.
//
// Design and cost
//   - Counted: admit_raw is three multiplications and a comparison, once a
//     turn; admit is a division and a remainder a segment, once a turn, at
//     most 2 × max_tokens + 1 segments. Nothing is allocated.
//   - admit_raw takes the raw text, not its length, so the text cannot be
//     passed where the token limit is meant, or the limit where the text's
//     length is (I.24).
//
// Verification
//   No test reaches admit_raw or admit with text near either limit
//   (tokenizer.h); the tokenizer suite runs both on every reference text,
//   each admitted.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.8    Prefer pure functions: the text's length and counts in, a
//            verdict out.
//     E.27   If you can't throw exceptions, use error codes systematically:
//            the verdict is an EncodeResult.
//     I.24   Avoid adjacent parameters that can be invoked by the same
//            arguments in either order with different meaning: admit_raw's
//            raw text is a string_view, not a count beside the limit.
//   C++ performance guidelines
//     EMB.4  Size the worst case of dynamic allocation: these refusals keep
//            an encode within the worst case tokenizer.h counts.

[[nodiscard]] EncodeResult admit_raw(std::string_view raw,
                                     std::uint32_t max_tokens,
                                     std::size_t longest_special) noexcept;

// Precondition: cover >= 1.
[[nodiscard]] EncodeResult admit(std::uint64_t specials,
                                 std::span<const std::uint64_t> segment_bytes,
                                 std::size_t cover,
                                 std::uint32_t max_tokens) noexcept;

}  // namespace frizzle::tokenizer
