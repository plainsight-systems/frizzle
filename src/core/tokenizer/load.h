#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "core/gguf/arrays.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::tokenizer {

// Axis K: changes with Gemma 3's tokenizer, SentencePiece BPE.
//
// Reads a tokenizer array by its key, for the loaders in this module: the
// vocabulary's tokenizer.ggml.tokens and tokenizer.ggml.token_type
// (vocabulary.h) and SentencePiece BPE's tokenizer.ggml.scores
// (bpe/sentencepiece_bpe.h). Every failure names the key.
//
// Contract
//   - read_string_array, read_int32_array, read_float32_array(source, index,
//     key, out): Ok with `out` holding every element of the array under
//     `key`; or a refusal, `out` untouched (gguf/arrays.h):
//       MissingKey: no such key; the subject is the key.
//       WrongKeyType: the key's value is not an array; the subject is the
//         key.
//       Unreadable: the array cannot be read whole as the type asked for
//         (its elements of another type, its bytes short, or bytes not yet
//         resident); the subject is the key, ": ", and GGUF's reason.
//   - No refusal is thrown; allocation failure, filling `out` or building
//     a subject, is not caught.
//
// Algorithm
//   1. Locate the array in the index (gguf/index.h); a missing key or a
//      value that is not an array is refused as above.
//   2. Read it whole in one pass (gguf/arrays.h); a read error is refused as
//      Unreadable.
//
// Design and cost
//   - Counted for Gemma 3 1B, once a load: the tokens, 262,144 strings and
//     2,315,366 bytes of text; the types and the scores, 262,144 four-byte
//     elements each, 1 MiB each. Each is read once, element by element,
//     into `out`; a refusal builds only its subject string.
//   - The three readers differ only in the element type, so one locate and
//     one check of the read's result serve all three.
//
// Verification
//   The tokenizer suite loads the pinned tokenizer through all three. No
//   test reaches a refusal.
//
// Corpus guidance
//   C++ Core Guidelines
//     E.27   If you can't throw exceptions, use error codes systematically:
//            every refusal is a LoadResult naming the key.
//     ES.79  Use default to handle common cases (only): the switch over
//            MetadataError names every enumerator and has no default, so a
//            new one is flagged; a value outside them is refused as
//            WrongKeyType after the switch.
//     SL.io.2 When reading, always consider ill-formed input: the key's
//            type and the array's element type and length are checked
//            before any element is used.

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
