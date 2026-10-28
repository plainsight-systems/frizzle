#pragma once

#include <cstdint>
#include <string>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/types.h"

namespace frizzle::gguf {

// Axis C: changes with the GGUF format, or with this parse contract.
//
// Reads a GGUF file's header, metadata and tensor index from an untrusted
// source into `out`.
//
//   - Never reads outside the file. Every offset and length is validated
//     against the source's authoritative size, with checked arithmetic.
//   - Never throws. Failures are named values (E.27).
//   - Reads the index only, never tensor data, so only the front of the file
//     needs to be resident. When a read reaches bytes that are not,
//     the result is NeedMoreBytes with how many bytes to supply.
//     Optimization (browser): the page judges a model from its first bytes,
//     16 MiB of a 700 MB file, before deciding to download the rest.
//   - Records every tensor, whatever its format. A format the harness cannot
//     run is not an error here; preflight decides, and names the tensor.
//   - Leaves `out` untouched unless the read succeeds.
[[nodiscard]] ReadResult read_index(ByteSource& source, TensorIndex& out);

// Reads element `element` of a string array the index located. Walks the
// array from its start, so it costs the elements before it; for looking up a
// few named tokens. A whole vocabulary is read_strings (arrays.h). ShortRead
// if the array holds fewer elements or is not an array of strings.
[[nodiscard]] ReadResult read_string_element(ByteSource& source,
                                             const ArrayLocation& array,
                                             std::uint64_t element,
                                             std::string& out);

}  // namespace frizzle::gguf
