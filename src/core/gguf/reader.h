#pragma once

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/types.h"

namespace frizzle::gguf {

// Axis C: changes with the GGUF file format, or with this parse contract.
//
// Reads a GGUF file's header, metadata and tensor infos from an untrusted
// source into the tensor index (index.h). The file arrives from a download, so
// every count, length, type and offset in it is a claim to be checked before it
// is used. The engine reads the index once a load, from the front of the file
// (runtime/engine.cpp); the browser supplies that front from the kept file
// (app/src/worker/engine/engine.ts).
//
// Contract
//   - read_index(source, out): Ok with `out` holding every tensor and every
//     metadata entry of the file; or a ReadError naming the first rule the
//     file breaks, with `out` untouched. On Ok `out` is replaced whole: a
//     previously filled index keeps nothing of its old file.
//   - Never reads outside the file. Every offset and length is checked
//     against source.size(), the whole file's length, not against what is
//     resident, with checked arithmetic (checked.h): a sum or product that
//     would wrap is OffsetOverflow or ElementCountOverflow, never a smaller
//     number that passes a bounds check.
//   - Nothing in a file makes it throw: every refusal is a named ReadError,
//     returned with bytes_needed in a ReadResult (E.27). It is not noexcept:
//     it allocates (the entries, their names and keys, the sort keys), each
//     allocation bounded by the kMax* limits of types.h, and an allocation
//     failure is not a ReadError and is not caught here.
//   - Reads the index only, never tensor data: every tensor's data range is
//     checked against the file's size, none of its bytes is read. A source
//     holding only the front of the file is enough. When a read reaches
//     bytes inside the file that the source does not hold, the result is
//     NeedMoreBytes with bytes_needed, the length of the front, from byte 0,
//     that the read needed; a file too short for a field is ShortRead. The
//     two are never confused: a partly resident source is never called
//     malformed for being partly resident.
//   - Records every tensor whatever its format, provided GGUF defines the
//     format (types.h, kFormatLayouts): a format no kernel here runs is
//     recorded, not refused. Whether a format runs is the load's question,
//     answered after the index: routes refuses a planned tensor whose format
//     the capability table does not list as UnsupportedFormat, naming it
//     (residency/routes.h).
//     A format number GGUF does not define is UnknownTensorType.
//   - On Ok, every tensor's data_offset is absolute, a multiple of the
//     alignment, and [data_offset, data_offset + data_length) lies inside
//     the file and overlaps no other tensor's range.
//
// Algorithm (read_index; each step refuses with the error named and stops)
//   1. Header: magic "GGUF" (BadMagic); version, uint32, must be 3
//      (UnsupportedVersion); tensor count and metadata count, uint64 each,
//      each at most 2^20 (CountTooLarge), checked before anything is
//      reserved. All values are little-endian.
//   2. Metadata, `metadata count` entries, each: key, a uint64 length then
//      its bytes, the length at most 2^20 (StringTooLong) and the key not
//      empty (EmptyKey); value type, uint32, one of GGUF's 13
//      (UnknownValueType); the value. A scalar is read and widened: unsigned
//      integers to 64 bits, signed ones sign-extended, FLOAT32 and FLOAT64
//      to double, BOOL true when its byte is not zero. A string is read,
//      length at most 2^20. An array is located, not read: element type,
//      uint32, known (UnknownValueType) and not an array (NestedArray);
//      element count, uint64, at most 2^24 (ArrayTooLong); then for strings
//      each element's length is read (at most 2^20) and its text skipped,
//      and for numbers the count times the element size, checked
//      (OffsetOverflow), is skipped. Skipping checks the range against the
//      file's size and reads none of its bytes. Then no key may appear twice
//      (DuplicateMetadataKey).
//   3. Alignment: general.alignment if present, which must be a UINT32 and a
//      power of two (BadAlignment); 32 if absent.
//   4. Tensor infos, `tensor count` of them, each: name, a string of at most
//      64 bytes (StringTooLong past 2^20, TensorNameTooLong past 64);
//      dimension count, uint32, at most 4
//      (TooManyDimensions); each extent, stored as int64, not negative
//      (NegativeDimension), their product checked (ElementCountOverflow);
//      format, uint32, one GGUF defines (UnknownTensorType); the first
//      extent, a row, a whole number of the format's blocks
//      (RowNotWholeBlocks); data offset, uint64, relative to the data
//      region; data length = elements / block elements × block bytes,
//      checked (OffsetOverflow). Then no name may appear twice
//      (DuplicateTensorName).
//   5. Data placement: the data region starts at the end of the tensor infos
//      rounded up to the alignment, checked (OffsetOverflow), inside the
//      file (TensorDataOutOfBounds).
//      Each tensor's offset must be a multiple of the alignment
//      (MisalignedTensorData); it is made absolute, checked
//      (OffsetOverflow), and its range must lie inside the file
//      (TensorDataOutOfBounds). Then no two ranges may overlap
//      (OverlappingTensorData).
//   6. On Ok only: the tensors and metadata move into `out`, and its
//      by-name order for find is built (index.h).
//
// Design and cost
//   - read_index takes an out parameter, untouched unless the read
//     succeeds, so a failed read leaves the caller's index as it was.
//   - Optimization: the duplicate-key, duplicate-name and overlap checks
//     sort, then compare neighbours: O(n log n) for n entries, where
//     comparing every pair would let a file declaring 2^20 tensors stall
//     the reader for about 2^39 comparisons. Each builds one vector of n
//     views or ranges for the sort.
//   - Arrays are skipped, not read. A string array costs one 8-byte length
//     read an element, each through ByteSource::read, a virtual call; Gemma 3
//     1B's vocabulary is 262,144 of them, counted from its element count.
//     An array of numbers costs one checked multiplication and one range
//     check, whatever its length: Gemma 3 1B's scores and token types,
//     262,144 elements each, read none of their bytes.
//   - What a partly resident source needs: every byte from the start of the
//     file to the end of the tensor infos, and no tensor data. A string
//     array needs every length field resident, so all of it but its last
//     text. An array of numbers is skipped by range check, so it never
//     causes NeedMoreBytes itself; the next read after it that needs bytes
//     does, and asks for bytes past the whole array. Each NeedMoreBytes asks
//     only through the end of the field it could not read, so a caller that
//     supplies exactly bytes_needed advances one field a round.
//   - The browser begins with the kept file's first 16 MiB, of the pinned
//     file's 721,918,496 bytes, and reads again at bytes_needed when that is
//     short (app/src/worker/engine/engine.ts). UNVERIFIED: whether Gemma 3
//     1B's index ends inside the first 16 MiB.
//   - A file whose header claims 2^20 metadata entries has storage for that
//     many reserved before the first entry is read; tensor storage is
//     reserved likewise once the metadata is read. The counts' bound limits
//     this; it is not otherwise small.
//   - Gemma 3 1B's pinned file, counted: 38 metadata keys and 340 tensors
//     sorted for the duplicate checks, 340 ranges for the overlap check.
//   - Tests: tests/contracts/gguf/model_file_test.cpp, over the files
//     scripts/make_gguf_fixtures.py writes to tests/fixtures/gguf/. Locked
//     by name, one fixture each unless noted: BadMagic,
//     UnsupportedVersion, ShortRead (a truncated header; an empty buffer),
//     CountTooLarge (the tensor count), EmptyKey (an empty key; a metadata
//     count larger than the entries present), NegativeDimension,
//     UnknownTensorType, RowNotWholeBlocks, DuplicateTensorName (adjacent;
//     apart), NestedArray, UnknownValueType, TensorDataOutOfBounds (an
//     offset past the end; data cut short), OffsetOverflow (an offset that
//     wraps), BadAlignment (wrong type; not a power of two),
//     MisalignedTensorData, OverlappingTensorData, DuplicateMetadataKey,
//     TensorNameTooLong.
//   - Locked for a partly resident source, on the valid fixture (two
//     tensors, one array of three strings): every cut below its last tensor
//     byte, presented as the whole file, is refused; every front, presented
//     with the whole file's size, reads or returns NeedMoreBytes with
//     bytes_needed past the front and within the file, the index's tensors
//     still empty; supplying bytes_needed until Ok converges with no tensor
//     data resident.
//   - Not locked by any test: StringTooLong, ArrayTooLong,
//     TooManyDimensions, ElementCountOverflow, CountTooLarge for the
//     metadata count; an array of numbers skipped without its bytes
//     resident (no fixture holds one); `out` untouched when it already held
//     an index, or its metadata untouched (only an empty index's tensors are
//     checked); a tensor of a format the capability table does not list
//     recorded (the valid fixture's are Q4_0 and F32). Every load runs
//     read_index on the pinned file, the oracle's and replay's among them.
//
// Corpus guidance
//   C++ Core Guidelines
//     SL.io.2  When reading, always consider ill-formed input: every field is
//              checked as it is read, and the first rule broken is the
//              answer.
//     SL.con.3 Avoid bounds errors: no read reaches past source.size(); a
//              length is checked before the string is sized for it; fixed
//              buffers are std::array, read and decoded through spans.
//     C.12     No reference data members: the reader holds its source by
//              pointer.
//     ES.103   Don't overflow: sums and products of file values are
//              checked; ranges are checked by subtraction.
//     E.27     Error codes systematically: ReadResult carries the ReadError
//              and, for NeedMoreBytes, bytes_needed.
//     E.12     noexcept where a throw is impossible or unacceptable: not
//              declared; nothing in a file throws, but allocation failure
//              propagates to the caller.
//     F.20     Prefer return values to out parameters: not followed, so that
//              the caller's index changes only on success.
//   C++ performance guidelines
//     WASM.9   Validate every offset and length against the authoritative
//              total size with checked arithmetic, and never assume the whole
//              asset is addressable: the index is read from the front alone,
//              every range checked against the whole file's size.

// Reads the file's index into `out` (Contract above). `out` is untouched
// unless the result is Ok.
[[nodiscard]] ReadResult read_index(ByteSource& source, TensorIndex& out);

}  // namespace frizzle::gguf
