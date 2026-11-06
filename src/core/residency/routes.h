#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "core/gguf/index.h"
#include "core/residency/plan.h"

namespace frizzle::residency {

// Axis D: changes with how the file's bytes reach the buffers the plan lays
// out.
//
// Where every byte of the file goes: one route for each piece of each
// weight, saying which whole blocks of the file it takes, how they are laid
// out on the device and where in which buffer they land. A deterministic
// function, with no I/O and no GPU, of the tensor index, the residency plan
// and the file's size. Upload carries the routes out (upload.h); the piece
// writer turns them into writes (piece_writer.h).
//
// Contract
//   - plan_routes(index, plan, file_size, find_format, routes, out): Ok with
//     `routes`, ordered by file offset, and `out`, the plan as upload carries
//     it out; or a RouteError naming the tensor at fault, with `routes` and
//     `out` untouched.
//   - A route takes a piece's run of whole rows, so whole blocks: its file
//     range is the tensor's data offset plus first_row × row bytes, row_count
//     × row bytes long; its destination is the piece's buffer and offset; its
//     length is the piece's bound length, those bytes rounded up to 4, which
//     the format's device layout fills exactly (formats/device_layout.h).
//   - A tensor's format comes from `find_format`; the engine passes
//     capability::find_format (runtime/engine.cpp), so the formats routed are
//     the ones the capability table lists. A type it does not list is
//     UnsupportedFormat, whether or not the tensor has rows. A tensor whose
//     rows are not a whole number of 32-weight groups and that is not a
//     scalar is RowNotSteppable (formats/format.h).
//   - Every offset and length is checked against the file's size and the
//     piece's buffer before any byte arrives, by subtraction so no sum can
//     wrap. A route that would read past the file or write past its buffer,
//     or whose bound length is not its bytes rounded up to 4, is OutOfRange,
//     never a write.
//
// Algorithm
//   1. For each planned tensor, in index order: look up its format; check it
//      steps by groups or is a scalar; with no pieces, stop (no rows, nothing
//      to write); row bytes = data length / Σ row_count; for each piece,
//      check its file range and buffer range, then append Route{tensor, file
//      offset, bytes / block bytes, layout, buffer, offset, length}.
//   2. Sort the routes by file offset; move them and the carried plan out.
//
// Design and cost
//   - Out parameters, untouched unless the call succeeds, as
//     plan_residency's.
//   - Cost, counted for Gemma 3 1B at WebGPU's default limits: 342 routes,
//     one a piece, built once a load with a copy of the plan.
//   - Nothing tests plan_routes alone. Every load runs it, the oracle's and
//     replay's on the pinned model among them.
//
// Corpus guidance
//   C++ Core Guidelines
//     E.27   Error codes systematically: RouteError, with the tensor at fault
//            as its subject.
//     ES.103 Don't overflow: ranges are checked by subtraction, products with
//            checked arithmetic.
//     F.20, F.21  Prefer returning values to out parameters: not followed, so
//            that the caller's objects change only on success.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            RouteError takes one byte, the smallest that holds it.
//   C++ performance guidelines
//     WASM.9 Validate every offset and length against the authoritative total
//            size with checked arithmetic, treating the file as untrusted:
//            every route against the file's size, before any byte arrives.

// Finds the format a tensor type runs as, or null if it runs as none.
using FindFormat = const formats::Format* (*)(gguf::TensorType type) noexcept;

struct Route {
  gguf::TensorId tensor;      // whose piece it is
  std::uint64_t file_offset;  // where its blocks start in the file
  std::uint64_t blocks;       // how many whole blocks it takes
  const formats::DeviceLayout* layout;
  BufferIndex buffer;
  std::uint64_t buffer_offset;  // where the piece starts in its buffer
  std::uint64_t length;  // the piece's bound length: its bytes, rounded up to 4
};

// One byte, all its values need: returned to upload, never stored or sent as
// a number.
enum class RouteError : std::uint8_t {
  Ok,
  // A tensor whose format the lookup does not list. The subject names it.
  UnsupportedFormat,
  // A piece that would read past the end of the file, or write past the
  // end of its buffer. The subject names the tensor.
  OutOfRange,
  // A tensor whose rows are not a whole number of unpack's 32-weight
  // groups, which no kernel can step through, and that is not a scalar
  // (format.h). The subject names it.
  RowNotSteppable,
};

struct RouteResult {
  RouteError error = RouteError::Ok;
  std::string subject;

  [[nodiscard]] bool ok() const noexcept { return error == RouteError::Ok; }
};

// The routes for every weight in `plan`, ordered by file offset, and the plan
// as upload carries it out. `out` and `routes` are left untouched unless it
// succeeds.
[[nodiscard]] RouteResult plan_routes(const gguf::TensorIndex& index,
                                      const ResidencyPlan& plan,
                                      std::uint64_t file_size,
                                      FindFormat find_format,
                                      std::vector<Route>& routes,
                                      ResidencyPlan& out);

}  // namespace frizzle::residency
