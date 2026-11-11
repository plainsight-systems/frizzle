#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>

#include "core/gpu/device.h"

namespace frizzle::residency {

// Axis D: changes with the WebGPU surface used to read buffers back, or how a
// device reports their failure.
//
// What a buffer mapping's callback says, decided once, for three users:
// upload's witness (residency/upload.h), each chunk the diagnostic check
// reads back (residency/upload_check.h), and the capture store's read
// (capture/reader.h). A completed mapping is the one thing a
// lost device cannot give (gpu/device.h), so how its outcome is read decides
// whether success can be claimed at all; it is therefore one deterministic
// function.
//
// Contract
//   - mapping_result(status, has_range, device): the outcome of one mapping,
//     from its callback's status, whether the mapped range came back, and
//     the device's status as the caller holds it. Pure and noexcept; it
//     reads `device` during the call and keeps nothing.
//
//       Success, with a range     Ok
//       Success, without a range  Internal: a mapping that cannot be read
//       CallbackCancelled         Cancelled: the instance went away
//       Error                     Internal: a mapping this engine should not
//                                 have asked for
//       Aborted                   DeviceLost if the device's status already
//                                 says it is lost; otherwise Unexplained,
//                                 since the lost callback may not have
//                                 arrived yet and the cause is not guessed
//       any other status          Internal
//
//   - What was mapped is the caller's to judge: the witness compares its
//     four bytes, the check a chunk's ranges, the reader hands the store's
//     words on.
//
// Algorithm: one switch over the status, row by row as the table; for
// Aborted, the device's lost flag decides.
//
// Design and cost
//   - Constant time, once a mapping: once a load for the witness, once a
//     chunk in a diagnostic build's check, once a run for the captures.
//   - It needs no device, so each row of the table can be checked without
//     one. No test does (tests/README.md; upload.h says the same).
//   - Mapped takes one byte, the smallest that holds its five values. It is
//     switched on and held in the engine's captured result, never written
//     to a GPU buffer, a file or a message; the module sends it to the page
//     as a name (src/wasm/bindings.cpp).
//
// Corpus guidance
//   C++ Core Guidelines
//     E.27   If you can't throw exceptions, use error codes systematically:
//            every status has exactly one outcome.
//     ES.79  Use default to handle common cases (only): the default is the
//            table's last row, a status webgpu.h may add later, Internal.
//     F.8    Prefer pure functions: the outcome depends on the three
//            arguments alone.
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: being pure, every row can be checked at run time
//            without a device.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            Mapped takes one byte, as the others in residency do.
//   C++ performance guidelines
//     Searched for reading a GPU mapping's completion status: the rules
//     found budget readbacks and pipeline them (GPU.1, GPU.7), and none
//     governs how a status is read. Searched for an enumeration's size: no
//     rule.

enum class Mapped : std::uint8_t {
  Ok,
  Cancelled,
  Internal,
  DeviceLost,
  Unexplained,
};

[[nodiscard]] Mapped mapping_result(WGPUMapAsyncStatus status, bool has_range,
                                    const gpu::DeviceStatus& device) noexcept;

}  // namespace frizzle::residency
