#pragma once

#include <webgpu/webgpu.h>

#include "core/gpu/device.h"

namespace frizzle::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// What a buffer mapping's callback says, decided once, for every mapping
// residency makes: upload's witness (upload.h) and each chunk the diagnostic
// check reads back (upload_check.h). A completed mapping is the one thing a
// lost device cannot give (gpu/device.h), so how its outcome is read decides
// whether success can be claimed at all; it is therefore one deterministic
// function, tested branch by branch without a device.
//
//   Success, with a range     Ok
//   Success, without a range  Internal: a mapping that cannot be read
//   CallbackCancelled         Cancelled: the instance went away
//   Error                     Internal: a mapping this harness should not
//                             have asked for
//   Aborted                   DeviceLost if the device's status already says
//                             it is lost; otherwise Unexplained, since the
//                             lost callback may not have arrived yet and the
//                             cause is not guessed
//   any other status          Internal
//
// What was mapped is the caller's to judge: the witness compares its four
// bytes, the check a chunk's ranges.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     E.27   If you can't throw exceptions, use error codes systematically —
//            every status has exactly one outcome.
//     P.6    What cannot be checked at compile time should be checkable at run
//            time — and is: the table above is the test.

enum class Mapped {
  Ok,
  Cancelled,
  Internal,
  DeviceLost,
  Unexplained,
};

[[nodiscard]] Mapped mapping_result(WGPUMapAsyncStatus status, bool has_range,
                                    const gpu::DeviceStatus& device) noexcept;

}  // namespace frizzle::residency
