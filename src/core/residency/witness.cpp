// Axis D: changes with the WebGPU surface used to create and fill buffers, or
// how a device reports their failure.
//
// Contract, algorithm, cost and corpus guidance: upload.h. This file defines
// what that header states for witness_result, apart from upload.cpp so that
// it builds without a device.

#include <algorithm>

#include "core/residency/upload.h"

namespace frizzle::residency {

UploadError witness_result(Mapped mapping, std::span<const std::byte> mapped,
                           std::span<const std::byte, 4> expected) noexcept {
  switch (mapping) {
    case Mapped::Ok:
      return std::ranges::equal(mapped, expected)
                 ? UploadError::Ok
                 : UploadError::WitnessMismatch;
    case Mapped::Cancelled:
      return UploadError::Cancelled;
    case Mapped::Internal:
      return UploadError::Internal;
    case Mapped::DeviceLost:
      return UploadError::DeviceLost;
    case Mapped::Unexplained:
      return UploadError::Unconfirmed;
  }
  return UploadError::Internal;  // no Mapped value reaches here; -Wswitch names
                                 // a new one
}

}  // namespace frizzle::residency
