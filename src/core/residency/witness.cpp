// upload.h's witness_result, apart from upload.cpp so that it is built and
// tested without a device.

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
