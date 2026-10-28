#include "core/residency/mapping.h"

namespace frizzle::residency {

Mapped mapping_result(WGPUMapAsyncStatus status, bool has_range,
                      const gpu::DeviceStatus& device) noexcept {
  switch (status) {
    case WGPUMapAsyncStatus_Success:
      return has_range ? Mapped::Ok : Mapped::Internal;
    case WGPUMapAsyncStatus_CallbackCancelled:
      return Mapped::Cancelled;
    case WGPUMapAsyncStatus_Error:
      return Mapped::Internal;
    case WGPUMapAsyncStatus_Aborted:
      // A lost callback may arrive after the mapping's; without it the
      // cause is not guessed.
      return device.lost ? Mapped::DeviceLost : Mapped::Unexplained;
    default:  // the table's last row: any status webgpu.h adds later
      return Mapped::Internal;
  }
}

}  // namespace frizzle::residency
