// Axis Q: changes with a kind of edit, or what one does.
//
// Contract, algorithm, cost and corpus guidance: edit_kernel.h. This file
// defines what that header states.

#include "core/edit/edit_kernel.h"

#include <cstddef>
#include <cstring>
#include <vector>

#include "frizzle/shaders_generated.h"

namespace frizzle::edit {
namespace {

// Binding 1, as edit.wgsl's EditLaunch lays it out.
struct Constants {
  std::uint32_t layer;
  std::uint32_t width;
};

// One workgroup a row: a layer's edits are few, and with none each reads
// two words.
constexpr std::uint32_t kWorkgroup = 64;

kernels::Binding whole(const residency::BufferRange& range) {
  return {range.buffer, range.offset, range.length};
}

}  // namespace

kernels::Launch edit_launch(std::uint32_t layer,
                            const residency::BufferRange& activation,
                            const residency::BufferRange& table,
                            std::uint32_t width) {
  const Constants constants{layer, width};
  std::vector<std::byte> bytes(sizeof constants);
  std::memcpy(bytes.data(), &constants, sizeof constants);
  return kernels::Launch{shaders::edit,
                         nullptr,
                         std::move(bytes),
                         {whole(table), whole(activation)},
                         kWorkgroup,
                         kWorkgroup,
                         kernels::Rows::EveryToken,
                         {}};
}

}  // namespace frizzle::edit
