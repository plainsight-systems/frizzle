#pragma once

#include <memory>

namespace frizzle::gpu {

// Axis D: changes with the WebGPU surface.
//
// A callback's record crosses WebGPU's C interface as userdata: handed off
// as a raw pointer when the call is made, taken back by the callback, which
// WebGPU runs exactly once. These are the only places in the harness that
// ownership becomes a raw pointer and back, so every such crossing reads the
// same way and none can leak or free twice by hand.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — except where a C
//            interface demands one, and then only here, from a unique_ptr
//            and back into one.
//     R.11   Avoid calling new and delete explicitly — the record is made
//            with make_unique and freed by the unique_ptr take_back returns.

template <typename T>
[[nodiscard]] void* hand_off(std::unique_ptr<T> record) noexcept {
  return record.release();
}

template <typename T>
[[nodiscard]] std::unique_ptr<T> take_back(void* userdata) noexcept {
  return std::unique_ptr<T>(static_cast<T*>(userdata));
}

}  // namespace frizzle::gpu
