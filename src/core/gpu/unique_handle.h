#pragma once

#include <utility>

namespace frizzle::gpu {

// Move-only owner of an opaque C handle, releasing it exactly once.
//
// WebGPU hands out reference-counted handles with paired wgpuXRelease
// functions. The wasm build compiles without exceptions, so E.25 applies:
// RAII is simulated rather than skipped. Manual release is what this replaces
// — a single early return past a release call leaks silently, and the
// compiler cannot help.
//
// Deliberately not std::unique_ptr: these handles are opaque pointers whose
// release function is not `delete`, and a custom deleter buys nothing here
// while making every alias signature noisier.
template <typename Handle, void (*ReleaseFn)(Handle)>
class UniqueHandle {
 public:
  UniqueHandle() = default;
  explicit UniqueHandle(Handle handle) noexcept : handle_(handle) {}

  ~UniqueHandle() { reset(); }

  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle& operator=(const UniqueHandle&) = delete;

  UniqueHandle(UniqueHandle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  UniqueHandle& operator=(UniqueHandle&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  // Releases any owned handle and takes ownership of `handle`.
  void reset(Handle handle = nullptr) noexcept {
    if (handle_ != nullptr) {
      ReleaseFn(handle_);
    }
    handle_ = handle;
  }

  // Relinquishes ownership without releasing. The caller becomes responsible.
  [[nodiscard]] Handle release() noexcept {
    return std::exchange(handle_, nullptr);
  }

  [[nodiscard]] Handle get() const noexcept { return handle_; }
  explicit operator bool() const noexcept { return handle_ != nullptr; }

 private:
  Handle handle_ = nullptr;
};

}  // namespace frizzle::gpu
