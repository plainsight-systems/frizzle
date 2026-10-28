#pragma once

#include <cstdint>

namespace frizzle {

// Serialises one asynchronous run at a time, and makes a late callback
// identifiable as late.
//
// WebGPU requests cannot be cancelled. A request that has already been
// declared timed out may still complete afterwards, and its callback will
// fire regardless. Without a generation stamp there is no way to distinguish
// that callback from a live one, so it would report a second result for a run
// already reported as failed.
//
// The rule is first-past-the-post: whichever of the real callback and the
// timeout calls complete() first owns the run, and the loser is stale.
class RunGuard {
 public:
  // Never a valid generation, so it doubles as "no run".
  static constexpr std::uint32_t kNoRun = 0;

  // Begins a run and returns its generation, or kNoRun if one is already
  // live. A caller that receives kNoRun must not start work.
  std::uint32_t begin() noexcept {
    if (active_) {
      return kNoRun;
    }
    active_ = true;
    // Wraps past UINT32_MAX, skipping kNoRun. Distinguishing two runs
    // 2^32 apart is not a case that can arise here; skipping zero is what
    // matters, because it would alias "no run".
    if (++generation_ == kNoRun) {
      ++generation_;
    }
    return generation_;
  }

  // Claims completion of `generation`. True only for the live run; a stale
  // or duplicate generation returns false and leaves the guard untouched,
  // so the caller knows to report nothing.
  bool complete(std::uint32_t generation) noexcept {
    if (!active_ || generation != generation_ || generation == kNoRun) {
      return false;
    }
    active_ = false;
    return true;
  }

  bool active() const noexcept { return active_; }
  std::uint32_t generation() const noexcept { return generation_; }

 private:
  bool active_ = false;
  std::uint32_t generation_ = kNoRun;
};

}  // namespace frizzle
