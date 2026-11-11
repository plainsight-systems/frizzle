#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace frizzle::kernels {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The order a program reports its outstanding steps in (kernels/program.h):
// the order they were run, whatever order they settle in. WebGPU does not
// order the mappings of two buffers, so a step may settle before the one run
// ahead of it; it is held until that one has been reported.
//
// Contract
//   - run(): a step is run; returns its number, counted from 0 as steps are
//     run. Precondition: !full().
//   - settle(step): step `step` has settled. Precondition: it is
//     outstanding, not yet settled.
//   - next(): the next step to report, if it has settled, now counted
//     reported; otherwise none, an earlier step still to settle.
//   - full(): kOutstanding (2) steps are outstanding. idle(): none is.
//   - Invariant: next_report <= every outstanding step < next_run, and
//     next_run − next_report <= kOutstanding, kept by the preconditions.
//     They are stated, not checked: a broken one reports a step out of order
//     or before it settled, and nothing reports that.
//   - No GPU object and no I/O; every call is noexcept. So the order can be
//     exercised with steps settling out of order, which a device cannot be
//     made to do on demand. No test does (tests/README.md).
//
// Algorithm
//   1. Step n's slot is n mod kOutstanding, holding whether it has settled.
//   2. run clears next_run's slot and returns next_run, then adds 1 to it.
//   3. settle sets step's slot.
//   4. next returns none when idle or when next_report's slot is clear;
//      otherwise it returns next_report, then adds 1 to it.
//
// Design and cost
//   - Two steps in flight, so the CPU may encode and submit one step while
//     the GPU runs the one before (GPU.7); that overlap is claimed only once
//     a timeline shows it, and none has. The program keeps its per-step
//     outstanding state, readback slots and timers kOutstanding deep, a step
//     in slot n mod kOutstanding (program.cpp).
//   - Counted: two 64-bit counters and two flags, 24 bytes with padding.
//     Each call reads or writes a counter and a flag, in constant time, and
//     allocates nothing.
//   - A slot is read through a span, never by subscripting the array: n mod
//     kOutstanding is within it by arithmetic, and the span checks nothing.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant: the one above.
//     I.5    State preconditions: run's and settle's, stated and not
//            checked.
//     E.12   noexcept where a throw is impossible: run, settle and next only
//            read and write the counters and flags.
//     SL.con.3 Avoid bounds errors: a slot's index is n mod kOutstanding, in
//            bounds by arithmetic rather than by a check.
//   C++ performance guidelines
//     GPU.7  Pipeline CPU and GPU work with multi-buffered resources: two
//            steps in flight, reported in the order they ran; the overlap
//            is not measured.
//   Searched, and no rule applies: bounds-checked access to a small fixed
//   array (the rules found are for fixed-capacity containers, heaps and
//   direct-indexed block tables).
class Order {
 public:
  static constexpr std::uint64_t kOutstanding = 2;

  [[nodiscard]] bool full() const noexcept {
    return next_run_ - next_report_ >= kOutstanding;
  }
  [[nodiscard]] bool idle() const noexcept { return next_run_ == next_report_; }

  // A step run: its number. Precondition: !full().
  std::uint64_t run() noexcept {
    settled(next_run_) = false;
    return next_run_++;
  }

  // Step `step` has settled. Precondition: it is outstanding, not yet
  // settled.
  void settle(std::uint64_t step) noexcept { settled(step) = true; }

  // The next step to report, if it has settled, now counted reported;
  // otherwise none, an earlier step still to settle.
  std::optional<std::uint64_t> next() noexcept {
    if (idle() || !settled(next_report_)) return std::nullopt;
    return next_report_++;
  }

 private:
  // Step `step`'s slot. step % kOutstanding is always within settled_; it is
  // read through a span, which checks nothing.
  [[nodiscard]] bool& settled(std::uint64_t step) noexcept {
    return std::span(settled_)[step % kOutstanding];
  }

  std::uint64_t next_run_ = 0;
  std::uint64_t next_report_ = 0;
  std::array<bool, kOutstanding> settled_{};
};

}  // namespace frizzle::kernels
