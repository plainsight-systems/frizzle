#pragma once

// Compile-time gate for diagnostic instrumentation.
//
// Three configurations, and the axis is instrumentation rather than
// optimisation:
//
//   clean       Release, gate OFF  — throughput numbers you may quote
//   diagnostic  Release, gate ON   — finding out why, knowingly perturbed
//   debug       Debug,   gate ON   — correctness, not performance at all
//
// The diagnostic build is Release on purpose. A Debug build cannot diagnose a
// Release performance problem, because the optimiser is off and the code being
// measured is not the code that ships.
//
// Default 0 here, so a translation unit that never receives the build-system
// definition still compiles, and compiles to the free version. Instrumentation
// is opt-in at configuration level; forgetting it cannot silently cost
// anything.
#ifndef FRIZZLE_DIAGNOSTICS_ENABLED
#define FRIZZLE_DIAGNOSTICS_ENABLED 0
#endif

namespace frizzle {

// Lets code branch at compile time on whether instrumentation exists, without
// #ifdef at the call site.
[[nodiscard]] constexpr bool diagnostics_enabled() noexcept {
  return FRIZZLE_DIAGNOSTICS_ENABLED != 0;
}

}  // namespace frizzle
