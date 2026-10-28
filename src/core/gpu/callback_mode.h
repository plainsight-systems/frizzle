#pragma once

#include <webgpu/webgpu.h>

namespace frizzle::gpu {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// How the WebGPU callbacks this harness asks for are delivered: every
// callback given a mode — requests, mappings, queue work, error scopes,
// device loss. The harness is single-threaded (no pthreads: GitHub Pages
// cannot set COOP/COEP), and each of these must run on that one thread,
// between the harness's own steps.
//
//   - In the browser, AllowSpontaneous: emdawnwebgpu runs callbacks from the
//     worker's own event loop, the one thread there is, and nothing else
//     would ever run them.
//   - Natively, AllowProcessEvents: Dawn may run a spontaneous callback on a
//     thread of its own, which the harness's state is not built for. Here a
//     callback runs only inside wgpuInstanceProcessEvents, which the native
//     tests call from their one thread (tests/gpu/support/pump.h).
//
// One callback is outside any mode: WebGPU fires the uncaptured-error
// callback spontaneously, always, and natively possibly on another thread
// (webgpu.h's asynchronous-operation rules). So it may touch no harness
// state; device.cpp's only writes to stderr. Anything the harness must act on
// is caught by an error scope instead, whose result arrives in this mode.
//
// The build chooses, not the core: CMakeLists.txt defines
// FRIZZLE_GPU_PROCESS_EVENTS as 1 natively and 0 for the browser, and a build
// that defines neither does not compile, so none can fall into the wrong mode
// unnoticed. The only place the two builds differ in how the GPU talks back.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     CP.1   Assume that your code will run as part of a multi-threaded
//            program — natively Dawn has threads of its own; no harness state
//            is reachable from one, and the one callback that can run on one
//            touches none.
//   C++ performance guidelines
//     WASM.12 Keep the compute core natively buildable so it can be profiled
//            properly — the harness's WebGPU code runs unchanged against
//            Dawn natively and the browser's WebGPU; this mode is the one
//            difference, and CMake, not the code, chooses it.
#if !defined(FRIZZLE_GPU_PROCESS_EVENTS)
#error \
    "FRIZZLE_GPU_PROCESS_EVENTS is set by CMakeLists.txt: 1 natively, 0 in the browser"
#endif

inline constexpr WGPUCallbackMode kCallbackMode =
#if FRIZZLE_GPU_PROCESS_EVENTS
    WGPUCallbackMode_AllowProcessEvents;
#else
    WGPUCallbackMode_AllowSpontaneous;
#endif

}  // namespace frizzle::gpu
