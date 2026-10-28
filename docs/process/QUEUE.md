# Remaining work

Slice 1 is built: the port, the engine's operations, the native tool and its
tests. The work is cut into slices, in order. Each ends in
something shown working, and the tests from the
[inventory](../../tests/README.md) that guard it land with it.

## 0. Architecture

- Logical overview, change axes and file mapping in `docs/architecture/`,
  diagrams first, covering the ported Gemma 3 path and the new capture and
  edit work.

## 1. The original reply, native

- Bring over Charlotte's Gemma 3 path as inventoried, with the build: CMake,
  the native and browser presets, shader embedding, Dawn and Emscripten pins.
  Rename `bllm` to `frizzle` and `BLLM_` to `FRIZZLE_` as the files land.
- Trim `capability.cpp`; drop preflight (the context is sized in
  `residency/plan`, which comes over whole).
- `check.py boundaries`, its C++ rule: the core includes nothing from
  `src/wasm/`, `tools/` or `tests/`.
- The engine's operations `load` and `generate`
  (src/core/runtime/engine.h).
- A native tool prints the reply to a prompt, through those operations.

## 2. Pinned neurons, native

- `generate` takes the edit set: edits as a set across layers, applied
  inside the pass at the last prompt position and held at every generated
  token. Captures at the last prompt position for every layer, sent with the
  run.
- The native tool prints the edited reply beside the original.
- Tests: replay and edits, edit specs, the kernels suite (the ported kernels,
  and the capture and edit outputs).

## 3. The loop in a browser, plain

- The Emscripten build; scaffold `app/` with its pinned dependencies; port the
  worker, protocol, download, verification, cache and chat-template modules
  to TypeScript.
- A plain page: prompt, pin neurons by number, the two replies side by
  side.
- `check.py boundaries`, its TypeScript rules (React only in the root,
  `views/` and `theme/`; the worker imports no page code).
- Measure the browser decode speed for Gemma 3 1B.
- Tests: browser end to end.

## 4. MLP workbench

- The layer deck (one folded grid per layer, sliding away as you scroll),
  ranked list, search, the neuron card with its push-toward tokens, and the
  pinned list, in the night-glass theme.

## 5. Pass map

- The isometric layer in three.js, stages at their real widths, layer
  stepping, clicking into a workbench.
- Attention weights written out at the last position (new kernel output),
  with its kernel test.

## 6. The site

- Scaffold `web/` in Hugo with the family shell, hosting the built app, and
  the hand-over into the app.
- Tests: site build.

## 7. Residual dimensions

- The residual workbench (the 48 × 24 grid) and its edits, extending the edit
  specs and replay tests.

## Open decisions

- For slice 7: whether a residual dimension is set or nudged.
- Whether to offer switching off a head or skipping a layer's output.
- Whether the map shows gate and up separately (new kernel outputs) or only
  the neurons.
- An early readout: what the model would predict if it stopped at a layer.
- Workbenches for stages other than the MLP.
- Supported browsers and devices.
- Measuring Gemma 3 1B's sampling settings; until then it runs on the
  defaults for an unmeasured model (policy.h).
- Which libc++ the native build links on macOS: today the system's
  `/usr/lib/libc++.1.dylib`, while compiling against LLVM 21.1.8's headers.
- Hosting and CI.
