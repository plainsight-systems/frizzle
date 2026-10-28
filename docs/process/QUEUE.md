# Remaining work

Nothing is built yet. In order:

## Architecture

- Write the logical overview, change axes and file mapping in
  `docs/architecture/`, covering the ported Gemma 3 path and the new capture
  and edit work.

## Engine

- Bring over Charlotte's Gemma 3 path as inventoried, with the build: CMake,
  the native and browser presets, shader embedding, Dawn and Emscripten pins.
- Trim `capability.cpp`; extract the context sizing from preflight.
- Design and build captures at the last position, edits applied inside the
  pass, writing the cache at a position, and edits held through decoding.
- Measure the browser decode speed for Gemma 3 1B.

## Interface and site

- Scaffold `app/` with its pinned dependencies; port the worker, protocol,
  download, verification, cache and chat-template modules to TypeScript.
- Build the pass map and the MLP workbench against the real engine.
- Scaffold `web/` in Hugo with the family shell, hosting the built app.

## Tests

- Build the tests in the [inventory](../../tests/README.md), each shown to
  fail before it is kept.

## Open decisions

- Editing the last position's residual dimensions is wanted next, after
  neurons: its workbench (the 48 × 24 grid) and whether it is set or nudged.
- Whether to offer switching off a head or skipping a layer's output.
- Whether the map shows gate and up separately (new kernel outputs) or only
  the neurons.
- An early readout: what the model would predict if it stopped at a layer.
- Workbenches for stages other than the MLP.
- Supported browsers and devices.
- Hosting and CI.
