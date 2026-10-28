# Engineering decisions

Current decisions only. Remaining work is in [QUEUE.md](QUEUE.md).

## What Frizzle is

A chat model in the browser that you can reach inside. Type a prompt; it runs
through the model's chat template (control tokens are never shown) and the
page shows the first token of the reply. Open any layer, change something at
the last position, and the first-token prediction updates live. On release,
the whole reply is generated with the edit held at every generated token and
shown beside the original reply, both from the same seed. The typical use:
pin a neuron, or several across any layers, and see what the model says.

The audience is curious people, not interpretability researchers. Frizzle does
not build on published interpretability resources.

## Scope: the last position

- The model predicts from the last position only; earlier tokens reach it
  through the keys and values attention reads.
- Everything shown and editable belongs to the last position. Earlier tokens
  are context: the prompt and their cached keys and values.
- Editing earlier positions is out of scope.

## Model

Gemma 3 1B, instruction-tuned, Q4_0 GGUF, pinned in
[`config/model.json`](../../config/model.json): `unsloth/gemma-3-1b-it-GGUF`,
721,918,496 bytes. 26 layers, residual width 1,152, MLP width 6,912, 4 query
heads sharing 1 key/value head of 256, attention local (512 tokens) in five of
every six layers and global in every sixth. Q4_0 attention, gate and up; Q4_1
for three `ffn_down`; F32 norms; a Q8_0 token embedding of about 321 MB that is
also the output head. The KV cache is F16. Decode speed in the browser is
estimated at about 140 tok/s and has not been measured.

## Platform and toolchain

- Browser only, on WebGPU. The engine is C++23, built by
  `emscripten/emsdk:6.0.8` (clang `24.0.0git`) with its built-in
  `emdawnwebgpu` port.
- Native builds exist for analysis and tests, on macOS arm64 and Linux
  x86-64, compiled by LLVM 21.1.8 clang against prebuilt Dawn
  `v20260423.175430` (commit `31e25af2`), the release the browser port comes
  from, so both builds compile against one `webgpu.h`. Moving either pin moves
  both.
- clang-format and clang-tidy are pinned to LLVM 21.1.8; tidy reads the
  native build's compilation database. Tools are installed locally; scripts
  refuse other versions and never install anything.
- The engine owns its WebGPU device in a worker. The scene uses three.js's
  WebGL renderer on the page. Captured values are copied across, all layers
  on every prediction (sizes in the
  [logical overview](../architecture/logical-overview.md)).
- The interface is React, react-three-fiber, TypeScript and Vite, with npm at
  exact versions and a committed lockfile. The site is Hugo.

## Relationship to Charlotte

Built separately, not forked. Charlotte's Gemma 3 path, 111 files listed in
the [inventory](../research/2026-10-27-charlotte-inventory.md), is copied
from commit `e003048` as the starting point and maintained here. One registry
is trimmed, the WebAssembly boundary is rewritten, the namespace `bllm`
becomes `frizzle` (a mechanical rename), and model preflight is
dropped; the context is sized to the device by the residency planner,
which comes over whole.

## Interface

- **The pass map:** one layer at a time, isometric, each stage drawn as the
  last position's vector at its real width. Arrow keys step layers.
- **Workbenches:** a working view per stage, wide vectors folded into
  grids (neurons 96 × 72). The MLP workbench is the layer deck: one card per
  layer, its neurons as the grid, positive amber and negative cool blue;
  scrolling slides the front card away to the next layer. Beside it, for the
  front layer: a ranked list, search by neuron number, a neuron card with
  Set, Scale × and Zero and the tokens the neuron pushes toward, and the
  pinned neurons across all layers.
- **What can change:** MLP neurons (set, scale, zero), any number at once, in
  one layer or several; each pinned neuron is listed and removable. Editing
  the last position's residual dimensions (set or nudge) is wanted next.
  Switching off a head and skipping a layer's output are candidates.
- **Theme:** the site uses the Plainsight family shell (navy masthead, ivory
  pages, brass, IBM Plex Serif and Mono). The app uses "night glass" from the
  banner: midnight surfaces, values as amber points of light. The masthead is
  constant; entering the app the page turns to night, the banner zooming into
  the app (a plain cut under reduced motion); a live figure in writing sits as
  a framed print in a cream border.

Concepts for each are in [`docs/design/concepts/`](../design/concepts/).

## Engine requirements

- Capture, at the last position for every layer: the residual, Q, attention
  weights over earlier positions and itself, and the neurons. Every prediction
  sends all 26 layers to the page (about 1.1 MiB for a 512-token prompt), so
  switching layers needs no request.
- Edits are a set, applied together in one pass: any number of neurons in
  any layers.
- Apply edits inside the pass. Rerun the last position from the first layer
  where the edit set differs from the one it was computed under (removing an
  edit is a difference). Earlier positions' cache entries are unchanged; the
  last position's own keys and values from that layer on are overwritten in
  place.
- Hold edits at every generated token. Generate the original and edited replies
  from the same seed.
- With no edits, or Scale × 1, the rerun reproduces the original prediction
  bit for bit.
