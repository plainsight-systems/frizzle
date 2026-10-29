# Engineering decisions

Current decisions only. Remaining work is in [QUEUE.md](QUEUE.md).

## What Frizzle is

A chat model in the browser that you can reach inside. Type a prompt; it runs
through the model's chat template (control tokens are never shown) and the
model writes its reply. Open any layer, change something at the last
position, and run the model again: the reply it writes with the edit held at
every generated token is shown beside the original, both from the same seed.
The typical use: pin a neuron, or several across any layers, and see what the
model says. Frizzle shows what the model produces and computes nothing beside
it.

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
- Tests use doctest v2.4.12, which CMake fetches pinned by commit. The
  llama.cpp reference is frozen in the repository; llama.cpp itself is fetched
  only to make it again (`tools/reference_logits/`).
- The engine owns its WebGPU device in a worker. The scene uses three.js's
  WebGL renderer on the page. Captured values are copied across, all layers
  on every run (sizes in the
  [logical overview](../architecture/logical-overview.md)).
- The interface is React, react-three-fiber, TypeScript and Vite, with npm at
  exact versions and a committed lockfile. The site is Hugo.
- The app runs on Node 24.12.0, pinned in `app/.nvmrc` and refused otherwise,
  as LLVM is. Packages are installed by hand with `npm ci`, with no install
  scripts. TypeScript 7's native `tsc` only type-checks; Vite builds.
- The TypeScript suites run on Node's built-in test runner, with no test
  dependency: Node strips the types itself, so the app uses only TypeScript
  that erases (no enum, namespace or parameter properties; `tsc` enforces
  it) and tested modules use no Vite-only features. Vitest, the alternative,
  would add about 20 packages.
- The browser scenario drives Chrome for Testing 155.0.8059.84, headless,
  with puppeteer-core 25.15.0 (the Chrome build that version is made for; a
  bump of one is a bump of both) and @puppeteer/browsers 3.2.5 to install
  it, in its own npm project (`tests/integration/browser/`), so the driver's
  packages never enter the app's lockfile. Playwright was the alternative;
  its Chromium is pinned only through its own version. On an Apple M3 Max,
  2026-10-28, that Chrome gave a WebGPU device headless and headed, with no
  flags: adapter vendor apple, architecture metal-3, shader-f16.
- The scenario answers the pinned URL from the local copy and blocks every
  other host, so it needs no network. It runs before every push as the
  pre-push hook (`.githooks/`, `scripts/pre_push.py`), which rebuilds the
  browser module first; not in CI, which has no GPU. It ends once the
  edited reply differs from the original, without waiting for it to end.

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
  Set, Scale × and Zero, and the pinned neurons across all layers. The
  tokens a neuron pushes toward are deferred (2026-10-29): the model never
  produces them; they would be computed from the weights, and in Gemma 3
  only approximately, since the norm after the MLP scales each neuron by
  the whole layer's output (QUEUE, open decisions).
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

## Engine operations

- Hosts call the engine's operations alone (`src/core/runtime/engine.h`):
  `load` in three calls, and `generate`, which takes the edit set. The
  engine composes the ported pieces; edits changed three of them: the graph
  gained the edit launch, and the runtime and generator gained `rewind`.
- `generate` runs the model over the rendered text and streams its reply.
  The turn loop reuses the prefix of the prompt the cache already holds.
- The engine renders no chat template; it reads text already rendered.
- Wrapping a message is the user interface's job: the page's chat template,
  and the native tool's wrapping of one user message, write Gemma 3's
  `<bos>` and turn markers; the page's template tests (slice 3) lock it.
  The tool wraps one message only, by hand, untested. The engine checks
  nothing of the template; text wrapped otherwise runs, and the reply is what
  the model makes of it (without `<bos>`, often nothing).
- In the browser the engine runs in a plain Web Worker, as one ES module
  (`frizzle.mjs` and its `.wasm`, built by the `wasm-release` preset), with
  no SharedArrayBuffer, so no cross-origin isolation is needed. The worker
  downloads the pinned file, keeps it in OPFS only once its size and SHA-256
  match the pin, and streams it into the engine in 16 MiB chunks. A kept
  copy is trusted by its record, written only after it matched, and is not
  hashed again at every load (about 2 s for 722 MB).
- A load can be cancelled while it downloads; once it streams into the
  engine it runs to its end, a few seconds, since the engine cannot abandon
  a load half made. A generation can be cancelled at any time.
- The module's heap starts at 128 MiB and may grow to 1 GiB as a fallback;
  both are estimates, not measured. Measuring the peak is deferred
  (2026-10-29) until there is a sign it matters (QUEUE, open decisions).
- Nothing is computed beside the model: no probabilities derived on the CPU,
  no second pass to keep in step with the first.
- The page sets no limit on a reply's length: it sends the engine's own
  "no other limit", so a reply ends when the model ends its turn or the
  context (32,768 tokens) is full. A strong edit can make the model repeat
  itself to the end of the context: setting layer 12 neuron 4031 to 100 and
  layer 6 neuron 100 to -200 does, at seed 7, for "Name three colours of the
  rainbow.", natively and in the browser. Running again supersedes it; the
  page has no Stop of its own.
- The page runs the original reply, then the edited one from the same seed,
  one after the other, since the engine runs one turn at a time.

## Engine requirements

- Capture, at the last position for every layer: the neurons from slice 4
  (6,912 a layer, after the edits: about 686 KiB a run at f32); the
  residual and Q with the pass map (slice 5), and the attention weights
  over earlier positions and itself after them. Every run sends all 26
  layers to the page, so switching layers needs no request.
- Edits are a set, applied together in one pass: any number of neurons in
  any layers.
- Apply edits inside the pass, at the last prompt position and at every
  generated token. Earlier positions' cache entries are unchanged.
- Hold edits at every generated token. Generate the original and edited replies
  from the same seed.
- With no edits, or Scale × 1, the edited reply is the original reply, bit
  for bit.
- Between runs the cache holds only entries computed without edits: a run
  with edits rewinds it, when it ends, to just before the edits began.
- Edits cost something when they are off, by decision (2026-10-28): the edit
  kernel runs once a layer in every step that scores a token, edits or not.
  About 40 µs a token, from Charlotte's 1.5 µs a dispatch on one machine; an
  estimate, not measured on Gemma 3. Speed is not this demo's aim.
