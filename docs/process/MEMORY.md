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
- The deck's dependency change (slice 4, step 3): three 0.186.1 (MIT, no
  dependencies of its own) and @react-three/fiber 9.8.2 (MIT), with
  @types/three 0.186.0 for the type check only. They added 19 packages to
  the lockfile, none with an install script: 13 at run time (three, fiber
  and its zustand, its-fine, suspend-react, react-use-measure,
  use-sync-external-store, buffer, base64-js, ieee754, @babel/runtime and
  two type packages) and 6 for the type check (@types/three and what it
  names, never bundled). Fiber's peers cap React below 19.4, so a React
  bump past that waits on fiber; fiber 10 is still alpha and canary. The
  page's script grew from 220 KB to 1,150 KB (317 KB gzipped), mostly
  three.js, past Vite's 500 KB warning; splitting it is not done. three.js
  and fiber are imported only by `views/` (`check.py boundaries`).
- IBM Plex Serif and Plex Mono, Regular and Medium, are kept in the app as
  four woff2 files with IBM's OFL licence (`app/src/theme/fonts/`), copied
  from @ibm/plex-serif 2.0.0 and @ibm/plex-mono 2.5.0 and recorded by
  SHA-256. The packages are not installed: both depend on
  @ibm/telemetry-js, which would enter the lockfile with an install script
  that only `.npmrc` keeps from running.
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
  browser module first; not in CI, which has no GPU. Its pins are one of
  each kind (layer 2 neuron 1000 scaled by -4, layer 16 neuron 3000 set to
  8, layer 24 neuron 100 zeroed, seed 7): they change the reply and let it
  end by itself, since captures arrive with a run's end, and each pin is
  read back from the page, exactly.

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
  layer, its neurons as the grid, a heat map of each neuron's size, either sign
  (2026-10-29: a neuron pushes the residual as far either way, so the
  sign is the tip's and the inspector's; small values stay dark);
  scrolling slides the front card away to the next layer. Beside it, for the
  front layer: the strongest neurons, the neuron card with Set, Scale ×
  and Zero, and Go to a neuron by number. Neurons are picked by clicking
  the grid, many by dragging a box (2026-10-30), and a card zooms and
  pans; Set, Scale × and Zero apply to the whole selection, the engine
  judging how many pins a run takes. The pins wait on the prompt, as
  chips, one a layer and edit, for the next run. On the reply's card a
  pinned cell shows the value it will send, so a pin answers on the grid
  at once; the steered reply's card shows what its run used. The
  workbench gives the deck the focus on opening, so the wheel and keys
  work without a first click on the grid.
- **The prompt starts it all (2026-10-30):** Load model, with the model's
  size and then its progress, sits on the prompt in Run's place until the
  model is ready. Each
  card is coloured against its own layer's 99th-percentile magnitude, so a
  few outliers do not darken the rest; the legend gives that scale to two
  significant digits; infinities are full heat, NaN
  magenta, and none sets the scale. The deck shows one side's neurons at
  the branch point at a time, the reply's or the steered reply's, by
  choice; the neuron card shows both. The page is the conversation, full width, with the replies side
  by side; the workbench is a layer over the whole window, opened from the
  prompt or a reply, and Run closes it so the replies show as they
  stream: reading and working on neurons happen in turns, and side by
  side each was too narrow. Sizes are in megabytes; values are
  rounded for reading, exact in a `<data value>`. The
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
  rainbow.", natively and in the browser. Stop beside Run ends it
  (2026-10-31: without one, a steered reply that ran away left Accept and
  the words disabled, a dead end): the run settles "done", end
  "cancelled", with no captures, and can be steered from or accepted; an
  accepted stopped reply's neurons at the point come from an inspection.
  Running again still supersedes it.
- **Steering from a word (2026-10-30, slice 4 step 4):** the reply on the
  left is words once done; clicking one sets the branch point after it,
  and the deck shows every layer's neurons there, read by an inspection
  (no tokens generated) unless the reply's own captures were taken there.
  Run, with the reply done and the same message and seed, generates from
  the point with the pins held into the right column, keeping the reply's
  text before it; a new message or seed starts a new reply, then steers
  from its start when pins are held. Accept makes the steered reply the
  reply, its edits its history, the point where it branched; the pins are
  kept throughout. Requests run one at a time, since the engine runs one
  turn at a time: a branch follows the reply and waits for an inspection
  in flight, and no word is chosen while a reply runs.

## Engine requirements

- Capture, at the last position for every layer: the neurons from slice 4
  (6,912 a layer, after the edits: about 702 KiB a run at f32); the
  residual and Q with the pass map (slice 5), and the attention weights
  over earlier positions and itself after them. Every run sends all 26
  layers to the page, so switching layers needs no request.
- Edits are a set, applied together in one pass: any number of neurons in
  any layers.
- Apply edits inside the pass, each over a range of steps, from a branch
  point in the reply on (2026-10-30, slice 4 step 4). Step k is the position
  that draws reply token k; a branch keeps the reply's first k tokens,
  which the page sends, and recomputes step k.
- Hold edits at every generated token. Generate a reply and every branch
  from it with the same seed.
- With no edits, or Scale × 1, a steered reply is the reply without them,
  bit for bit; a branch with a reply's own edits continues it bit for bit.
- The cache keeps edited entries, with the edits of the run that wrote
  them, and a run reuses it only to the first position whose tokens or
  edits in force differ (2026-10-30, replacing the rewind after every
  edited run, which would discard an accepted reply's entries).
- An accepted reply, steered from a word, is the next reply to steer
  from; the pins are kept throughout (the user, 2026-10-30).
- Captures are taken in the step that ends the prompt, after that layer's
  edit, and read back once a run, after the turn; a turn that is cancelled
  or fails reports none, since the store may still hold an earlier run's.
  The capture launch runs once a layer in every step that scores a token
  and stops at once in decode: about 40 µs a token, an estimate, as for
  edits.
- Edits cost something when they are off, by decision (2026-10-28): the edit
  kernel runs once a layer in every step that scores a token, edits or not.
  About 40 µs a token, from Charlotte's 1.5 µs a dispatch on one machine; an
  estimate, not measured on Gemma 3. Speed is not this demo's aim.
