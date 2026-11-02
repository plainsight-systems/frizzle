# Remaining work

Slice 1 is built: the port, the engine's operations, the native tool and its
tests. Slice 2 is built: edits reach the model through `generate`, with the edit
kernel and the cache rewind; the native tool's `--pin` prints the edited
reply after the original; the edit specs, the edit kernel's unit test and
the replay and edits scenario guard it. The ported kernel families get no
unit tests of their own; the oracle covers them end to end.

The work is cut into slices, in order. Each ends in something shown working.
Each step in a slice runs headers, then tests, then code (workflow.md): the
tests from the [inventory](../../tests/README.md) that guard a step are
written in that step, before its code. A slice's "Tests" line names them;
from slice 3 on, the plan for each step assigns them.

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
  token. When a run with edits ends, the cache rewinds to just before
  the edits began, so it holds no edited entry between runs.
- The native tool prints the edited reply beside the original.
- Tests: replay and edits, edit specs, the kernels suite (the edit kernel;
  the oracle covers the ported kernels).

## 3. The loop in a browser, plain

Ends with a plain page in a real browser: download and
verify the pinned file, type a message, pin neurons by number, and see the
edited reply beside the original. Each step: headers reviewed, then its
tests, then its code.

1. **The app's scaffold and its rules.** `app/` with Vite, React and
   TypeScript at exact versions and a committed lockfile; the page and
   worker entries; a TypeScript test runner. `check.py` gains the app's
   type check and its tests, and `check.py boundaries` its TypeScript rules
   (React only in the root, `views/` and `theme/`; the worker imports no
   page code), in force before the first module lands.
   Tests: the boundary rules run in `check.py all`; nothing else to lock.
   The test runner is Node's own (MEMORY). The C++ boundary rule listed in
   slice 1 lands here: it had not.
2. **The chat template** (`worker/template/`, T). Gemma 3's template, read
   from the file's `tokenizer.chat_template`, rendered by a vendored Jinja.
   Tests first, a new unit suite, Chat template: one user message renders
   `<bos><start_of_turn>user\n{message, trimmed}<end_of_turn>\n<start_of_turn>model\n`;
   a conversation renders each turn, assistant as `model`; a system prompt
   joins the first user turn; turns that do not alternate are refused by
   name. Against Charlotte's Gemma template fixture, rewritten to these
   contracts.
3. **The worker side, end to end** (`worker/delivery/`, L; `src/wasm/`,
   R; `worker/engine/`, `worker/worker.ts`, `protocol/`). Delivery: fetch
   the pinned file in chunks, SHA-256 as it streams, keep it in OPFS only
   once it matches, and stream it into the engine. The Emscripten build of
   the engine and its boundary: load (in its three calls) and generate with
   edits, streaming text; the engine reports the file's chat template,
   cancels a turn, and gives the bytes a short front needs as a number.
   The worker answers load, generate and cancel; the page's client gives
   each request one answer, the latest generation winning. Tests first,
   two new unit suites: Delivery and Protocol (tests/README.md). The
   boundary is thin translation, locked by the browser scenario.
4. **The plain page** (`views/prompt/`, H). A message box, pins by number
   (layer, neuron, kind, value), the two replies side by side, the same
   seed. Tests first, the integration scenario Browser, end to end
   (slot 3 of 5): in a real browser with WebGPU, fetch and verify the
   pinned file (answered from the local copy, no network), generate a
   reply, pin neurons in two layers, and get an edited reply that differs
   from the original beside it. Run as the pre-push hook, not in CI.
   Decision: puppeteer-core driving Chrome for Testing, at their pins.

## 4. MLP workbench

Ends with the layer deck in the night-glass theme: every run's neurons at
the last position, in all 26 layers, with the ranked list, search, the
neuron card and the pins beside it. Captures were the neurons only until
step 5, which adds the residual, what each block adds and the attention
weights; Q comes with the pass map. Each step: headers and tests
reviewed, then its code, then review.

1. **Captures in the engine** (C++, O). The capture store in
   `core/capture/`, a slot per layer, filled at the last prompt position
   after the edit launch, so an edited run holds the value each pin set;
   decode steps leave it as it is; `generate`'s end carries it, read back
   once a run. Tests: one unit test of the capture kernel and its reader
   on the GPU, with no model, in the kernels suite: the last row, in the
   step that ends the prompt and no other, into its own layer's slot, bit
   for bit. No integration test is added: the browser read-back
   (step 2) checks the wiring, and the oracle the values the model makes.
2. **Captures reach the page** (bindings, worker, `protocol/`, session).
   Each run's store crosses to the page; the session keeps the original's
   and the edited's. The plain page shows each pin's neuron in both runs.
   Tests: the browser scenario, with one pin of each kind, reads each pin
   back from the page: in the edited run, exactly what it sent (a set's
   value, zero, or the original's value scaled, in f32). That checks every
   edit crosses the page, worker and module exactly as sent, which a
   changed reply only suggests. No protocol test: the client passes the
   answer through untouched, so one would lock only the browser's message
   copying. Before the theme, so the check never depends on the scene.
3. **The workbench** (`theme/`, `views/`, `shape/`, session). Night glass
   and the IBM Plex fonts kept in the app; a header with the GPU and the
   model in megabytes; the conversation full width (the prompt with Run
   on it and the pins as chips, then the two replies side by side); and
   the workbench, a layer over the whole window opened from the prompt or
   a reply and closed by Run: the layer deck (one 96 × 72 card per layer,
   sliding away as you scroll, a heat map of each neuron's size; neurons
   picked by clicking, or many by dragging a box; zoom and pan; a pinned
   cell on the original reply's card showing what it will send) beside the
   inspector (which reply, the strongest in the front layer, the neuron
   or selection card with Set, Scale × and Zero, Go to a neuron). A range
   pinned at once is one change and one chip. The panels, planned as a step 4, are in this
   step: without them the deck could not be used as designed. three.js
   and react-three-fiber join the app, a dependency change declared as one
   (MEMORY, Platform). Tests: a new unit suite, Shape: what the deck shows
   at a cell is the neuron folded there, a heat map of its size against
   its layer's 99th-percentile magnitude, with infinities, NaN and empty
   cells named; a box of cells back to neurons; overlays at the slot's
   scale; ranking by strength; sizes and values as people read them. A
   second, Pins: the pin set (adding, removing, grouping) and what a pin
   sends.
   The browser scenario makes its pins through Go to and the neuron card,
   and reads them back from the card's exact values (a contract change of
   how it drives the page, not of what it checks).
4. **Steering from a word** (edit, cache, runtime, bindings, `protocol/`,
   session, views). The reply on the left is read as words; clicking
   one sets the branch point after it, and the deck shows the neurons
   there, at the word's last token, where the model has read the word and
   chooses what follows. Run generates forward from that point with the
   pins held into the right column; Accept moves it to the left as the
   new reply to steer from; the pins are kept throughout. An architectural
   change (MEMORY, Engine requirements; logical-overview.md's invariants):
   edits gain a range of steps, the edit kernel edits every row its range
   holds, the cache is reused only to the first position whose tokens or
   edits in force differ, the engine keeps the edits of the run that wrote
   the cache, the page sends the kept reply tokens, a limit of 0 inspects,
   and each emitted token crosses with its identifier. Three parts, each
   reviewed: (a) headers and tests; (b) the engine and boundary
   code, to the edit specs, cache, turn, kernels and replay suites;
   (c) the page: words, the branch point, the deck at it, Run
   forward, Accept, with the Branch suite and the browser scenario.
   The left column is "Reply" and the right "Steered reply", in
   the page and the scenario, since after an accept the left is no longer
   the original. A fix after review: Stop beside Run
   (client cancel, session.stop) and the running token count, since a
   steered reply that ran to the context was a dead end; the browser
   scenario's step 7 stops one and accepts it. Rollback: (b)
   adds to generate and keeps its old runs as k = 0 with edits from step
   0; reverting (c) leaves (b) unused.
5. **More captures in the engine** (capture, the norm and attention
   kernels, graph, engine), built, awaiting review of the code. So a view can show
   which layer did the work at a word: at the point, every layer's
   residual coming in, what its attention and feed-forward blocks add
   into it, and its attention weights over the words before, beside the
   neurons (the user, 2026-11-01). The norm kernel writes the residual and
   what a block adds, which exist only in its registers; the attention
   kernel writes the last row's scores and its final maximum and sum; a
   new launch (`capture/weights.h`) turns them into weights (weights
   over raw scores, the user, 2026-11-02; MEMORY's one exception). An
   architectural change: two ported kernels gain outputs, the store's
   contract changes (every word written in the step that ends the
   prompt, 3.2 MiB for Gemma 3 1B, from 702 KiB), and the step's `logits`
   is read by kernels; 26 more dispatches a decode step, an estimate. The
   neurons cross to the page as before; the rest is read back and does
   not cross yet. Q, planned with the pass map, is not captured. Tests:
   in the kernels suite, the norm's, the attention's and the weights
   launch's captures, and the layout case changed with the contract; the
   replay scenario's case 7 compares the whole store. The oracle and the
   replay scenario run, since two inference kernels change. Rollback:
   revert the step; the store holds the neurons alone again, and nothing
   outside the engine read the rest.
6. **The other captures reach the page, and the Map** (bindings, worker,
   `protocol/`, session, views). The store crosses whole; the Map
   (docs/design/concepts/explore.html) shows every layer at the point,
   on one scale across layers. Its design waits on the concept's review.

## 5. Pass map

- The isometric layer in three.js, stages at their real widths, layer
  stepping, clicking into a workbench.
- Captures gain Q at the last position. (The residual and the attention
  weights are captured from slice 4, step 5.)

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
- The tokens a neuron pushes toward, on the neuron card: computed from the
  weights rather than produced by the model, and approximate in Gemma 3.
- Measuring decode speed in the browser, edits off and on, if it comes to
  seem slow; until then MEMORY's figures are estimates.
- Measuring the module's heap peak, if a load comes to fail or strain for
  memory; until then its 128 MiB start and 1 GiB limit are estimates.
- Measuring Gemma 3 1B's sampling settings; until then it runs on the
  defaults for an unmeasured model (policy.h).
- Which libc++ the native build links on macOS: today the system's
  `/usr/lib/libc++.1.dylib`, while compiling against LLVM 21.1.8's headers.
- Hosting and CI.
- The browser scenario hung once (2026-11-02): the page's WebGPU status
  never settled, before the model loaded; not reproduced in 88 tries after.
  If it hangs again, the test is rethought, not retried (the user,
  2026-11-02).
