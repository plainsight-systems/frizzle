# File mapping

Draft for review, 2026-10-28. How the [change axes](change-axes.md) become
directories and files. Files ported whole are mapped by directory and listed
in the [inventory](../research/2026-10-27-charlotte-inventory.md); files that
change or are new are mapped one by one.

**Legend:** **[P]** ported from Charlotte as it is · **[C]** ported, then
changed · **[N]** new.

The axis belongs to the file. A directory holds one axis plus, where there is
one, that family's contract header. `src/core/arch/` is the exception: its
contract, `describe` (C) and `gemma3/` (A) sit together because each names the
others.

## The repository

An arrow reads "imports, includes or loads". Every boundary that matters is an
arrow that is not there; the rules below say which.

```mermaid
flowchart LR
  subgraph app["app/src · TypeScript"]
    direction TB
    frame["root · H(frame)"]
    views["views/ · H(view)"]
    theme["theme/ · H(theme)"]
    session["session/ · H(session)"]
    shape["shape/ · H(shape)"]
    mirrors["engine/ · O, Q mirrors"]
    protocol["protocol/ · R"]
    worker["worker/ · R, L, T"]
  end
  subgraph src["src · C++"]
    direction TB
    wasm["wasm/ · R"]
    core["core/ · A–G, I–K, M, O, Q"]
  end
  subgraph native["tools · tests"]
    direction TB
    tools["tools/ · V"]
    tests["tests/ · V"]
  end
  config[("config/model.json · J")]
  frame --> views
  frame --> theme
  views --> session
  views --> shape
  views --> theme
  theme --> shape
  session --> mirrors
  session --> protocol
  protocol --> mirrors
  worker --> protocol
  worker -->|"the built module"| wasm
  worker --> config
  wasm --> core
  tools --> core
  tests --> core
```

| Rule | Why |
| --- | --- |
| `src/core/` includes nothing from `src/wasm/`, `tools/` or `tests/` | The core never knows it is in a browser, so it builds and tests natively |
| React is imported only by the root, `views/` and `theme/` | Swapping React touches H only ([not an axis](change-axes.md#not-axes)) |
| three.js and react-three-fiber are imported only by `views/` | The scene is a view's; `theme/` hands it colours as plain numbers, and `shape/` stays testable in Node |
| `session/`, `shape/`, `engine/`, `protocol/` and `worker/` are plain TypeScript | Tested without React; the worker pulls no React into its bundle |
| `worker/` imports nothing from `views/`, `session/`, `shape/` or `theme/` | The worker is the engine's side; the page talks to it only through `protocol/` |

`python3 scripts/check.py boundaries` will enforce them, the C++ rule from
slice 1 and the TypeScript rules from slice 3.

## The engine, `src/`

| Path | Axis | Origin | What it holds |
| --- | --- | --- | --- |
| `core/arch/`, `core/model/` | A, C, contract | [P] | Gemma 3's description and the order its graph runs the blocks |
| `core/graph/graph.*` | A | [C] | The layer's blocks; the ◉ and ✎ points |
| `core/formats/<format>/` | B | [P] | F32, F16, Q4_0, Q4_1, Q8_0 |
| `core/gguf/` | C | [P] | The file reader |
| `core/gpu/`, `core/residency/`, `core/kernels/program`, `order`, `schedule`, `core/run_guard.h` | D | [P] | Device, placement, upload, launching; the context offered is sized in `residency/plan` |
| `core/kernels/<kernel>/` | E | [P] | gather, matmul, norm, rope, attention, top-k |
| `core/kernels/attention/` | E | [C] | Writing the last row's scores and their maximum and sum into the capture store (slice 4, step 5) |
| `core/kernels/norm/` | E | [C] | Writing the residual and what each block adds into the capture store (slice 4, step 5); opening up the add and the norm, if slice 7 needs it |
| `core/sampler/` | F | [P] | Keyed by seed and position |
| `core/cache/kv.*` | G | [C] | The three-part cache in the [overview](logical-overview.md#the-state-the-cache-in-three-parts) |
| `core/cache/prefix.*` | G | [P] | The prefix diff |
| `core/capability/capability.cpp` | I | [C] | Trimmed to Gemma 3 and its formats |
| `core/policy/` | contract | [P] | What can be configured. The values are J's: they come from `config/model.json`, which gains the measured settings once measured |
| `core/tokenizer/` | K | [P] | SentencePiece BPE |
| `core/runtime/runtime.*`, `turn.*`, `stops.*` | M | [P] | The turn loop |
| `core/runtime/generator.*` | M | [P] | Text in, text out over the turn loop |
| `core/runtime/engine.*` | M | [N] | `load` and `generate`, the operations every host calls |
| `core/capture/` | O | [N] | The store, its slots a layer and kind, its layout (`capture.h`); `capture.wgsl`, the neurons' copy, and its launcher; `weights.wgsl` and its launcher, the attention scores into weights (`weights.h`); the reader that brings the store back once a run (`reader.h`) |
| `core/edit/` | Q | [N] | Edit kinds and what each does as a plain function; `edit.wgsl` and its launcher |
| `core/diagnostics.h` | U | [P] | The build's instrumentation gate |
| `ported.json` | V | [N] | Each ported file's SHA-256 as ported, written by `scripts/port_charlotte.py`; tells unchanged files from changed ones |
| `wasm/bindings.cpp` | R | [N] | The four operations for the worker; the only Emscripten-aware C++ |
| `shaders/vector_add.wgsl` | D | [P] | The device self-check |

Not ported: `preflight/`. Its one caller is Charlotte's `bindings.cpp`, which
is rewritten, and the context offered is already sized in `residency/plan`.

Q's kernel sits in `core/edit/`, not `core/kernels/`: it is Q's rule written
in WGSL, and adding an edit kind changes both files.

## The app, `app/src/`

```text
main.tsx, App.tsx      H(frame)
views/prompt/          H(view)   prompt, both replies
views/pass-map/        H(view)   slice 5
views/deck/            H(view)   the MLP workbench (slice 4): Workbench.tsx, Deck.tsx
                                 (the layer deck), Inspector.tsx, cards.ts
views/residual/        H(view)   slice 7
theme/                 H(theme)  night glass: theme.css (tokens, the page's elements),
                                 palette.ts (the deck's colours), fonts/ (Plex, kept)
session/               H(session) session.ts (the page's state), pins.ts (the pin set:
                                 adding, removing, grouping, what a pin sends)
shape/                 H(shape)  fold.ts (a slot into the deck's grid, a box of cells
                                 back to neurons), paint.ts (a slot into pixels, a heat
                                 map of size, and overlays), rank.ts (the
                                 strongest), numbers.ts (MB, values for reading)
engine/captures.ts     O         the store's layout, read from the wire
engine/edits.ts        Q         the edit kinds, as sent
protocol/              R         messages, ids, the page's client, starting the worker
worker/worker.ts       R         the worker's entry; answers the page's requests
worker/engine/         R         the engine's module, behind the worker's calls
worker/delivery/       L         fetch, SHA-256, OPFS, streaming into the engine
worker/template/       T         Gemma's chat template; @huggingface/jinja from npm
```

Two Vite entries, the page and the worker. `worker/delivery/` reads
`config/model.json`.

## Build, tools and tests

| Path | Axis | What it holds |
| --- | --- | --- |
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/` | U | Native and browser presets, shader embedding, the Dawn pin |
| `app/package.json`, lockfile, Vite config | U | Exact npm versions |
| `tools/frizzle/` | V | The native tool: the reply to a prompt; the edited reply beside it from slice 2 |
| `tools/reference_logits/` | V | Makes llama.cpp's logits, the frozen reference, again; its own build against llama.cpp |
| `scripts/` | U, V | Python: checks, fetching Dawn and the pinned model, the mechanical port, the test fixtures |
| `tests/integration/<scenario>/` | V | One directory per scenario in the [inventory](../../tests/README.md): `reference-logits`, `replay`, `site` |
| `tests/contracts/<suite>/` | V | One per suite: `tokenizer`, `gguf`, `kernels`, `edits` |
| `tests/fixtures/` | V | Small inputs and frozen references |
| `tests/support/` | V | What the tests share: reading files, driving the GPU, doctest's main |
| `web/` | S | Hugo; mapped in slice 6 |

Charlotte's tests come over only where one implements an inventory row.

## Names and axis lines

- All C++ is in namespace `frizzle`, macros prefixed `FRIZZLE_`. Ported files
  are renamed mechanically on porting, which the
  [instructions](../../AGENTS.md#design-in-the-files) do not count as a
  change.
- 82 of the 128 ported files (112 C++, 16 WGSL) name no axis in their
  header. 21 of them are headers; most of the rest are `.cpp` files whose
  header carries it.
  This mapping is their record; each gains its line when it next changes for
  another reason.

## Not settled

- **`platform.js` and `platform_policy.js`,** inventoried for porting: which
  browsers Frizzle runs in and the memory budget. Their home waits on the open
  decision about supported browsers and devices.
- **Where the Emscripten build lands** for Vite to load (U, slice 3).
