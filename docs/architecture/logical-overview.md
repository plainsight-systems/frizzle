# Logical overview

What Frizzle is made of, what state it keeps, and where values are read and
changed. ◉ capture · ✎ edit.

## The parts

Arrows say what crosses, or who owns what.

```mermaid
flowchart TB
  subgraph Page["Page"]
    App["App · React<br/>prompt, replies<br/>the workbench · the edit set"]
  end
  subgraph Worker["Engine worker"]
    Host["Worker host · TypeScript<br/>download, SHA-256, OPFS<br/>chat template → text"]
    Wasm["Engine · C++ → wasm<br/>tokenizer, weights, pass, cache<br/>sampler, captures, edits"]
  end
  File[("Pinned GGUF<br/>config/model.json")]
  GPU["WebGPU device"]
  subgraph Native["Native: analysis and tests only"]
    Tool["Native tool and tests"]
    Dawn["Engine · C++ on Dawn"]
  end
  App -->|"requests, each with an id"| Host
  Host -->|"replies (streamed), captures (copies)"| App
  File -->|"downloaded once, verified"| Host
  Host -->|"the operations"| Wasm
  Wasm -->|"owns"| GPU
  Tool -->|"the operations"| Dawn
```

One engine source, two builds, one `webgpu.h`. Everything the native tool and
the tests exercise is the same code the browser runs.

| Part | Owns | Offers | Does not own |
| --- | --- | --- | --- |
| App | The edit set, what is on screen, copies of captures | Requests | The device, the model, tokens |
| Worker host | The file: download, verify, keep; the chat template | The worker side of the protocol | The GPU |
| Engine | Weights on the GPU, the cache, the sampler, the capture store | `load`, `generate` | The page; the file on disk; the chat template |
| Native tool, tests | Fixtures, the frozen reference | Nothing to the product | Anything the browser build does not also do |

The worker renders the chat template, so the page sends only a message's
text and the engine reads text already rendered. The template writes Gemma
3's `<bos>` and turn markers; the engine checks none of it, and text wrapped
otherwise runs and gets whatever reply the model makes of it. The native
tool wraps one user message by hand.

## The model

One file, pinned by size and SHA-256 in `config/model.json`; a download that
does not match is refused.

| | Gemma 3 1B, instruction-tuned, Q4_0 GGUF |
| --- | --- |
| File | 721,918,496 bytes |
| Layers | 26 |
| Residual width | 1,152 |
| MLP width (neurons) | 6,912 |
| Attention | 4 query heads sharing 1 key/value head, 256 wide |
| Attention span | Local, 512 keys, in five of every six layers; global in every sixth |
| Trained context | 32,768 tokens |
| Weight formats | Q4_0 attention, gate and up; Q4_1 for three `ffn_down`; F32 norms |
| Token embedding | Q8_0, about 321 MB, also the output head |
| KV cache | F16 |
| Decode speed in the browser | About 140 tok/s, estimated, not measured |

## The operations

| Operation | Does | Returns |
| --- | --- | --- |
| `load`, in three calls: `begin_load`, `write`, `finish_load` | Read the GGUF's index and tokenizer, plan residency within the device's limits, take the file's bytes in order, build the graph | Ready, or a named refusal |
| `generate(text, kept, edits, policy)` | Run the model over the rendered text and the reply tokens a branch keeps, each edit at the steps its range holds; a limit of 0 inspects, drawing nothing | The reply, streamed a token at a time with its identifier; then its end, its identifiers, and every layer's captures at step k |

Step k is the position that draws reply token k: the input's last position.
A run from the reply's start is k = 0, the prompt's last position.

The page sets no limit on a reply's length: a reply ends when the model ends
its turn, the context is full, or Stop cancels it. A strong edit can make
the model repeat itself to the end of the context.

## Loading

```mermaid
flowchart LR
  Pin[("config/model.json<br/>URL, size, SHA-256")]
  Kept{"OPFS copy<br/>with a matching record?"}
  Fetch["fetch in chunks<br/>SHA-256 as it streams"]
  Keep["keep in OPFS,<br/>then write the record"]
  Refuse(["refused by name"])
  Stream["stream into the engine<br/>16 MiB a write"]
  Ready(["ready"])
  Pin --> Kept
  Kept -->|"no"| Fetch
  Fetch -->|"size and hash match"| Keep --> Stream
  Fetch -->|"mismatch"| Refuse
  Kept -->|"yes: trusted by its record,<br/>not hashed again"| Stream
  Stream -->|"begin_load, write …, finish_load"| Ready
```

- The engine runs in a plain Web Worker as one ES module (`frizzle.mjs` and
  its `.wasm`), with no SharedArrayBuffer, so the page needs no cross-origin
  isolation.
- A load can be cancelled while it downloads. Once it streams into the
  engine it runs to its end, a few seconds: the engine cannot abandon a load
  half made. A generation can be cancelled at any time.
- The module's heap starts at 128 MiB and may grow to 1 GiB. Both are
  estimates, not measured.

## The state: the cache in three parts

```mermaid
flowchart LR
  P["0 … n−2<br/><b>the prefix</b><br/>computed once per prompt<br/>never written again"]
  L["n−1<br/><b>the last position</b><br/>a function of the prefix<br/>and the edit set<br/>recomputed every run"]
  G["n …<br/><b>the reply</b><br/>appended while decoding<br/>cut away before the next run"]
  P --- L --- G
```

Cutting the reply away never recomputes the prompt, whatever the reply's
length. The cache keeps edited entries with the edits of the run that wrote
them; a run reuses it only up to the first position whose tokens or edits in
force differ (invariant 1).

## Inside one layer: where values are read and changed

Top to bottom is the order of work at step k.

```mermaid
flowchart TD
  N1["Norm that opens the layer<br/>adds the last block into X"]
  MA(["◉ what layer l−1's feed-forward added<br/>written by the norm"])
  X(["◉ residual, the last row of X<br/>written by the norm"])
  QKV["Q, K, V matmul"]
  R["QK-norm, RoPE,<br/>cache write"]
  A["Attention"]
  AW(["◉ scores, maximum and sum<br/>written by attention;<br/>the weights launch turns them into weights"])
  O["Out projection<br/>post-norm, add into X,<br/>FFN norm"]
  AA(["◉ what the attention added<br/>written by the norm"])
  GU["Gate · up matmul<br/>writes the neurons"]
  E(["✎ neurons<br/>then ◉ neurons, copied"])
  D["Down projection<br/>added into X by the next layer's norm<br/>(the last layer's by the final norm)"]
  N1 --> MA --> X --> QKV --> R --> A --> AW --> O --> AA --> GU --> E --> D
```

Each capture is written at its point into the capture store, a slot per
layer and kind, so it holds what the model used even if the buffer it came
from is reused (`capture/capture.h`).

| Captured value | Written by | Why there |
| --- | --- | --- |
| Neurons | A copy launch of its own, after the edit | An edited stage is captured after its edit: the screen shows what the model used |
| Residual, what each block adds | The norm kernel | The added vector exists only in its registers |
| Attention scores, maximum and sum | The attention kernel, for the last row | Only it has the scores |
| Attention weights | A launch after attention (`capture/weights.h`) | The raw scores, in log2 units, cannot be read; the weights differ from the factors the model applied by rounding only |

The attention weights are the one value formed beside the model's own
outputs, on the GPU in the same step, from the attention kernel's scores.

| Per layer, at f32 | Size |
| --- | --- |
| Neurons, 6,912 | 27 KiB |
| Residual, 1,152 | 4.5 KiB |
| What the attention adds, 1,152 | 4.5 KiB |
| What the feed-forward adds, 1,152 | 4.5 KiB |
| Attention weights, 4 heads × keys | 8 KiB for a 512-key window; 512 KiB for a global layer, sized to the trained 32,768 |
| Each head's softmax maximum and sum | 32 B |
| **The store, 26 layers** | **3.2 MiB, 2 MiB of it the 4 global layers' weights** |

That is a small fraction of one decode step's reads, the 722 MB of weights.
The unedited values are the store from the run without edits, which the app
keeps.

## Edits

| | |
| --- | --- |
| What | MLP neurons: Set to a value, Scale ×, Zero |
| How many | A set, any number of neurons in any layers, applied together in one pass |
| Where | Inside the pass, on the neurons' row, at every step its range holds |
| When | From the branch point k on, held at every generated token |
| Cost when off | The edit launch runs once a layer in every step, over every row, edits or not, and edits only the rows whose position a range holds: about 40 µs a step, estimated, not measured |

## A run's captures: when the store is written and read

The store is written once a run, inside the step that ends the prompt, and
read once, after the turn ends. The copy and weights launches run in every
step that asks for logits; they tell the step that ends the prompt from a
decode step by whether the step's token was fed from the last draw. The norm
and attention run in every step, so they also read whether the step asks for
logits (`capture/capture.h`). The diagram follows the neurons; the other
captures are written in the same step.

```mermaid
flowchart TD
  G["generate(text, kept, edits)<br/>edit table written once"]
  P{"next step"}
  PF["prefill block, not the last<br/>logits 0"]
  PE["the step that ends the prompt<br/>logits 1, tokens given (fed 0)"]
  DE["decode step<br/>logits 1, fed 1"]
  SKIP["edit and capture launches<br/>not dispatched"]
  subgraph L["each of the 26 layers, in order"]
    E["ffn.edit<br/>pins applied to the last row"]
    C["ffn.capture<br/>last row → the layer's slot"]
    D["ffn.down"]
    E --> C --> D
  end
  EXIT["ffn.edit holds the pins;<br/>ffn.capture reads fed, stops"]
  END{"how the turn ended"}
  READ["Reader::read<br/>one copy, one mapping, 3.2 MiB"]
  T(["on_end: reply's end + Taken captures"])
  N(["on_end: reply's end + NotTaken"])
  U(["on_end: reply's end + Unread, why"])

  G --> P
  P --> PF --> SKIP --> P
  P --> PE --> L --> P
  P --> DE --> EXIT --> P
  P -->|"turn over"| END
  END -->|"Stop, Limit, Context"| READ
  END -->|"Cancelled or failed:<br/>the store may hold<br/>an earlier run's"| N
  READ -->|"mapped"| T
  READ -->|"mapping failed"| U
```

- The store is never cleared. Only a turn whose step that ends the prompt
  ran is read, and that step writes every word of it, so the page is never
  shown an earlier run's captures as this run's.
- The turn's own result is reported as it was in every case: a capture never
  turns a reply into a failure, or a failure into a reply.
- Cost, estimated and not measured: 52 dispatches a decode step (the copy's
  and the weights') that read the step and stop, about 80 µs a token; once a
  run, 3.2 MiB copied on the GPU and one readback.

How a run's neurons reach the page, once a run, with its answer. The rest of
the store is read back with them and stays in the engine.

```mermaid
flowchart LR
  S["engine<br/>the store read back<br/>(capture/reader.h)"]
  B["bindings.cpp<br/>answer JSON with slots;<br/>neurons sliced out of the heap,<br/>same call"]
  E["engine.ts<br/>captures checked:<br/>taken fills its slots,<br/>or fails by name"]
  W["worker.ts<br/>done, the neurons'<br/>buffer transferred"]
  C["client.ts<br/>passed through<br/>untouched"]
  P["session and page<br/>a pin's value looked up:<br/>slot.first + neuron"]
  S -->|"span, valid in the callback"| B -->|"one copy"| E --> W -->|"no copy"| C --> P
```

One copy in all, out of the module's memory: the worker hands the buffer to
the page, and the page reads it where it lies. All 26 layers' neurons, 702
KiB, cross on every run, so switching layers needs no request.

## The loop

```mermaid
sequenceDiagram
  participant You
  participant App
  participant Worker as Worker and engine
  You->>App: type a prompt
  App->>Worker: generate(text, nothing kept, no edits, seed)
  Worker-->>App: the reply, streamed, each token with its identifier, then its captures
  You->>App: click a word: the branch point k
  App->>Worker: generate(text, the reply's first k tokens, its history, maxTokens 0)
  Worker-->>App: the captures at step k (an inspection)
  You->>App: pin neurons, Run
  App->>Worker: generate(text, the first k tokens, history cut at k + pins from k, same seed)
  Worker-->>App: the steered reply, beside the reply, then its captures at k
  You->>App: Accept: the steered reply becomes the reply, its edits its history
```

The reply is generated once per prompt and seed, before any edit; after
that every run keeps its tokens to the branch point, and an accepted steered
reply takes its place. Requests run one at a time, since the engine runs one
turn at a time. [Interface](interface.md) has what the page shows at each
point.

## What must stay true

1. A position's cache entry is a function of the tokens up to it and the
   edits in force at it. A run reuses the cache only up to the first
   position where either differs from the run that wrote it, and computes
   the rest; the prompt is never computed again while they agree.
2. The branch position, step k, the input's last, depends only on what
   comes before it and the edits in force there, and is recomputed on every
   run. For a run from the reply's start, k = 0, it is position n−1.
3. With no edits, or Scale × 1, the reply is bit for bit the original; a
   branch with a reply's own edits continues that reply bit for bit.
4. The app sends the whole edit set with every run, and the reply tokens a
   branch keeps. Neither holds a change list for the other.
5. A capture is a copy taken inside the pass at its point, step k. An edited
   stage is captured after its edit. The capture store always holds the
   last run's values, for every layer, and is read back whole.
6. Every run of one reply uses its seed. The sampler is keyed by seed and
   position, and edits apply at exactly the positions their ranges hold, so
   the edits are the only difference between a reply and a branch from it.
7. Every request carries an id, and the page shows only the answer to its
   latest one.
8. An accepted reply is its tokens and the edits it was generated with,
   each over its range; the page holds both, and recomputing it from them
   gives it again.

## Pinned versions

Every version that decides behavior is pinned in one place. Tools are
installed by hand; the scripts refuse another LLVM or Node and install
nothing.

| What | Where it is pinned |
| --- | --- |
| The model file | `config/model.json` |
| Browser build: Emscripten in the `emscripten/emsdk` container, with its `emdawnwebgpu` port | `CMakeLists.txt`, `scripts/tooling.py` |
| Native build: LLVM clang, against prebuilt Dawn, the release the browser port comes from | `cmake/dawn.json`, `scripts/tooling.py` |
| clang-format and clang-tidy, the native build's LLVM | `scripts/tooling.py` |
| doctest, fetched by CMake by commit | `tests/CMakeLists.txt` |
| Node, refused otherwise | `app/.nvmrc` |
| npm packages, exact versions, installed with `npm ci` and no install scripts | `app/package.json`, `app/package-lock.json`, `app/.npmrc` |
| IBM Plex fonts, kept in the app by SHA-256 | `app/src/theme/fonts/README.md` |

Native builds run on macOS arm64 and Linux x86-64. Both builds compile
against one `webgpu.h`, so moving the Dawn pin or the Emscripten pin moves
both.
