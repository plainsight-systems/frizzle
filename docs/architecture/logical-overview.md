# Logical overview

Draft for review, 2026-10-27. What Frizzle is made of, what state it keeps,
and where values are read and changed. Decisions themselves are in
[MEMORY](../process/MEMORY.md).

**Legend:** **[P]** ported from Charlotte as it is · **[C]** ported, then
changed · **[N]** new. ◉ capture · ✎ edit.

## The parts

Arrows are labelled: what crosses, or who owns what.

```mermaid
flowchart TB
  subgraph Page["Page"]
    Site["Site · Hugo [N]"]
    App["App · React [N]<br/>prompt, replies<br/>pass map in three.js WebGL<br/>workbenches · the edit set"]
  end
  subgraph Worker["Engine worker"]
    Host["Worker host · TypeScript [C]<br/>download, SHA-256, OPFS<br/>chat template → text"]
    Wasm["Engine · C++ → wasm [C]<br/>tokenizer, weights, pass, cache<br/>sampler, captures, edits"]
  end
  File[("Pinned GGUF<br/>config/model.json")]
  GPU["WebGPU device"]
  subgraph Native["Native: analysis and tests only"]
    Tool["Native tool and tests [N]"]
    Dawn["Engine · C++ on Dawn [C]"]
  end
  Site -->|"hosts"| App
  App -->|"requests, each with an id"| Host
  Host -->|"replies (streamed), captures (copies)"| App
  File -->|"downloaded once, verified"| Host
  Host -->|"the operations"| Wasm
  Wasm -->|"owns"| GPU
  Tool -->|"the operations"| Dawn
```

One engine source, two builds, one `webgpu.h`. Everything the native tool and
the tests exercise is the same code the browser runs.

The worker host renders the chat template, so the page sends only the
prompt's text. Charlotte renders it on the page; it moves here to sit with
the file it is read from.

| Part | Owns | Offers | Does not own |
| --- | --- | --- | --- |
| Site | Pages, the family shell | A home for the built app | Anything in the model |
| App | The edit set, what is on screen, copies of captures | Requests | The device, the model, tokens |
| Worker host | The file: download, verify, cache; the chat template | The worker side of the protocol | The GPU |
| Engine | Weights on the GPU, the cache, the sampler | `load`, `generate` | The page; the file on disk |
| Native tool, tests | Fixtures, references | Nothing to the product | Anything the browser build does not also do |

## The operations

| Operation | Does | Returns |
| --- | --- | --- |
| `load(file)` | Read the GGUF, put the weights on the GPU | Ready, or a named refusal |
| `generate(text, edits, seed)` | Run the model over the rendered prompt, the edits applied at the last position and held at every generated token | The reply, streamed; captures (slice 2) |

Frizzle shows what the model produces and computes nothing beside it. The
cache keeps the prompt's prefix between runs, so running the same prompt
with other edits recomputes only the last position before decoding.

## The state: the cache in three parts

```mermaid
flowchart LR
  P["0 … n−2<br/><b>the prefix</b><br/>computed once per prompt<br/>never written again"]
  L["n−1<br/><b>the last position</b><br/>a function of the prefix<br/>and the edit set<br/>recomputed every run"]
  G["n …<br/><b>the reply</b><br/>appended while decoding<br/>cut away before the next run"]
  P --- L --- G
```

These are Frizzle's requirements on the cache. Cutting the reply away never
recomputes the prompt, whatever the reply's length. Charlotte's cache is where
the code starts, and it changes wherever it falls short.

## Inside one layer: where values are read and changed

Top to bottom is the order of work for position n−1.

```mermaid
flowchart TD
  N1["Norm that opens the layer<br/>adds the last block into X [P]"]
  X(["◉ residual<br/>last row of X [N copy]<br/>✎ residual, slice 7"])
  QKV["Q, K, V matmul [P]"]
  R["QK-norm, RoPE,<br/>cache write at n−1 [C]"]
  Q(["◉ Q, normed and rotated [N copy]"])
  A["Attention [P]"]
  AW(["◉ attention weights<br/>slice 5 [N output]"])
  O["Out projection [P]<br/>post-norm, add into X,<br/>FFN norm [P]"]
  GU["Gate · up matmul<br/>writes activation [P]"]
  E(["✎ neurons [N]<br/>then ◉ neurons [N copy]"])
  D["Down projection [P]<br/>added into X by the next layer's norm"]
  N1 --> X --> QKV --> R --> Q --> A --> AW --> O --> GU --> E --> D
```

Each capture is copied out at its point into a capture store with a slot
per layer, so it holds what the model used even if the buffer it came from
is reused. Every run sends the whole store to the page, so switching layers
never asks the engine for anything.

| Per layer, at f32 | Size |
| --- | --- |
| Residual, 1,152 | 4.5 KiB |
| Q, 4 heads × 256 | 4 KiB |
| Neurons, 6,912 | 27 KiB |
| Attention weights, 4 heads × positions (slice 5) | 16 B per position |
| **All 26 layers, 512-token prompt** | **about 1.1 MiB** |

That is a small fraction of one decode step's reads (the 722 MB of
weights). An edited stage is captured after its
edit: the screen shows what the model used. The unedited values are the
store from the run without edits, which the app keeps.

Editing the residual (slice 7) lands between the add and the normalization,
which are one kernel today. How to open that up is slice 7's design.

## The loop

```mermaid
sequenceDiagram
  participant You
  participant App
  participant Worker as Worker and engine
  You->>App: type a prompt
  App->>Worker: generate(text, no edits, seed)
  Worker-->>App: the original reply, streamed; captures
  You->>App: pin neurons
  App->>Worker: generate(text, whole edit set, same seed)
  Worker-->>App: the edited reply, beside the original; captures
```

The original reply is generated once per prompt, before any edit, and kept.

## What must stay true

1. The prefix, 0 … n−2, is computed once per prompt and never written again.
2. Position n−1 depends only on the prefix and the edit set, and is
   recomputed on every run.
3. With no edits, or Scale × 1, the reply is bit for bit the original.
4. The app sends the whole edit set with every run. Neither holds a change
   list for the other.
5. A capture is a copy taken inside the pass at its point. An edited stage is
   captured after its edit. The capture store always holds the last run's
   values, for every layer, and is sent whole.
6. The original and edited replies use the same seed. The sampler is keyed by
   seed and position, and edits are applied at every decode step, so the edits
   are the only difference.
7. Every request carries an id, and the page shows only the answer to its
   latest one.

## Not settled here

- Whether gate and up are shown separately (new kernel outputs).
- How the residual edit opens up the add-and-norm kernel (slice 7).
