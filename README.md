<p align="center">
  <img src="docs/assets/frizzle-readme.png" alt="An engraved night scene: a small vintage coach drives along a road through a row of glass panes, each a grid of points of light; on one pane a single point glows amber beside a brass dial that has just been turned." width="100%" />
</p>

# Frizzle

Frizzle runs a chat model in your browser and lets you reach inside it. Ask it
something, open any of its 26 layers, pin one or more neurons to values of
your choosing, and read the reply it writes instead, beside the one it would
have written.

The model is Gemma 3 1B, running entirely on your device through WebGPU, with
an engine in C++ compiled to WebAssembly. Nothing runs on a server.

**Status:** working, run locally; not yet hosted. The engine runs Gemma 3 1B
in the browser through WebGPU. The app writes a reply, opens a workbench of
every layer's MLP neurons, pins the ones you choose, and writes the steered
reply beside it from the same seed. Steering from any word of the reply is
next; the [work queue](docs/process/QUEUE.md) says what follows.

## Why Frizzle

Ms. Frizzle drives her class bus inside things to see how they work: a
beehive, a bloodstream, the solar system. "Take chances, make mistakes, get
messy!" That is the idea here. Travel inside a running language model, turn
something up by hand, and watch what it does to what the model says.

Frizzle follows [Charlotte](https://github.com/plainsight-systems/charlotte),
which runs models in the browser, and
[Dolittle](https://github.com/plainsight-systems/dolittle), which looks inside
their layers. Its Gemma 3 path starts from Charlotte's code. Frizzle is R&D
under Plainsight Systems LLC.

## Find your way around

| Path | What it holds |
| --- | --- |
| `src/` | The C++ engine, compiled to WebAssembly for the browser and natively for tests |
| `app/` | The interface: React, react-three-fiber, TypeScript and Vite |
| `web/` | The project site, which will host the app (not yet built) |
| `config/model.json` | The one pinned model file |
| `tests/` | The tests, and the inventory of what each one guards |
| `tools/`, `scripts/` | Reference and fixture generators; checks and pinned tooling |
| `docs/process/` | Current decisions, remaining work and the workflow |
| `docs/architecture/` | The design, written before the code it governs |
| `docs/research/` | Investigations, each dated |
| `docs/design/` | Interface and theme concepts |

## Working on Frizzle

Start with [project memory](docs/process/MEMORY.md), the
[work queue](docs/process/QUEUE.md) and the
[workflow](docs/process/workflow.md). Agent entry points are
[`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

## License

Code and configuration use Apache-2.0; documentation and original artwork use
CC BY 4.0. See [LICENSE](LICENSE) for scope and [NOTICE](NOTICE) for
attribution. Model weights keep their own terms.
