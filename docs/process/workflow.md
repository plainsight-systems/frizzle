# Workflow

Frizzle follows the workflow of Serenity, Dolittle and Bullet Witch, with one
deliberate difference: a much smaller, stricter test suite. This document is
self-contained; the repository imports no governance files.

## Design and implementation

1. Establish the logical architecture, its change axes and the file mapping
   in `docs/architecture/` before writing engine code.
2. For C++ header design, query both corpus MCP servers as below. Write each
   new or changed file's contract, algorithm and design in its opening
   comments, with the rules read. Review those headers before implementing.
3. For C++ implementation, query both servers again for the concrete work.
   Implement the change and its tests together, applying the guidance and
   keeping the header comments current.
4. Check correctness and cost, then update [MEMORY](MEMORY.md) and
   [QUEUE](QUEUE.md).

There are no implementation packets. Architecture documents describe the
system; file headers carry the design of the code they accompany.

## Required C++ corpus consultation

Header design and implementation must each query both local MCP servers:

- `cpp-guidelines`: the C++ Core Guidelines.
- `cpp-perf-guidelines`: the C++ performance corpus.

This applies to all new and changed C++, including code ported from Charlotte
when it is changed. Performance consultation is required in both phases.

Use each server's `search_guidelines` for the real concerns, then
`get_guideline` to read the rules used. Design queries cover contracts,
ownership, lifetimes, interfaces and expected cost; implementation queries
cover the concrete algorithms, layout, allocations, copies, dispatch and
synchronization. Cite the corpus and exact rule ID in the file's opening
comments with how it applies. Never invent IDs, cite unread rules, or
substitute memory or web advice. If a search finds no applicable rule, record
the topic and that result.

If either server is unavailable, say which and stop the affected C++ work.
Proceed without it only with the user's explicit agreement.

## File header comments

Each new or changed C++ or WGSL file begins with:

1. **Axis and responsibility:** its change axis and the work it owns.
2. **Contract:** inputs and outputs, preconditions and postconditions,
   invariants, ownership and lifetime, failure behavior, numerical
   requirements.
3. **Algorithm:** the part of the architecture this file performs, as ordered
   steps from inputs to outputs, with dimensions defined and rounding points
   stated. A reader should follow it without opening the linked document.
4. **Design and cost:** consequential choices, allocations and copies,
   dispatches and readbacks, planned versus measured costs.
5. **Corpus guidance:** the rules read, with IDs and their application.

Ported files keep Charlotte's headers until they are changed; a changed file
is brought up to this standard. A mechanical rename is not a change for this
rule: on porting, the namespace `bllm` becomes `frizzle` and the macro prefix
`BLLM_` becomes `FRIZZLE_`, with nothing else touched. A ported file whose
header names no change axis gains that line when it next changes for another
reason; until then the [file mapping](../architecture/file-mapping.md) is the
record of its axis.

## Performance design

- Count the cost before building a data path: bytes, copies, buffer writes,
  dispatches, readbacks and synchronization, in model dimensions and context
  length. Measurement calibrates the count. In the browser every dispatch,
  `writeBuffer` and readback has a cost of its own (Charlotte measured about
  1.5 µs per dispatch and about 0.5 ms per readback, on one machine).
- Design optimizations in, marked `Optimization:` with their reason.
- Captures must not slow the normal path when they are off. Edits may, by a
  stated fixed cost (MEMORY, Engine requirements). Measure decode speed with
  each off and on.
- Record the browser, its version, the machine and the workload with every
  number. A figure from a desktop app's built-in browser is not a Chrome
  figure.

## Tests

This is where Frizzle departs from its siblings. Many small unit tests make a
codebase brittle: every change becomes a test change, and the suite ends up
describing the implementation rather than guarding the behavior. Frizzle keeps
a short list of tests that each mean something.

**What earns a test.** A test guards a named contract that someone would
notice breaking: a visible behavior (the model's output, an edit's effect, a
refused input) or an invariant the design depends on (exact replay, the cache
untouched behind the last position). It does not test private helpers,
incidental structure, call sequences, or how the code currently happens to be
written.

**Refactors never change tests.** A behavior-preserving change that forces a
test change means the test was asserting implementation: narrow it to its
contract or delete it. A deliberate change of contract is the only reason to
change a test, and the commit says so.

**The inventory is the suite.** [`tests/README.md`](../../tests/README.md)
lists every test and the contract it guards. Adding a test is a decision
recorded there, with its contract; removing one is equally deliberate.
Integration tests are capped at five across the repository. Do not hide
several workflows in one test or relabel an integration test to dodge the cap.

**Ported tests are not inherited.** A test from Charlotte comes over only if
it guards a contract on the inventory.

**Reference first.** Numerical correctness is judged against an independent
reference, llama.cpp's logits for the same file and tokens, never against the
engine's own earlier output. Freeze the reference, its inputs and its
tolerance before investigating a failure; do not adjust the reference or relax
the tolerance to make a failure pass. Exact-replay checks are bit for bit, with
no tolerance. A tolerance comparison fails on NaN.

**Unit and integration.** Contract tests run in process on small in-memory or
fixture inputs, with no builds, servers, browsers, downloads or full model
runs. Anything that loads the real model, drives the GPU through the engine,
builds the site or opens a browser is an integration test and counts toward
the five.

**The oracle runs when inference changes.** The comparison with llama.cpp is
an integration test, not part of the routine suite. Run it when the inference
path changes: the GGUF reader, formats, residency and upload, graph, kernels,
program, cache, sampler or turn loop.

## Checks

- **Formatting:** the pinned Google `.clang-format`, through
  `python3 scripts/check.py format` (`--fix` applies it).
- **Static analysis:** `python3 scripts/check.py tidy` runs the pinned
  clang-tidy over every entry in the native build's compilation database;
  every finding fails. Fix findings rather than excluding checks; any narrow
  exception is justified beside the code. Code that exists only in the
  browser build is not covered, and is reviewed instead. Files ported from
  Charlotte and unchanged since porting (their SHA-256 still matches
  `src/ported.json`) are not analyzed, as translation units or as headers,
  and every run lists them; a ported file that changes is analyzed from then
  on, and every finding in it is fixed.
- **Tests:** `python3 scripts/check.py test` builds and runs the contract
  suites on the native build. The oracle runs by hand when the inference
  path changes ([inventory](../../tests/README.md)); the end-to-end scenario
  runs in a browser.
- A build passing, or the page loading, is not evidence of correctness.
  Report the checks actually run.

## Records

Current engineering decisions go in [MEMORY](MEMORY.md) and remaining work in
[QUEUE](QUEUE.md); both are public and describe the present, not a history.
Conversation summaries, approvals, implementation diaries, build logs and
reports go in ignored `.local/`, and tracked documents do not link to them.
`docs/research/` holds substantive investigations, each dated with its method
and conditions. `docs/writeups/` is written only when publication is asked for.

Reduce scope rather than lower quality within it. Record intentional changes
to this workflow here.
