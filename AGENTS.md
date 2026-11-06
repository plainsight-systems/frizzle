# Working on Frizzle

## Purpose

Frizzle is a chat model in the browser that you can reach inside. Type a
prompt and the model writes its reply. Open any layer, change something at
the last position, and run the model again: the reply it writes with the
edit held is shown beside the original, both from the same seed. The
audience is curious people, not interpretability researchers.

- Frizzle shows what the model produces and computes nothing beside it. A
  value the model makes that cannot be read as it is goes to the user as a
  choice; it is not converted on our own initiative.
- Everything shown and editable belongs to the last position. Earlier
  tokens are context. Inspecting or editing earlier positions is out of
  scope.
- The model runs in the browser only, on WebGPU. Native builds exist for
  analysis and tests, never as a product.
- A tech demo is a complete implementation of a narrow scope. Reduce scope
  rather than lowering correctness within it.

Read [docs/architecture/README.md](docs/architecture/README.md) and the
documents it lists for the parts you touch.

- Code from Charlotte comes over only as listed in
  `docs/research/2026-10-27-charlotte-inventory.md`, and `src/ported.json`
  records each ported file as it came. Anything else copied from another
  project goes on that record or a dated note like it.
- Keep `app/` (the interface) separate from `web/` (the site that hosts it).
- The pinned model and everything downloaded or generated stay out of Git.
  So do machine-specific paths and credentials.
- Do not import governance documents, submodules or symlinks from other
  projects. This file is the one instruction set.
- Keep proposed design, implemented behavior and verified behavior distinct
  in what you write.
- Do not commit or push without the user's word.

## Architecture documents

`docs/architecture/` describes the system as it is. Its README lists each
document and what it covers in full.

- Visuals first: a diagram where a reader needs to see what moves where, a
  table where they compare, prose only for what neither can say. Short.
- Content is invariants and flows: what is always true, what moves where,
  who owns what, in the model's dimensions and human units.
- No plans, phases, steps, slices, rollback, verification lists,
  alternatives considered or decision records. Git history is the record;
  plans live untracked under `research/`.
- No status words. A document that says what is done, pending or next is a
  plan, not architecture.
- No outside sources (see Sources and citations).

## Recurring mistakes

Errors an agent or the user has caught more than once. Read them as rules.
Add an entry when the same mistake comes back.

- **Computing beside the model.** A softmax over read-back logits, a derived
  score, a "prediction": anything the running model did not produce drifts
  from it and has to be guarded. Show the model's own values. The one
  exception is attention weights, formed on the GPU in the same step from
  the attention kernel's own scores (`src/core/capture/weights.h`).
- **Inheriting choices as requirements.** A sibling project's choices and
  ported code's structure are starting points, not platform facts and not
  requirements. State what Frizzle needs, then change ported code where it
  falls short. Label anything taken from elsewhere as a verified platform
  fact or as that project's choice.
- **Narrowing scope unasked.** Offer options; the user sets the limits.
- **Design only in chat, or in a separate document.** A step's design is the
  opening comments of the files it creates or changes, in any language.
- **Steps too small.** A step is one coherent capability with its tests, not
  one file per review.
- **Code moved to suit a test.** A test that cannot reach a unit has exposed
  a boundary; split by the boundary, never relocate code to make a test
  compile.
- **An integration test that overclaims.** Before writing one, state what it
  checks that nothing else does and which bugs would still pass it. Assert
  on the value itself (a captured activation), not a downstream effect (the
  text changed).
- **A page below the design bar.** Human units (MB, two places), graphical
  legends with rounded ticks, Dolittle's layout (status in the header, Run
  beside the prompt, settings folded away), selection on the visual itself,
  and the look of the approved concepts in `docs/design/concepts/`. No step
  ships a page worse than the concept.
- **A fix committed apart from its feature.** A fix to uncommitted work goes
  in that work's commit.

## Design in the files

Each new or changed C++, WGSL or TypeScript file opens with its design:

1. **Axis and responsibility:** its change axis and the work it owns.
2. **Contract:** inputs and outputs, preconditions and postconditions,
   invariants, ownership and lifetime, failure behavior, numerical
   requirements.
3. **Algorithm:** the part of the architecture this file performs, as
   ordered steps from inputs to outputs, with dimensions defined and
   rounding points stated. A reader follows it without opening another
   document.
4. **Design and cost:** consequential choices, allocations and copies,
   dispatches and readbacks, counted costs and measured ones, labelled as
   which.
5. **Corpus guidance** (C++ and WGSL): the rules read, by ID, and how each
   applies.

Ported files keep Charlotte's headers until they change; a changed file is
brought to this standard and gains its axis line. On porting, the namespace
`bllm` became `frizzle` and the macro prefix `BLLM_` became `FRIZZLE_`;
that rename is not a change for this rule.

**Corpus consultation.** Design and implementation of C++ and WGSL each
query both local MCP servers, `cpp-guidelines` and `cpp-perf-guidelines`:
`search_guidelines` for the real concerns, then `get_guideline` for each rule
used. Design queries cover contracts, ownership, lifetimes, interfaces and
expected cost; implementation queries cover algorithms, layout, allocations,
copies, dispatch and synchronization. Cite only rules read in this way,
never an invented or remembered ID. A search that finds no rule is recorded
as its topic and that result. If either server is unavailable, say which and
stop the affected work until the user agrees to go on without it.

**Cost.** Count the cost before building a data path: bytes, copies, buffer
writes, dispatches, readbacks and synchronization, in model dimensions and
context length. In the browser every dispatch, `writeBuffer` and readback
has a cost of its own. Optimizations are designed in and marked
`Optimization:` with their reason. Every measured number records the
browser, its version, the machine and the workload; a desktop app's built-in
browser is not Chrome.

## Sources and citations

A source is anything a design follows that is not in this repository:
engine source at a pinned revision, another project's code, a paper, a talk,
a vendor document. Citations to sources exist for review and do not ship.

These are part of Frizzle and are named freely:

- The corpus rule IDs in file headers.
- llama.cpp where it is named as the oracle: the reference logits, their
  generator and the test that compares against them, and the documents and
  scripts that say when that test runs.
- The record of the port: the Charlotte inventory and `src/ported.json`.
- How each frozen reference under `tests/fixtures/` was made: its generator
  and where its inputs came from, so it can be made again.

For every other source:

- The slice plan names each source it follows: a tree with its revision, or
  a paper, talk or document with its year and location. No tracked file
  lists sources.
- A pinned tree is read from a blob-less fetch of that revision alone, with
  no working tree, at the path under its key in `research/local.md`. Read one
  file at a time with
  `GIT_TERMINAL_PROMPT=0 git -C <checkout> show <revision>:<path>`, and
  `ls-tree` to list paths. Never check it out, fetch another ref or widen it;
  `git grep`, `git log -p` and `git diff` against it fetch blobs and are not
  run.
- While a slice is open, every type and function that follows a source
  carries `// Source: <source>, <revision or year>, <path or section>:<lines>`,
  and every step of the source left out has a note beside the declaration
  saying why Frizzle does not reach it. The reviewer reads the cited source,
  not the citation.
- When the user accepts the slice, before it is committed, every `Source:`
  citation and omission note leaves the tree. A comment that explains what
  the code does or why stays, in our words, naming no source; a platform
  fact is stated as the fact. A number measured outside Frizzle is stated as
  an estimate, not measured in Frizzle. The citations and omissions stay in
  the plan.
- Before every commit, `git grep -n -P '\bSource:' -- src tests tools app web
  cmake docs scripts` is empty, and so is a case-insensitive grep over the
  same paths, outside the files named above, for each source the slice plan
  lists.

Why: an agent designs from the code and the architecture documents in front
of it. A source named beside the code pulls the next change toward that
source where Frizzle would depart.

## Slices

Records live untracked under `research/` in the main checkout, the first
entry of `git worktree list`; a worktree has no `research/`.

- `research/plan.md` is the project plan: what Frizzle is for and the
  slices in order. The project session writes it and the slice plans.
- `research/slices/<slice>/plan.md` is one slice: its scope, research,
  guidelines read, steps, test list, ownership lists and omissions.
- `research/open.md` holds decisions not yet made.
- `research/local.md` holds this machine's paths: pinned source trees and
  anything else a brief needs. No absolute path of this machine goes in a
  tracked file.

Outside `research/`: `docs/research/` holds the dated investigations kept
public, such as the Charlotte inventory, each with its method and
conditions; ignored `.local/` holds tool output such as clang-tidy's logs.

A fresh session per slice, in its own worktree, runs `/architect <slice>`,
which uses the `researcher`, `implementer` and `reviewer` agents under
`.claude/agents/`.

Every step runs in this order: the files with their design and the step's
tests; the user reviews both; the code, until the tests pass; the user
reviews it; the next step. Tests are never a step of their own at the end.

## Testing

Frizzle keeps a short list of tests that each mean something. Many small
tests make a codebase brittle: every change becomes a test change, and the
suite ends up describing the implementation instead of guarding behavior.

- **What earns a test.** A test guards a named contract someone would
  notice breaking: a visible behavior (the model's output, an edit's effect,
  a refused input) or an invariant the design depends on (exact replay, the
  cache untouched behind the last position). Before writing or keeping one,
  name what it asserts: a header's contract or an architecture statement.
  Not private helpers, incidental structure, call sequences, or how the code
  happens to be written.
- **Refactors never change tests.** A behavior-preserving change that forces
  a test change means the test asserted implementation: narrow it to its
  contract or delete it. A changed contract is the only reason to change a
  test, and the change goes back to review.
- **The inventory is the suite.** [`tests/README.md`](tests/README.md) lists
  every test and the contract it guards. Adding or removing one is a
  decision recorded there. Integration tests are capped at five; do not hide
  several workflows in one test or relabel one to dodge the cap.
- **Unit and integration.** A unit test exercises one unit: a function, a
  module, or one kernel, on the GPU where the kernel needs it, in process, on
  small in-memory or fixture inputs. Anything that runs several units
  together (kernels back to back), loads the real model, drives the GPU
  through the engine, builds the site or opens a browser is integration and
  counts toward the five.
- **Reference first.** Numerical correctness is judged against llama.cpp's
  logits for the same file and tokens, never against the engine's own
  earlier output. Freeze the reference, its inputs and its tolerance before
  investigating a failure; never adjust the reference or relax the tolerance
  to make a failure pass. Exact-replay checks are bit for bit. A tolerance
  comparison fails on NaN.
- **Ported tests are not inherited.** A test from Charlotte comes over only
  if it guards a contract on the inventory.

## Checks

`python3 scripts/check.py all` runs, in order, stopping at the first
failure:

- **Formatting:** the pinned Google `.clang-format` for C++, the app's
  pinned Prettier for TypeScript (`format --fix` applies both).
- **Boundaries:** the import rules of
  [file-mapping.md](docs/architecture/file-mapping.md), read from the
  sources, C++ and TypeScript.
- **Static analysis:** the pinned clang-tidy over the native build's
  compilation database; every finding fails. Fix findings rather than
  excluding checks; a narrow exception is justified beside the code. Code
  only in the browser build is reviewed instead. Ported files unchanged since
  porting (their SHA-256 matches `src/ported.json`) are skipped and listed;
  a ported file that changes is analyzed from then on.
- **Tests:** the contract suites on the native build, the kernels suite
  among them, which needs a GPU; the app's type check and TypeScript suites.

By hand, when their trigger changes:

- The oracle, `frizzle_oracle`, when the inference path changes: the GGUF
  reader, formats, residency and upload, graph, kernels, program, cache,
  sampler or turn loop.
- The replay and edits scenario, `frizzle_replay`, when edits, the graph or
  the turn loop change.
- The browser module, the `wasm-release` preset in the
  `emscripten/emsdk:6.0.8` container, when `src/` or `src/wasm/` changes.

A build passing, or the page loading, is not evidence of correctness. Report
only the checks actually run.
