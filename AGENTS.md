# Working on Frizzle

## Session entry

Read `docs/process/MEMORY.md`, `docs/process/QUEUE.md` and
`docs/process/workflow.md`, then the relevant architecture documents and
source. `CLAUDE.md` points to the same entry path. `brief.md` and
`CLAUDE.local.md` are local planning files; read them if present, never commit
them.

## Local rules

- Follow the self-contained workflow in `docs/process/workflow.md`. Do not
  import governance documents, submodules or symlinks from other projects.
- Frizzle is about the output: the next-token prediction and the reply, and
  whether a change inside one layer moves them. Everything shown and editable
  belongs to the last position. Inspecting or editing earlier positions is out
  of scope.
- The model runs in the browser only, on WebGPU. Native builds exist for
  analysis and tests, never as a product.
- Design the architecture before the engine: logical overview, change axes,
  then file mapping. Design each change in its file headers.
- C++ header design and implementation must each query the local
  `cpp-guidelines` and `cpp-perf-guidelines` MCP servers, and cite the rules
  read in opening file comments. Follow the consultation and unavailable-server
  rules in the workflow.
- Code from Charlotte comes over only as listed in
  `docs/research/2026-10-27-charlotte-inventory.md`, which records the source
  commit. Add anything else copied from a sibling project to that record, or to
  a dated research note like it.
- Keep `app/` (the interface) separate from `web/` (the site that hosts it).
- **Tests lock contracts.** A test guards a named contract that someone would
  notice breaking, and never needs changing for a refactor. The inventory in
  `tests/README.md` is the whole test suite; adding to it is a decision.
  Integration tests are capped at five. Follow the test rules in the workflow.
- **Headers and tests, review, code, review.** Every step, in any language:
  the files with their design in their opening comments, and the step's
  tests; both reviewed; then the code; that reviewed; then the next step.
  Tests are never a step of their own at the end.
- Run `python3 scripts/check.py all` for C++ changes. Use the pinned Google
  formatter and the fatal clang-tidy policy; fix findings instead of disabling
  check families. Report only checks actually performed.
- Keep current decisions in MEMORY and remaining work in QUEUE; both are
  public. Conversation summaries, approvals, diaries and logs go in ignored
  `.local/`. Reserve `docs/research/` for investigations.
- A tech demo is a complete implementation of a narrow scope. Reduce scope
  rather than lowering correctness within it.
- Keep machine-specific paths and credentials out of tracked files.
