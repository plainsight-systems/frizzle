# Checks and tooling

Run from the repository root with Python 3.9 or later.

```sh
python3 scripts/check.py format --fix
python3 scripts/check.py all
```

- `format` checks every C/C++ file under `src/`, `tests/` and `tools/` with
  the Google `.clang-format`; `--fix` applies it.
- `tidy` runs clang-tidy over every entry in the native build's
  `compile_commands.json` (default `build/native-debug`), each in isolation,
  and fails on any finding or on an authored translation unit missing from the
  database. Logs and a summary go to ignored `.local/clang-tidy/`. Frozen
  test data under `tests/fixtures/` is not analyzed, nor is
  `tools/reference_logits/`, which builds only against llama.cpp in a project
  of its own. In `tests/`, magic numbers are allowed, and so is doctest's
  do/while in its assertion macros (`tests/.clang-tidy`).
- `test` builds and runs the contract suites (`frizzle_tests`) through CTest.
  The llama.cpp oracle is not part of it ([tests](../tests/README.md)).
- `all` runs all three. The site check joins it when the site exists.

## Pinned tools

clang-format and clang-tidy are pinned to **LLVM 21.1.8**, as in Bullet Witch.
`tooling.py` looks in `.venv-tools/bin`, versioned and unversioned PATH entries
and Homebrew's `llvm@21` and `llvm` prefixes, and accepts only that exact
version. `CLANG_FORMAT` and `CLANG_TIDY` select an executable explicitly; a
wrong version fails rather than being substituted. Nothing is installed by
these scripts.

On macOS, `tidy` gives the analyzer the Xcode SDK's system root, which the
compile command leaves implicit, and nothing else. The native build is
compiled by LLVM 21.1.8 clang against that LLVM's own libc++ headers, which
the analyzer finds by itself; adding the SDK's libc++ headers as well would
put two standard libraries in one translation unit. The native tool links
the system's `/usr/lib/libc++.1.dylib` against those headers, and runs so on
macOS 26.6.2 (checked 2026-10-28); whether to pin that pairing is open
([QUEUE](../docs/process/QUEUE.md)).

`tidy` leaves out files ported from Charlotte and unchanged since (their
SHA-256 still matches `src/ported.json`) and lists them on every run; see the
[workflow](../docs/process/workflow.md#checks).

`port_charlotte.py --charlotte <checkout>` is the mechanical port itself: it
reads the inventoried files from Charlotte's pinned commit, renames the
namespace and macro prefix, formats C++, and writes the manifest. It never
overwrites a file that has changed since it was ported. `--verify` redoes the
port in memory and checks the manifest against it.

`scripts/clang-format` is the editor's formatter: it forwards to the same
pinned binary and fails if it is missing.

`editor_wasm.py` gives the editor the browser-only C++ (`src/wasm/`), which
the native database cannot hold. It configures the wasm-release preset in the
pinned container (which exports that build's database), copies the
container's headers into ignored `.cache/emsdk-6.0.8/`, and rewrites the
browser-only entries as commands for the pinned clang++ in ignored
`build/wasm-editor/compile_commands.json`, which `.vscode/settings.json`
lists after the native database. It parses every rewritten file with
`-fsyntax-only` and fails if one does not compile, so the editor never reads
a database it cannot follow. The module is still built only by em++ in the
container. Run it after a wasm build, and again when the image or the
browser-only files' flags change:

```sh
python3 scripts/editor_wasm.py
```

## Building

```sh
python3 scripts/fetch_dawn.py
CXX=$(brew --prefix llvm@21)/bin/clang++ cmake --preset native-debug
docker run --rm -v "$PWD":/src -w /src emscripten/emsdk:6.0.8 emcmake cmake --preset wasm-release
```

`fetch_dawn.py` downloads the Dawn release pinned in `cmake/dawn.json`,
checks its SHA-256 and unpacks it into ignored `.cache/dawn/`; an archive
that does not match is not unpacked. CMake refuses any native compiler but
LLVM clang 21.1.8 and any Emscripten but 6.0.8, rather than substituting one.

## The native tool

```sh
python3 scripts/fetch_model.py
CXX=$(brew --prefix llvm@21)/bin/clang++ cmake --preset native-release
cmake --build build/native-release
build/native-release/frizzle "Write a short poem about the sea."
```

`make_gguf_fixtures.py` writes the small GGUF files the model-file tests read
into `tests/fixtures/gguf/`; `--check` fails if a committed file differs from
what it makes.

`make_tokenizer_fixture.py` copies the pinned model's tokenizer entries, and
nothing else, into `tests/fixtures/tokenizer/gemma-3-1b-it-q4_0.gguf` (6.5 MB),
so the tokenizer tests need no model. It checks the model against its pin
first; run it again only when the pin changes. `--check` fails if the
committed file differs.

`fetch_model.py` downloads the model pinned in `config/model.json` into
ignored `.cache/models/`, checking its size and SHA-256 before the file takes
its final name. The tool takes the prompt with Gemma's chat template already
applied; `frizzle` with no arguments prints the rendered form. A file keeps
the newline after `model` that a shell's `$(...)` would drop.
