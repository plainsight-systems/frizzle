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
  database. Logs and a summary go to ignored `.local/clang-tidy/`.
- `all` runs both. Building, the test inventory and the site check join it as
  those parts of the repository are created.

Until there is C++ in the repository, `format` reports that there is nothing to
check and fails, and `tidy` fails for want of a compilation database. That is
the honest result, not a broken script.

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
put two standard libraries in one translation unit. Which libc++ library the
native build links against on macOS is still open: nothing links yet.

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
