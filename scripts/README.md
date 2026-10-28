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

On macOS, `tidy` adds the selected Xcode SDK's libc++ headers for the
separately installed analyzer, as Bullet Witch does. The native build's own
compiler is also LLVM 21.1.8 clang; which libc++ it uses on macOS is still
open.

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
