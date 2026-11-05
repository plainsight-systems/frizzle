#!/usr/bin/env python3
"""The editor's view of the browser-only C++ (tooling.BROWSER_ONLY, src/wasm/).

Those files compile only inside the pinned Emscripten container, so the
native compilation database the editor reads has no entry for them, and
their headers (<emscripten.h>, the browser's webgpu.h) are not on the host.
This writes a compilation database for them that the host's tools can
follow, and checks that it does.

What it does, in order; each step fails by name:
  1. In the container (tooling.EMSDK, as the browser build): configures the
     wasm-release preset, which exports its compilation database; prints the
     flags em++ adds to every compile with the emdawnwebgpu port; and copies
     the headers those flags name (the sysroot's, the port's, clang's own)
     into ignored .cache/emsdk-<version>/, at the paths they have in the
     container under that prefix. An earlier copy is replaced whole.
  2. On the host: takes the database's entries under BROWSER_ONLY and
     rewrites each as a command for the pinned LLVM clang++ (tooling.py):
     em++'s added flags first, then the entry's own; "--use-port" (which
     those flags expand) and "-mllvm" options (code generation only, and
     clang 21 does not take them all) dropped; container paths moved to the
     repository and the header copy, with clang's own headers as its
     resource directory. Written to ignored
     build/wasm-editor/compile_commands.json, which .vscode/settings.json
     lists after the native database.
  3. Runs every rewritten command with -fsyntax-only. A file that does not
     compile there fails the run, so the editor is never handed a database
     it cannot follow.

What it is not: the browser build. The module is compiled by em++ in the
container, never by these commands; this is only what the editor and its
analysis read, and a passing run says the host's clang parses the files as
Emscripten's would, not that the module builds. It writes machine paths
only into ignored files.

Needs Docker, the pinned LLVM, and the wasm build's generated headers (a
wasm build run once: scripts/README.md, Building).

    python3 scripts/editor_wasm.py
"""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from tooling import BROWSER_ONLY, EMSDK, ROOT, find_tool

CONTAINER_ROOT = '/src/'
CONTAINER_EMSDK = '/emsdk/'
HEADERS = ROOT / '.cache' / f'emsdk-{EMSDK.split(":")[1]}'
DATABASE = ROOT / 'build' / 'wasm-release' / 'compile_commands.json'
OUTPUT = ROOT / 'build' / 'wasm-editor' / 'compile_commands.json'
RESOURCE = 'emsdk/upstream/lib/clang'  # clang's own headers, under the copy
# The container's script: configure (quietly), copy the headers, then print
# em++'s flags as the last line of standard output.
CONTAINER = f'''set -e
emcmake cmake --preset wasm-release >&2
flags=$(em++ --cflags --use-port=emdawnwebgpu 2>/dev/null)
rm -rf "/src/{HEADERS.relative_to(ROOT)}"
copy() {{ mkdir -p "/src/{HEADERS.relative_to(ROOT)}$1" && cp -R "$1/." "/src/{HEADERS.relative_to(ROOT)}$1"; }}
copy /emsdk/upstream/emscripten/cache/sysroot/include
copy /emsdk/upstream/emscripten/cache/ports/emdawnwebgpu/emdawnwebgpu_pkg/webgpu/include
copy /emsdk/upstream/emscripten/cache/ports/emdawnwebgpu/emdawnwebgpu_pkg/webgpu_cpp/include
for resource in /emsdk/upstream/lib/clang/*; do copy "$resource/include"; done
echo "$flags"
'''


def _in_container():
    result = subprocess.run(['docker', 'run', '--rm', '-v', f'{ROOT}:/src', '-w', '/src', EMSDK,
                             'sh', '-c', CONTAINER], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f'the container step failed ({EMSDK}, exit {result.returncode})')
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise ValueError('em++ printed no flags')
    return shlex.split(lines[-1])


def _resource_dir():
    versions = sorted((HEADERS / RESOURCE).iterdir()) if (HEADERS / RESOURCE).is_dir() else []
    if len(versions) != 1:
        raise ValueError(f'expected one clang resource directory under {HEADERS / RESOURCE}, '
                         f'found {len(versions)}')
    return versions[0]


def _moved(argument):
    """A container path, wherever it starts in the argument, moved to the host."""
    for prefix, host in ((CONTAINER_ROOT, f'{ROOT}/'), (CONTAINER_EMSDK, f'{HEADERS}/emsdk/')):
        at = argument.find(prefix)
        if at != -1 and (at == 0 or argument[at - 1] in '=I'):
            return argument[:at] + host + argument[at + len(prefix):]
    return argument


def _without_mllvm(arguments):
    kept, skip = [], False
    for argument in arguments:
        if skip:
            skip = False
        elif argument == '-mllvm':
            skip = True
        elif not argument.startswith('--use-port='):
            kept.append(argument)
    return kept


def _entries(clang, added):
    if not DATABASE.is_file():
        raise ValueError(f'{DATABASE.relative_to(ROOT)} was not written by the configure step')
    browser_only = [f'{CONTAINER_ROOT}{name}/' for name in BROWSER_ONLY]
    entries = []
    for entry in json.loads(DATABASE.read_text()):
        if not any(entry['file'].startswith(prefix) for prefix in browser_only):
            continue
        own = entry['arguments'][1:] if 'arguments' in entry else shlex.split(entry['command'])[1:]
        arguments = [clang, f'-resource-dir={_resource_dir()}',
                     *(_moved(a) for a in _without_mllvm([*added, *own]))]
        entries.append({'directory': _moved(entry['directory'].rstrip('/') + '/').rstrip('/'),
                        'file': _moved(entry['file']), 'arguments': arguments})
    if not entries:
        raise ValueError(f'the database has no entry under {", ".join(BROWSER_ONLY)}')
    return entries


def _compiles(entry):
    arguments = list(entry['arguments'])
    if '-o' in arguments:  # the object is never written
        at = arguments.index('-o')
        del arguments[at:at + 2]
    result = subprocess.run([*arguments, '-fsyntax-only'], cwd=entry['directory'],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f'{Path(entry["file"]).relative_to(ROOT)} does not compile with the '
                         f'rewritten command:\n{result.stderr}')


def main():
    try:
        if shutil.which('docker') is None:
            raise ValueError('Docker is required: the headers come from the pinned container')
        clang = find_tool('clang++')
        added = _in_container()
        entries = _entries(clang, added)
        for entry in entries:
            _compiles(entry)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(entries, indent=2) + '\n')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'editor_wasm: {error}', file=sys.stderr)
        return 1
    names = ', '.join(str(Path(e['file']).relative_to(ROOT)) for e in entries)
    print(f'The editor\'s browser-only database: {len(entries)} files ({names}), each parsed by '
          f'the host\'s clang++; {OUTPUT.relative_to(ROOT)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
