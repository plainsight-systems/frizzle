"""Pinned local tool discovery shared by checks and editor formatting; no installs."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LLVM_VERSION = '21.1.8'
SOURCE_ROOTS = ('src', 'tests', 'tools')
TRANSLATION_SUFFIXES = {'.c', '.cc', '.cpp', '.cxx'}
SUFFIXES = TRANSLATION_SUFFIXES | {'.h', '.hh', '.hpp', '.hxx'}
# Built in a project of its own against llama.cpp, never in Frizzle's build,
# so no compilation database of Frizzle's can hold it: formatted, not analyzed.
SEPARATE_BUILDS = ('tools/reference_logits',)
# Compiled only by the browser build (Emscripten), so no native compilation
# database holds it: formatted, and reviewed rather than analyzed
# (AGENTS.md, Checks).
BROWSER_ONLY = ('src/wasm',)
# The container the browser build runs in (CMakePresets.json, wasm-release).
EMSDK = 'emscripten/emsdk:6.0.8'


def find_tool(name):
    requested = os.environ.get(name.upper().replace('-', '_'))
    candidates = [requested] if requested else [
        str(ROOT / '.venv-tools' / 'bin' / name), f'{name}-21', name]
    if not requested and sys.platform == 'darwin' and shutil.which('brew'):
        for formula in ('llvm@21', 'llvm'):
            result = subprocess.run(['brew', '--prefix', formula], capture_output=True, text=True)
            if result.returncode == 0:
                candidates.append(str(Path(result.stdout.strip()) / 'bin' / name))
    # Accept only the pinned version and never recurse through the editor wrapper.
    for candidate in candidates:
        binary = shutil.which(candidate)
        if not binary or Path(binary).resolve() == ROOT / 'scripts' / name:
            continue
        result = subprocess.run([binary, '--version'], stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if result.returncode == 0 and re.search(r'\bversion ' + re.escape(LLVM_VERSION) + r'\b', result.stdout):
            return str(Path(binary).absolute())
    raise ValueError(f'{name} {LLVM_VERSION} is required; select it with '
                     f'{name.upper().replace("-", "_")}. No tool was installed or substituted.')


def source_files():
    result = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
                                      '--exclude-standard', '--', *SOURCE_ROOTS], cwd=ROOT)
    return sorted({ROOT / os.fsdecode(name) for name in result.split(b'\0') if name
                   and Path(os.fsdecode(name)).suffix in SUFFIXES
                   and (ROOT / os.fsdecode(name)).is_file()})


def format_sources(fix=False):
    files = source_files()
    if not files:
        raise ValueError('No C++ sources under src/, tests/ or tools/ yet; nothing was checked')
    flags = ['-i'] if fix else ['--dry-run', '--Werror']
    subprocess.run([find_tool('clang-format'), '--style=file', *flags, *map(str, files)], cwd=ROOT, check=True)
    print(f'Google formatting: {len(files)} files {"formatted" if fix else "clean"}; LLVM {LLVM_VERSION}', flush=True)
