#!/usr/bin/env python3
"""Ports Charlotte's Gemma 3 path into src/, mechanically, and records it.

Each file listed in docs/research/2026-10-27-charlotte-inventory.md is read
from Charlotte's pinned commit (not its working tree), given only the
mechanical changes AGENTS.md allows on porting, and written to the same
path here:

  1. the namespace `bllm` becomes `frizzle`, and the macro prefix `BLLM_`
     becomes `FRIZZLE_`, as whole words, so a mention of one of Charlotte's
     own functions such as `bllm_generate` is left as it is;
  2. C++ files are formatted with this repository's pinned clang-format.

The SHA-256 of each file as ported is recorded in src/ported.json. A file
still matching its recorded hash is unchanged since porting; scripts/tidy.py
reads the manifest to tell ported-unchanged files from changed ones.

    python3 scripts/port_charlotte.py --charlotte <Charlotte checkout> [--verify]

A file that has changed since it was ported (its hash no longer matches the
manifest) is never overwritten; it is listed and left as it is.

--verify writes nothing: it redoes the port in memory and fails unless the
manifest matches it exactly, then lists the files that have changed since
porting. Nothing is downloaded or installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from tooling import ROOT, SUFFIXES, find_tool

COMMIT = 'e003048f249ec4b021a54d2578fbdf8608d1e274'
MANIFEST = ROOT / 'src' / 'ported.json'

# The inventory, as paths under Charlotte's src/. A trailing '/' takes every
# file directly in that directory at the commit.
INVENTORY = [
    # Architecture
    'core/arch/architecture.h', 'core/arch/describe.h', 'core/arch/describe.cpp', 'core/arch/gemma3/',
    'core/model/model_description.h', 'core/policy/policy.h',
    # Model file
    'core/gguf/arrays.h', 'core/gguf/arrays.cpp', 'core/gguf/byte_source.h', 'core/gguf/checked.h',
    'core/gguf/index.h', 'core/gguf/index.cpp', 'core/gguf/reader.h', 'core/gguf/reader.cpp',
    'core/gguf/types.h',
    # Formats
    'core/formats/format.h', 'core/formats/device_layout.h', 'core/formats/f32/', 'core/formats/f16/',
    'core/formats/q4_0/', 'core/formats/q4_1/', 'core/formats/q8_0/',
    'core/capability/capability.h', 'core/capability/capability.cpp',
    # GPU
    'core/gpu/device.h', 'core/gpu/device.cpp', 'core/gpu/device_requirements.h',
    'core/gpu/wgpu_handles.h', 'core/gpu/unique_handle.h', 'core/gpu/userdata.h',
    'core/gpu/callback_mode.h', 'core/gpu/self_check.h', 'core/gpu/self_check.cpp',
    'core/gpu/dispatch_math.h', 'core/gpu/dispatch_math.cpp',
    # Weights on the GPU
    'core/residency/file_ranges.h', 'core/residency/file_ranges.cpp',
    'core/residency/mapping.h', 'core/residency/mapping.cpp',
    'core/residency/piece_writer.h', 'core/residency/piece_writer.cpp',
    'core/residency/plan.h', 'core/residency/plan.cpp',
    'core/residency/routes.h', 'core/residency/routes.cpp',
    'core/residency/upload.h', 'core/residency/upload.cpp',
    'core/residency/upload_check.h', 'core/residency/upload_check.cpp',
    'core/residency/weight_view.h', 'core/residency/witness.cpp',
    # Graph and kernels
    'core/graph/graph.h', 'core/graph/graph.cpp',
    'core/kernels/interface.h', 'core/kernels/order.h',
    'core/kernels/program.h', 'core/kernels/program.cpp',
    'core/kernels/schedule.h', 'core/kernels/schedule.cpp', 'core/kernels/step.wgsl',
    'core/kernels/attention/', 'core/kernels/gather/', 'core/kernels/matmul/', 'core/kernels/norm/',
    'core/kernels/rope/rope.h', 'core/kernels/rope/rope.cpp', 'core/kernels/rope/rope.wgsl',
    'core/kernels/rope/rope_cache.wgsl', 'core/kernels/topk/',
    # rope.cpp builds the encoder form too, so its shader comes over with it,
    # unused by Gemma 3 (docs/research/2026-10-27-charlotte-inventory.md).
    'core/kernels/rope/rope_encode.wgsl',
    # Cache
    'core/cache/kv.h', 'core/cache/kv.cpp', 'core/cache/prefix.h', 'core/cache/prefix.cpp',
    # Sampling and the turn
    'core/sampler/',
    'core/runtime/runtime.h', 'core/runtime/runtime.cpp', 'core/runtime/turn.h', 'core/runtime/turn.cpp',
    'core/runtime/generator.h', 'core/runtime/generator.cpp', 'core/runtime/stops.h', 'core/runtime/stops.cpp',
    # Tokenizer
    'core/tokenizer/tokenizer.h', 'core/tokenizer/load.h', 'core/tokenizer/load.cpp',
    'core/tokenizer/vocabulary.h', 'core/tokenizer/vocabulary.cpp',
    'core/tokenizer/special.h', 'core/tokenizer/special.cpp',
    'core/tokenizer/admission.h', 'core/tokenizer/admission.cpp',
    'core/tokenizer/detokenizer.h', 'core/tokenizer/detokenizer.cpp',
    'core/tokenizer/unicode.h', 'core/tokenizer/unicode.cpp',
    'core/tokenizer/unicode_tables.h', 'core/tokenizer/unicode_tables.cpp',
    'core/tokenizer/utf8_stream.cpp',
    'core/tokenizer/bpe/sentencepiece_bpe.h', 'core/tokenizer/bpe/sentencepiece_bpe.cpp',
    'core/tokenizer/bpe/sentencepiece_merges.h', 'core/tokenizer/bpe/sentencepiece_merges.cpp',
    'core/tokenizer/bpe/merge.h', 'core/tokenizer/bpe/merge.cpp',
    'core/tokenizer/bpe/merge_table.h', 'core/tokenizer/bpe/merge_table.cpp',
    # Other
    'core/diagnostics.h', 'core/run_guard.h',
    'shaders/vector_add.wgsl',
]


def git(charlotte, *arguments):
    return subprocess.check_output(['git', '-C', str(charlotte), *arguments])


def expand(charlotte):
    """Inventory entries to Charlotte paths, failing on any entry that names nothing."""
    listed = set(git(charlotte, 'ls-tree', '-r', '--name-only', COMMIT, '--', 'src').decode().split('\n'))
    paths = []
    for entry in INVENTORY:
        path = f'src/{entry}'
        if entry.endswith('/'):
            found = sorted(p for p in listed if p.startswith(path) and '/' not in p[len(path):])
            if not found:
                raise ValueError(f'{path} is empty at {COMMIT[:7]}')
            paths += found
        elif path in listed:
            paths.append(path)
        else:
            raise ValueError(f'{path} does not exist at {COMMIT[:7]}')
    if len(set(paths)) != len(paths):
        raise ValueError('An inventory entry is listed twice')
    return paths


def rename(text):
    return re.sub(r'\bBLLM_', 'FRIZZLE_', re.sub(r'\bbllm\b', 'frizzle', text))


def port(charlotte, path, clang_format):
    data = rename(git(charlotte, 'show', f'{COMMIT}:{path}').decode('utf-8'))
    if Path(path).suffix in SUFFIXES:
        data = subprocess.run([clang_format, '--style=file', f'--assume-filename={path}'], cwd=ROOT,
                              input=data, capture_output=True, text=True, check=True).stdout
    return data.encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--charlotte', type=Path, required=True, help="a checkout of Charlotte's repository")
    parser.add_argument('--verify', action='store_true', help='write nothing; fail unless everything matches')
    args = parser.parse_args()
    clang_format = find_tool('clang-format')
    paths = expand(args.charlotte)
    manifest = {
        '$comment': 'Written by scripts/port_charlotte.py. Each file as ported from Charlotte: the '
                    'namespace and macro prefix renamed, C++ formatted, nothing else. A file whose '
                    'SHA-256 still matches is unchanged since porting.',
        'source': 'Charlotte',
        'commit': COMMIT,
        'files': {},
    }
    recorded = json.loads(MANIFEST.read_text())['files'] if MANIFEST.is_file() else {}
    changed = []
    for path in paths:
        data = port(args.charlotte, path, clang_format)
        manifest['files'][path] = hashlib.sha256(data).hexdigest()
        target = ROOT / path
        if target.is_file() and target.read_bytes() != data:
            current = hashlib.sha256(target.read_bytes()).hexdigest()
            if path in recorded and current != recorded[path]:
                changed.append(path)
                continue
            if args.verify:
                raise ValueError(f'{path} differs from a fresh port but matches its recorded hash; '
                                 'the manifest is wrong')
        if not args.verify:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    text = json.dumps(manifest, indent=1) + '\n'
    unchanged = len(paths) - len(changed)
    if args.verify:
        if not MANIFEST.is_file() or MANIFEST.read_text() != text:
            print(f'{MANIFEST.relative_to(ROOT)} differs from a fresh port of Charlotte {COMMIT[:7]}')
            return 1
        print(f'The manifest matches a fresh port of Charlotte {COMMIT[:7]}; '
              f'{unchanged} of {len(paths)} files are as ported')
    else:
        MANIFEST.write_text(text)
        print(f'Ported {len(paths)} files from Charlotte {COMMIT[:7]}; manifest: {MANIFEST.relative_to(ROOT)}')
    if changed:
        print('Changed since porting, left as they are:', *changed, sep='\n  ')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Port failed: {error}', file=sys.stderr)
        sys.exit(1)
