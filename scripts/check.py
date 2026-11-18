#!/usr/bin/env python3
"""One check entry point. Installs nothing, downloads nothing.

Adapted from Bullet Witch. `format` (clang-format, and Prettier for the app),
`boundaries` (the import rules of docs/architecture/file-mapping.md), `tidy`,
`test` (the routine suite, tests/README.md, and the app's type check) and
`site` (the Site build scenario, which builds the engine, the app and the
site: tests/integration/site/).
The llama.cpp oracle is not a check here: it runs when the inference path
changes (AGENTS.md, Checks).
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import app
import boundaries
from tooling import ROOT, format_sources
import tidy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('format', 'boundaries', 'tidy', 'test', 'site', 'all'))
    parser.add_argument('--fix', action='store_true', help='format action only: apply Google formatting and Prettier')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/native-debug',
                        help='native build whose compile_commands.json clang-tidy reads, and whose tests run')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 1, 4))
    args = parser.parse_args()
    if args.jobs < 1 or (args.fix and args.action != 'format'):
        parser.error('Invalid option combination; --jobs must be positive')
    build = args.build_dir.resolve()
    if not build.is_relative_to(ROOT / 'build'):
        parser.error("--build-dir must be under this repository's ignored build/")
    # Gates run in order; the first failure stops the rest.
    if args.action in ('format', 'all'):
        format_sources(args.fix)
        app.format_app(args.fix)
    if args.action in ('boundaries', 'all'):
        boundaries.check()
    if args.action in ('tidy', 'all'):
        tidy.run(build, args.jobs)
    if args.action in ('test', 'all'):
        subprocess.run(['cmake', '--build', str(build), '--target', 'frizzle_tests',
                        'frizzle_kernels', '-j', str(args.jobs)], check=True)
        subprocess.run(['ctest', '--test-dir', str(build), '--output-on-failure'], check=True)
        app.typecheck()
        app.suites()
    if args.action in ('site', 'all'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests/integration/site'),
                        '-p', 'test_*.py', '-v'], check=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Check failed: {error}', file=sys.stderr)
        sys.exit(1)
