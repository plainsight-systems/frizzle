#!/usr/bin/env python3
"""One check entry point. Installs nothing, downloads nothing.

Adapted from Bullet Witch. `format` and `tidy` work today; the build, test and
site checks join `all` as those parts of the repository exist (docs/process/QUEUE.md).
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from tooling import ROOT, format_sources
import tidy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('format', 'tidy', 'all'))
    parser.add_argument('--fix', action='store_true', help='format action only: apply Google formatting')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/native-debug',
                        help='native build whose compile_commands.json clang-tidy reads')
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
    if args.action in ('tidy', 'all'):
        tidy.run(build, args.jobs)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Check failed: {error}', file=sys.stderr)
        sys.exit(1)
