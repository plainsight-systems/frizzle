"""Pinned analysis of every project compilation entry, with fatal diagnostics.

Use an isolated database per command; retain actual compile flags. On macOS,
add the selected Xcode SDK's libc++ headers for the separately installed analyzer.
CMake's database is unchanged. Headers are analyzed through their includers.

Files ported from Charlotte and unchanged since porting are not analyzed: a
file whose SHA-256 still matches src/ported.json is left out, as a translation
unit and as a header, and every run lists what was left out. A ported file
that has changed is analyzed like any other (docs/process/workflow.md).
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from tooling import ROOT, BROWSER_ONLY, SEPARATE_BUILDS, SOURCE_ROOTS, SUFFIXES, TRANSLATION_SUFFIXES, LLVM_VERSION, find_tool, source_files


def macos_arguments(arguments):
    """The macOS SDK system root, which the compile command leaves implicit.

    The analyzer needs it for the C headers. Nothing else is added: CMake
    refuses any native compiler but the pinned LLVM clang, which uses that
    LLVM's own libc++ headers, and the analyzer from the same LLVM finds them
    by itself. Naming the SDK's libc++ as well would put two standard
    libraries in one translation unit.
    """
    if sys.platform != 'darwin':
        return []
    if any(flag == '-isysroot' or flag.startswith(('-isysroot', '--sysroot=')) for flag in arguments):
        return []
    return ['-isysroot', subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()]


# The configuration's own exclusions (.clang-tidy), kept when ported headers are added.
# Generated, frozen test data (tests/fixtures/) is not authored code.
CONFIGURED_EXCLUSION = r'.*/(build|_deps|\.cache|tests/fixtures)/.*'


def ported_unchanged():
    """Ported files whose current SHA-256 still matches the port manifest."""
    manifest = ROOT / 'src' / 'ported.json'
    if not manifest.is_file():
        return set()
    files = json.loads(manifest.read_text())['files']
    return {(ROOT / path).resolve() for path, digest in files.items()
            if (ROOT / path).is_file() and hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest}


def run(build, jobs):
    binary = find_tool('clang-tidy')
    database_path = build / 'compile_commands.json'
    if not database_path.is_file():
        raise ValueError(f'No compilation database at {database_path}; configure the native build first')
    database = json.loads(database_path.read_text())
    unchanged = ported_unchanged()
    excluded_headers = sorted(p for p in unchanged if p.suffix in SUFFIXES - TRANSLATION_SUFFIXES)
    exclusion = '|'.join([CONFIGURED_EXCLUSION, *(re.escape(str(p)) for p in excluded_headers)])
    commands = []
    for entry in database:
        directory = Path(entry['directory']).resolve()
        source = (directory / entry['file']).resolve()
        if not any(source.is_relative_to(ROOT / name) for name in SOURCE_ROOTS) or source in unchanged:
            continue
        if source.suffix not in TRANSLATION_SUFFIXES:
            raise ValueError(f'Unsupported analysis language: {source}')
        if not source.is_file() or not directory.is_dir():
            raise ValueError(f'Stale compilation entry: {source}; reconfigure here')
        arguments = entry.get('arguments') or shlex.split(entry['command'])
        arguments += macos_arguments(arguments)
        commands.append({'directory': str(directory), 'file': str(source), 'arguments': arguments})
    # Missing authored translation units must fail coverage, not disappear from lint.
    expected = {str(p.resolve()) for p in source_files()
                if p.suffix in TRANSLATION_SUFFIXES and p.resolve() not in unchanged
                and not any(p.is_relative_to(ROOT / name) for name in SEPARATE_BUILDS + BROWSER_ONLY)}
    found = {entry['file'] for entry in commands}
    if not commands or expected - found:
        raise ValueError(f'Incomplete analysis database; missing: {sorted(expected - found)}. Configure the native build with tests enabled.')
    report = ROOT / '.local/clang-tidy'
    report.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='commands-', dir=report) as temporary:
        def check(item):
            index, entry = item
            directory = Path(temporary) / str(index)
            directory.mkdir()
            # Isolate duplicate source entries so each uses its own compilation flags.
            (directory / 'compile_commands.json').write_text(json.dumps([entry]))
            result = subprocess.run([binary, entry['file'], '-p', str(directory), '--quiet',
                                     f'--exclude-header-filter={exclusion}'],
                                    cwd=ROOT, capture_output=True, text=True)
            log = report / f'{index:03d}-{Path(entry["file"]).name}.log'
            log.write_text(result.stdout + result.stderr)
            relative = str(Path(entry['file']).relative_to(ROOT))
            return {'file': relative, 'exit_code': result.returncode, 'log': str(log)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(check, enumerate(commands)))
    for result in results:
        print(f'{"PASS" if result["exit_code"] == 0 else "FAIL"} tidy: {result["file"]}', flush=True)
        if result['exit_code']:
            print(f'  Diagnostics: {result["log"]}', flush=True)
    if unchanged:
        print(f'Not analyzed: {len(unchanged)} files ported from Charlotte and unchanged since '
              f'(src/ported.json):', flush=True)
        for path in sorted(unchanged):
            print(f'  {path.relative_to(ROOT)}', flush=True)
    failed = sum(row['exit_code'] != 0 for row in results)
    (report / 'summary.json').write_text(json.dumps({'llvm': LLVM_VERSION, 'build': str(build), 'failed': failed, 'results': results}, indent=2) + '\n')
    if failed:
        raise ValueError(f'clang-tidy failed for {failed}/{len(results)} commands. Diagnostics remain enabled.')
