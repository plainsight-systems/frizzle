"""Pinned analysis of every project compilation entry, with fatal diagnostics.

Use an isolated database per command; retain actual compile flags. On macOS,
add the selected Xcode SDK's libc++ headers for the separately installed analyzer.
CMake's database is unchanged. Headers are analyzed through their includers.
"""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from tooling import ROOT, SOURCE_ROOTS, TRANSLATION_SUFFIXES, LLVM_VERSION, find_tool, source_files


def macos_arguments(arguments):
    if sys.platform != 'darwin':
        return []
    selected = None
    for index, flag in enumerate(arguments):
        if flag == '-isysroot':
            selected = arguments[index + 1]
        elif flag.startswith('-isysroot'):
            selected = flag[len('-isysroot'):]
        elif flag.startswith('--sysroot='):
            selected = flag.split('=', 1)[1]
    sdk = Path(selected or subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip())
    compiler = Path(subprocess.check_output(['xcrun', '--find', 'clang++'], text=True).strip())
    headers = next((p for p in (sdk / 'usr/include/c++/v1', compiler.parent.parent / 'include/c++/v1') if p.is_dir()), None)
    if headers is None:
        raise ValueError('Selected Xcode SDK/toolchain has no libc++ headers')
    return ['-isysroot', str(sdk), '-isystem', str(headers)]


def run(build, jobs):
    binary = find_tool('clang-tidy')
    database_path = build / 'compile_commands.json'
    if not database_path.is_file():
        raise ValueError(f'No compilation database at {database_path}; configure the native build first')
    database = json.loads(database_path.read_text())
    commands = []
    for entry in database:
        directory = Path(entry['directory']).resolve()
        source = (directory / entry['file']).resolve()
        if not any(source.is_relative_to(ROOT / name) for name in SOURCE_ROOTS):
            continue
        if source.suffix not in TRANSLATION_SUFFIXES:
            raise ValueError(f'Unsupported analysis language: {source}')
        if not source.is_file() or not directory.is_dir():
            raise ValueError(f'Stale compilation entry: {source}; reconfigure here')
        arguments = entry.get('arguments') or shlex.split(entry['command'])
        arguments += macos_arguments(arguments)
        commands.append({'directory': str(directory), 'file': str(source), 'arguments': arguments})
    # Missing authored translation units must fail coverage, not disappear from lint.
    expected = {str(p.resolve()) for p in source_files() if p.suffix in TRANSLATION_SUFFIXES}
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
            result = subprocess.run([binary, entry['file'], '-p', str(directory), '--quiet'], cwd=ROOT, capture_output=True, text=True)
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
    failed = sum(row['exit_code'] != 0 for row in results)
    (report / 'summary.json').write_text(json.dumps({'llvm': LLVM_VERSION, 'build': str(build), 'failed': failed, 'results': results}, indent=2) + '\n')
    if failed:
        raise ValueError(f'clang-tidy failed for {failed}/{len(results)} commands. Diagnostics remain enabled.')
