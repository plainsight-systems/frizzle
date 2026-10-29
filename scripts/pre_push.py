"""The pre-push hook: the browser scenario, before anything leaves this machine.

Enabled once per clone with `git config core.hooksPath .githooks`; git then
runs .githooks/pre-push, which runs this, before every push, and a failure
stops the push. `git push --no-verify` skips it. The repository has no remote
yet, so the hook fires once one is added.

What it does, in order; each step fails by name and stops the push:
  1. Refuses unless what is pushed is what is checked out: every commit git
     names on standard input is HEAD, and the tracked files have no
     uncommitted changes. The scenario runs the working tree, so any other
     push would be checked against something else. A push that only deletes
     a remote branch pushes no commit, and passes.
  2. Checks Node is app/.nvmrc's (app.py), as every app check does.
  3. Builds the engine's browser module in the pinned container
     (emscripten/emsdk:6.0.8; CMake's wasm-release preset, target
     frizzle_web), so the scenario never runs a stale module. Needs Docker.
  4. Runs the browser scenario (tests/integration/browser/,
     `npm run scenario`), which checks its own prerequisites by name.

Not run: the routine checks (`python3 scripts/check.py all`), which are run
while working; this hook is the scenario CI cannot run, as it needs a GPU.
"""
import subprocess
import sys

import app
from tooling import ROOT

ZERO = '0' * 40
SCENARIO = ROOT / 'tests' / 'integration' / 'browser'
EMSDK = 'emscripten/emsdk:6.0.8'
WASM_BUILD = 'emcmake cmake --preset wasm-release && cmake --build build/wasm-release --target frizzle_web'


def _git(*arguments):
    return subprocess.run(['git', *arguments], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def _pushed_is_checked_out(lines):
    head = _git('rev-parse', 'HEAD')
    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f'git gave the hook a line it does not read: {line!r}')
        local_ref, local_sha = parts[0], parts[1]
        if local_sha == ZERO:
            continue  # a deletion pushes no commit
        if local_sha != head:
            raise ValueError(f'{local_ref} ({local_sha[:12]}) is not the checked-out commit '
                             f'({head[:12]}); the scenario runs the working tree, so check out '
                             'what you push, or push with --no-verify.')
    changed = _git('status', '--porcelain', '--untracked-files=no')
    if changed:
        raise ValueError('tracked files have uncommitted changes; the scenario would run them, '
                         f'not what is pushed:\n{changed}')


def main():
    try:
        _pushed_is_checked_out([line for line in sys.stdin.read().splitlines() if line.strip()])
        app.check_node()
        if not (SCENARIO / 'node_modules').is_dir():
            raise ValueError('tests/integration/browser/node_modules is missing: run `npm ci` '
                             'there. Nothing was installed.')
        print(f'pre-push: building the browser module in {EMSDK}', flush=True)
        subprocess.run(['docker', 'run', '--rm', '-v', f'{ROOT}:/src', '-w', '/src', EMSDK,
                        'sh', '-c', WASM_BUILD], check=True)
        print('pre-push: the browser scenario', flush=True)
        subprocess.run(['npm', 'run', '--silent', 'scenario'], cwd=SCENARIO, check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'pre-push refused the push: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
