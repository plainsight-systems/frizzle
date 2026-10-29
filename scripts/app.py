"""The app's checks, through its own npm scripts (app/package.json). No installs.

Node is pinned as LLVM is: the version in app/.nvmrc, and no other is used or
substituted. The packages are app/package-lock.json's, which a person
installs with `npm ci` in app/; a missing app/node_modules fails here, by
name, rather than installing anything.

- format: Prettier over the app's TypeScript and HTML (--fix applies it).
- typecheck: tsc over each of the app's type-check projects.
- suites: Node's built-in runner over tests/contracts/**/*.test.ts, the
  TypeScript suites of the inventory (tests/README.md), run as written: Node
  strips the types itself.
"""
from pathlib import Path
import subprocess

from tooling import ROOT

APP = ROOT / 'app'
NODE_VERSION = (APP / '.nvmrc').read_text().strip()


def _node():
    try:
        version = subprocess.run(['node', '--version'], capture_output=True, text=True,
                                 check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f'Node {NODE_VERSION} is required (app/.nvmrc); none runs here: {error}')
    if version != f'v{NODE_VERSION}':
        raise ValueError(f'Node {NODE_VERSION} is required (app/.nvmrc); this is {version}. '
                         'No other version was used or substituted.')
    if not (APP / 'node_modules').is_dir():
        raise ValueError('app/node_modules is missing: run `npm ci` in app/. Nothing was installed.')


def _run(script):
    _node()
    subprocess.run(['npm', 'run', '--silent', script], cwd=APP, check=True)


def format_app(fix=False):
    _run('format' if fix else 'format:check')
    print(f'Prettier: the app {"formatted" if fix else "clean"}; Node {NODE_VERSION}', flush=True)


def typecheck():
    _run('typecheck')
    print('TypeScript: the app type-checks', flush=True)


def suites():
    files = sorted(Path(ROOT / 'tests' / 'contracts').glob('**/*.test.ts'))
    if not files:
        print('TypeScript suites: none yet (tests/README.md)', flush=True)
        return
    _node()
    # A test that never settles fails at a minute, by name, rather than
    # holding the run open.
    subprocess.run(['node', '--test', '--test-timeout=60000', *map(str, files)], cwd=ROOT, check=True)
