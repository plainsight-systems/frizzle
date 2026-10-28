"""The import rules of docs/architecture/file-mapping.md, read from the sources.

C++: src/core/ includes nothing from src/wasm/, tools/ or tests/, and nothing
of Emscripten's, so the core never knows it is in a browser and builds and
tests natively.

TypeScript, under app/src/:
- React (react, react-dom and their subpaths) is imported only by the root
  files, views/ and theme/.
- session/, shape/, engine/, protocol/ and worker/ are plain TypeScript: no
  .tsx files, and no React.
- worker/ imports nothing from the root files, views/, session/, shape/ or
  theme/.
- Nothing outside worker/ imports from worker/: the page reaches the worker
  only through protocol/.

Imports are read with regular expressions over each file: `import ... from`,
`export ... from`, side-effect `import '...'`, and `import('...')`. A worker
started by URL (`new Worker(new URL(...))`) is not an import, and is the
protocol's to make. Every violation is listed, by file and line.
"""
from pathlib import Path
import re

from tooling import ROOT

CORE = ROOT / 'src' / 'core'
APP_SRC = ROOT / 'app' / 'src'

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
CORE_FORBIDDEN = ('wasm/', 'tools/', 'tests/')


def _forbidden_in_core(included):
    return included.startswith(CORE_FORBIDDEN) or 'emscripten' in included

SPECIFIER = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|^\s*import\s+)['"]([^'"]+)['"]""", re.MULTILINE)
REACT = re.compile(r'^(react|react-dom)(/|$)')
REACT_ALLOWED = {'', 'views', 'theme'}  # '' is the root files
PLAIN = {'session', 'shape', 'engine', 'protocol', 'worker'}
WORKER_FORBIDDEN = {'', 'views', 'session', 'shape', 'theme'}


def _area(path):
    """The top folder under app/src/ that `path` is in; '' for the root files."""
    parts = path.relative_to(APP_SRC).parts
    return parts[0] if len(parts) > 1 else ''


def _cpp():
    found = []
    for path in sorted(CORE.rglob('*')):
        if path.suffix not in {'.h', '.cpp', '.wgsl'}:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = INCLUDE.match(line)
            if match and _forbidden_in_core(match.group(1)):
                found.append(f'{path.relative_to(ROOT)}:{number}: the core includes {match.group(1)}')
    return found


def _typescript():
    found = []
    if not APP_SRC.is_dir():
        return found
    files = sorted(p for p in APP_SRC.rglob('*') if p.suffix in {'.ts', '.tsx'})
    for path in files:
        area = _area(path)
        where = path.relative_to(ROOT)
        if area in PLAIN and path.suffix == '.tsx':
            found.append(f'{where}: {area}/ is plain TypeScript, so holds no .tsx')
        text = path.read_text()
        for match in SPECIFIER.finditer(text):
            spec = match.group(1)
            number = text.count('\n', 0, match.start()) + 1
            if REACT.match(spec) and area not in REACT_ALLOWED:
                found.append(f'{where}:{number}: React ({spec}) outside the root, views/ and theme/')
            if not spec.startswith('.'):
                continue
            target = (path.parent / spec).resolve()
            if not target.is_relative_to(APP_SRC.resolve()):
                continue
            target_area = _area(target if target.suffix else target / 'x')
            if area == 'worker' and target_area in WORKER_FORBIDDEN:
                found.append(f'{where}:{number}: the worker imports page code ({spec})')
            if area != 'worker' and target_area == 'worker':
                found.append(f'{where}:{number}: imports the worker directly ({spec}); use protocol/')
    return found, len(files)


def check():
    cpp = _cpp()
    typescript, count = _typescript()
    problems = cpp + typescript
    for problem in problems:
        print(problem, flush=True)
    if problems:
        raise ValueError(f'{len(problems)} boundary violations (docs/architecture/file-mapping.md)')
    print(f'Boundaries: the core includes nothing of src/wasm/, tools/ or tests/; '
          f'{count} TypeScript files follow the app\'s import rules', flush=True)
