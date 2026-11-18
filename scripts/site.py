#!/usr/bin/env python3
"""The project site (web/README.md): build it with the app inside, or serve it. Installs nothing.

build, in order, the first failure stopping the rest:
  1. The engine's module: CMake's wasm-release preset, configured and built
     in the pinned container (tooling.EMSDK), as scripts/README.md builds it.
  2. The app: `npm run build` in app/ at the pinned Node (app.check_node),
     into app/dist/.
  3. Hugo at its pin (tooling.find_hugo) over web/, which mounts app/dist/
     at app/ (web/hugo.toml), into a fresh directory beside the output.
  4. Only when Hugo succeeded, that directory replaces the output, by
     rename: the last good site is never left half written.
  The output is ignored .cache/site/public/, or --out (a directory that is
  missing or empty). --base-url replaces hugo.toml's production address.

serve: Hugo's server, listening on 127.0.0.1 and opened at
http://localhost:1313/ (--port; its live reload accepts that address alone),
over web/, with the app as last built, never cached by the browser, so a
rebuild shows on reload; refuses when app/dist/ is missing rather than
serving a site whose app/ is empty. It builds nothing.

Hugo's processed images are kept in ignored .cache/site/resources/
(hugo.toml, resourceDir), what its server writes in ignored
.cache/site/served/, and no build lock is written, so the tree stays clean.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import app
from tooling import EMSDK, ROOT, find_hugo

WEB = ROOT / 'web'
DIST = ROOT / 'app' / 'dist'
OUTPUT = ROOT / '.cache' / 'site' / 'public'
SERVED = ROOT / '.cache' / 'site' / 'served'  # what Hugo's server writes to disk


def build_engine():
    subprocess.run(['docker', 'run', '--rm', '-v', f'{ROOT}:/src', '-w', '/src', EMSDK, 'sh', '-c',
                    'emcmake cmake --preset wasm-release && cmake --build --preset wasm-release'],
                   check=True)


def build_app():
    app.check_node()
    subprocess.run(['npm', 'run', '--silent', 'build'], cwd=ROOT / 'app', check=True)


def build_site(out, base_url):
    hugo = find_hugo()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty; nothing was written')
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = out.with_name(out.name + '.building')
    shutil.rmtree(fresh, ignore_errors=True)
    command = [hugo, '--source', str(WEB), '--destination', str(fresh), '--environment', 'production',
               '--panicOnWarning', '--noBuildLock']
    if base_url is not None:
        command += ['--baseURL', base_url]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        shutil.rmtree(fresh, ignore_errors=True)
        raise
    if out.exists():
        out.rmdir()
    fresh.rename(out)


def build(out=OUTPUT, base_url=None):
    """Steps 1 to 4. With the default output, the last good site is replaced only after Hugo succeeds."""
    build_engine()
    build_app()
    if out == OUTPUT and OUTPUT.exists():
        staged = OUTPUT.with_name('public.next')
        shutil.rmtree(staged, ignore_errors=True)
        build_site(staged, base_url)
        old = OUTPUT.with_name('public.old')
        shutil.rmtree(old, ignore_errors=True)
        OUTPUT.rename(old)
        staged.rename(OUTPUT)
        shutil.rmtree(old)
    else:
        build_site(out, base_url)
    print(f'Site built into {out}', flush=True)


def serve(port):
    if not (DIST / 'index.html').is_file():
        raise ValueError('app/dist/ has no built app: run `python3 scripts/site.py build` first. '
                         'Nothing was served.')
    shutil.rmtree(SERVED, ignore_errors=True)  # no earlier build's files left to serve
    subprocess.run([find_hugo(), 'server', '--source', str(WEB), '--destination', str(SERVED),
                    '--bind', '127.0.0.1', '--port', str(port),
                    '--baseURL', f'http://localhost:{port}/', '--noBuildLock', '--noHTTPCache'],
                   check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest='action', required=True)
    build_parser = actions.add_parser('build')
    build_parser.add_argument('--out', type=Path, default=OUTPUT)
    build_parser.add_argument('--base-url')
    serve_parser = actions.add_parser('serve')
    serve_parser.add_argument('--port', type=int, default=1313)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.out.resolve(), args.base_url)
    else:
        serve(args.port)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Site failed: {error}', file=sys.stderr)
        sys.exit(1)
