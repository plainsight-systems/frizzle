#!/usr/bin/env python3
"""Fetches the one model Frizzle runs, as pinned in config/model.json.

Native analysis and tests read the same file the browser downloads. This
streams it from the pinned URL into .cache/models/<id>.gguf, hashing as it
goes, and keeps it only if both its size and its SHA-256 match the pin; a
file that does not match is deleted and the run fails. A file already in
place is checked against the pin, not fetched again.

Standard library only; nothing is installed.

    python3 scripts/fetch_model.py
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / 'config' / 'model.json'
DESTINATION = ROOT / '.cache' / 'models'
CHUNK = 1 << 20


def digest_of(path):
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(CHUNK), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    pin = json.loads(PIN.read_text())
    target = DESTINATION / f"{pin['id']}.gguf"
    if target.is_file():
        if target.stat().st_size == pin['sizeBytes'] and digest_of(target) == pin['sha256']:
            print(f'{target.relative_to(ROOT)}: present and matches the pin', file=sys.stderr)
            return 0
        sys.exit(f'{target.relative_to(ROOT)} does not match config/model.json; delete it and fetch again')

    # Written beside the target, then moved into place, so a partial download
    # is never mistaken for the model.
    DESTINATION.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    sha = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(pin['url']) as response, partial.open('wb') as file:
        for block in iter(lambda: response.read(CHUNK), b''):
            file.write(block)
            sha.update(block)
            size += len(block)
    if size != pin['sizeBytes'] or sha.hexdigest() != pin['sha256']:
        partial.unlink()
        sys.exit(f"downloaded {size} bytes with SHA-256 {sha.hexdigest()}; the pin is "
                 f"{pin['sizeBytes']} bytes, {pin['sha256']}; not kept")
    partial.replace(target)
    print(f'{target.relative_to(ROOT)}: downloaded and matches the pin', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
