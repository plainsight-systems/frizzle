#!/usr/bin/env python3
"""Writes tests/fixtures/tokenizer/gemma-3-1b-it-q4_0.gguf: the pinned model's
tokenizer alone, so the tokenizer tests need no model.

The output is a GGUF with no tensors whose metadata is every tokenizer.* entry
of the pinned file (config/model.json), copied byte for byte in the file's
order: the vocabulary, its scores and types, the special tokens, the chat
template. About 6.5 MB of the 722 MB file. The model is checked against its
pinned size and SHA-256 first; run scripts/fetch_model.py to get it.

Run it again only when the pinned model changes.

    python3 scripts/make_tokenizer_fixture.py           # write the fixture
    python3 scripts/make_tokenizer_fixture.py --check   # fail if it differs

Standard library only; nothing is installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / 'config' / 'model.json'
MODELS = ROOT / '.cache' / 'models'
OUT = ROOT / 'tests' / 'fixtures' / 'tokenizer'
CHUNK = 1 << 20
ALIGNMENT = 32

STRING, ARRAY = 8, 9
# Bytes of each fixed-size GGUF value type, by type number.
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, file):
        self.file = file

    def take(self, fmt):
        size = struct.calcsize('<' + fmt)
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError('the file ends inside its metadata')
        return struct.unpack('<' + fmt, data)

    def string(self):
        (length,) = self.take('Q')
        data = self.file.read(length)
        if len(data) != length:
            raise ValueError('the file ends inside a string')
        return data

    def skip_value(self, vtype):
        if vtype == STRING:
            self.string()
        elif vtype == ARRAY:
            element, count = self.take('IQ')
            if element == STRING:
                for _ in range(count):
                    self.string()
            elif element in FIXED:
                self.file.seek(FIXED[element] * count, 1)
            else:
                raise ValueError(f'an array of value type {element}')
        elif vtype in FIXED:
            self.file.seek(FIXED[vtype], 1)
        else:
            raise ValueError(f'value type {vtype}')


def digest_of(path):
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(CHUNK), b''):
            sha.update(block)
    return sha.hexdigest()


def tokenizer_only(model):
    """The GGUF made from `model`'s header and its tokenizer.* entries."""
    with model.open('rb') as file:
        read = Reader(file)
        magic = file.read(4)
        if magic != b'GGUF':
            raise ValueError(f'{model} is not a GGUF file')
        (version,) = read.take('I')
        _tensors, entries = read.take('QQ')
        kept = []
        for _ in range(entries):
            start = file.tell()
            key = read.string()
            (vtype,) = read.take('I')
            read.skip_value(vtype)
            if key.startswith(b'tokenizer.'):
                end = file.tell()
                file.seek(start)
                kept.append(file.read(end - start))
    head = b'GGUF' + struct.pack('<I', version) + struct.pack('<QQ', 0, len(kept))
    body = head + b''.join(kept)
    return body + b'\0' * ((-len(body)) % ALIGNMENT)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='fail if the committed fixture differs from what this script makes')
    args = parser.parse_args()
    pin = json.loads(PIN.read_text())
    model = MODELS / f"{pin['id']}.gguf"
    if not model.is_file():
        print(f'{model.relative_to(ROOT)} is missing; run scripts/fetch_model.py', file=sys.stderr)
        return 1
    if model.stat().st_size != pin['sizeBytes'] or digest_of(model) != pin['sha256']:
        print(f'{model.relative_to(ROOT)} does not match the pin in config/model.json', file=sys.stderr)
        return 1
    data = tokenizer_only(model)
    target = OUT / f"{pin['id']}.gguf"
    if args.check:
        if not target.is_file() or target.read_bytes() != data:
            print(f'{target.relative_to(ROOT)} differs from what the pinned model makes', file=sys.stderr)
            return 1
        print(f'{target.relative_to(ROOT)} matches the pinned model')
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f'{target.relative_to(ROOT)}: {len(data):,} bytes from {model.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'make_tokenizer_fixture: {error}', file=sys.stderr)
        sys.exit(1)
