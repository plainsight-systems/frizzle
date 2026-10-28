#!/usr/bin/env python3
"""Writes the small GGUF files the model-file tests read, into
tests/fixtures/gguf/.

One valid file, and each malformed file one deliberate change from a valid
one, so a reviewer can see what is wrong with each byte rather than trust a
hand-edited binary. Taken from Charlotte e003048's tools/make_fixture_gguf.py,
keeping only the cases tests/contracts/gguf/ reads; the bytes are the same.

    python3 scripts/make_gguf_fixtures.py           # write every case
    python3 scripts/make_gguf_fixtures.py --check   # fail if a file differs
"""
import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'tests' / 'fixtures' / 'gguf'

MAGIC = b'GGUF'
VERSION = 3

# gguf_type
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# ggml_type
T_F32, T_F16, T_Q4_0 = 0, 1, 2


def gstr(s: bytes) -> bytes:
    return struct.pack('<Q', len(s)) + s


def kv(key: bytes, vtype: int, payload: bytes) -> bytes:
    return gstr(key) + struct.pack('<I', vtype) + payload


def build(tensors, metadata=(), *, magic=MAGIC, version=VERSION,
          tensor_count=None, metadata_count=None, alignment=32):
    """tensors: list of (name, dims, ggml_type, data_bytes)."""
    md = b''.join(metadata)
    md_count = metadata_count if metadata_count is not None else len(metadata)
    t_count = tensor_count if tensor_count is not None else len(tensors)

    head = magic + struct.pack('<I', version)
    head += struct.pack('<QQ', t_count, md_count)
    head += md

    # Tensors back to back, each padded to `alignment`.
    infos, blobs, offset = b'', b'', 0
    for name, dims, ttype, data in tensors:
        infos += gstr(name) + struct.pack('<I', len(dims))
        for d in dims:
            infos += struct.pack('<q', d)
        infos += struct.pack('<I', ttype) + struct.pack('<Q', offset)
        pad = (-len(data)) % alignment
        blobs += data + b'\0' * pad
        offset += len(data) + pad

    body = head + infos
    body += b'\0' * ((-len(body)) % alignment)
    return body + blobs


def q4_0_blocks(n: int, scale_bits: int = 0x3C00, nibble_byte: int = 0x10) -> bytes:
    """n Q4_0 blocks: an fp16 scale, then 16 bytes of packed nibbles."""
    return (struct.pack('<H', scale_bits) + bytes([nibble_byte]) * 16) * n


def valid() -> bytes:
    return build(
        tensors=[
            (b'token_embd.weight', [64, 2], T_Q4_0, q4_0_blocks(4)),
            (b'output_norm.weight', [4], T_F32, struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)),
        ],
        metadata=[
            kv(b'general.architecture', STRING, gstr(b'qwen3')),
            kv(b'general.alignment', U32, struct.pack('<I', 32)),
            kv(b'qwen3.block_count', U32, struct.pack('<I', 28)),
            kv(b'qwen3.attention.head_count', U32, struct.pack('<I', 16)),
            kv(b'tokenizer.ggml.tokens', ARRAY,
               struct.pack('<IQ', STRING, 3) + gstr(b'a') + gstr(b'bb') + gstr(b'ccc')),
        ],
    )


def with_tensor_offset(bogus_offset: int) -> bytes:
    """A structurally valid file whose one tensor claims `bogus_offset`."""
    data = q4_0_blocks(1)
    head = MAGIC + struct.pack('<I', VERSION) + struct.pack('<QQ', 1, 0)
    info = gstr(b't') + struct.pack('<I', 1) + struct.pack('<q', 32)
    info += struct.pack('<I', T_Q4_0) + struct.pack('<Q', bogus_offset)
    body = head + info
    body += b'\0' * ((-len(body)) % 32)
    return body + data + b'\0' * ((-len(data)) % 32)


def tensors_at(*placements, alignment=32):
    """A file whose tensors sit at chosen relative offsets: (name, offset)."""
    data = q4_0_blocks(1)
    head = MAGIC + struct.pack('<I', VERSION) + struct.pack('<QQ', len(placements), 0)
    info = b''
    for name, offset in placements:
        info += gstr(name) + struct.pack('<I', 1) + struct.pack('<q', 32)
        info += struct.pack('<I', T_Q4_0) + struct.pack('<Q', offset)
    body = head + info
    body += b'\0' * ((-len(body)) % alignment)
    return body + data + b'\0' * 64


def with_alignment(vtype, payload, alignment=32):
    return build([(b't', [4], T_F32, b'\0' * 16)],
                 metadata=[kv(b'general.alignment', vtype, payload)], alignment=alignment)


CASES = {
    'valid': valid,
    'bad_magic': lambda: build([], magic=b'GGUX'),
    'bad_version': lambda: build([], version=2),
    'truncated_header': lambda: MAGIC + struct.pack('<I', VERSION) + b'\x01\x02',
    # More tensors than the file could hold.
    'tensor_count_too_large': lambda: build([], tensor_count=1 << 40),
    'metadata_count_lies': lambda: build([], metadata_count=5),
    # A declared offset that puts the data past the end of the file.
    'offset_past_eof': lambda: with_tensor_offset(1 << 40),
    # An offset where data start + offset wraps past 2^64: a naive
    # `start + offset > size` computes a small number and passes it.
    'offset_overflow': lambda: with_tensor_offset((1 << 64) - 64),
    # Cut inside declared tensor data, not padding.
    'data_truncated': lambda: build([(b't', [64], T_Q4_0, q4_0_blocks(2))])[:-40],
    'negative_dimension': lambda: build([(b't', [-4], T_F32, b'\0' * 16)]),
    'unknown_tensor_type': lambda: build([(b't', [4], 999, b'\0' * 16)]),
    # 33 elements is not a whole number of Q4_0 blocks.
    'not_block_aligned': lambda: build([(b't', [33], T_Q4_0, q4_0_blocks(2))]),
    'duplicate_tensor_name': lambda: build([
        (b'dup', [4], T_F32, b'\0' * 16),
        (b'dup', [4], T_F32, b'\0' * 16),
    ]),
    # Apart, so a check of neighbours in file order alone misses them.
    'duplicate_tensor_name_apart': lambda: build([
        (b'dup', [4], T_F32, b'\0' * 16),
        (b'other', [4], T_F32, b'\0' * 16),
        (b'dup', [4], T_F32, b'\0' * 16),
    ]),
    # The alignment must be a power-of-two uint32; neither is a reason to
    # fall back to the default.
    'alignment_wrong_type': lambda: with_alignment(U64, struct.pack('<Q', 32)),
    'alignment_not_power_of_two': lambda: with_alignment(U32, struct.pack('<I', 48)),
    # Every tensor offset is a multiple of the alignment.
    'misaligned_tensor_offset': lambda: tensors_at((b't', 16)),
    'overlapping_tensor_data': lambda: tensors_at((b'a', 0), (b'b', 0)),
    'empty_metadata_key': lambda: build([], metadata=[kv(b'', U32, struct.pack('<I', 1))]),
    'duplicate_metadata_key': lambda: build([], metadata=[
        kv(b'general.name', STRING, gstr(b'a')),
        kv(b'general.type', STRING, gstr(b'model')),
        kv(b'general.name', STRING, gstr(b'b')),
    ]),
    # The format's limit on a name is 64 bytes.
    'tensor_name_65_bytes': lambda: build([(b'n' * 65, [4], T_F32, b'\0' * 16)]),
    'nested_array': lambda: build([], metadata=[kv(b'bad', ARRAY, struct.pack('<IQ', ARRAY, 1))]),
    'unknown_value_type': lambda: build([], metadata=[kv(b'bad', 99, b'\0' * 4)]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='fail if a written file differs from what this script makes')
    args = parser.parse_args()
    differing = []
    for name, make in CASES.items():
        path = OUT / f'{name}.gguf'
        data = make()
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                differing.append(path.name)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if differing:
        print('differ from the script: ' + ', '.join(differing), file=sys.stderr)
        return 1
    print(f'{len(CASES)} GGUF fixtures ' + ('match' if args.check else 'written'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
