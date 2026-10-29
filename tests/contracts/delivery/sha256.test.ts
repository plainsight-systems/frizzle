// Axis V: the delivery suite (tests/README.md), SHA-256.
//
// Contract locked (app/src/worker/delivery/sha256.ts): the digest of the
// bytes given is FIPS 180-4's, whole and however the bytes are split into
// updates; a finished hash cannot be updated or read again.
//
// Inputs: the test vectors published with FIPS 180-2's examples (NIST,
// "Secure Hash Standard", appendix B: "abc", the 448-bit message, one
// million "a"), and the empty message's well-known digest.

import assert from "node:assert/strict";
import { test } from "node:test";

import { Sha256 } from "../../../app/src/worker/delivery/sha256.ts";

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);

const VECTORS: readonly {
  readonly input: Uint8Array;
  readonly digest: string;
}[] = [
  {
    input: bytes(""),
    digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  },
  {
    input: bytes("abc"),
    digest: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  },
  {
    input: bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
    digest: "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
  },
  {
    input: new Uint8Array(1_000_000).fill(0x61),
    digest: "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
  },
];

test("each published vector, given whole", () => {
  for (const { input, digest } of VECTORS) {
    assert.equal(new Sha256().update(input).hex(), digest);
  }
});

test("the 448-bit message split at every byte, and byte by byte", () => {
  const vector = VECTORS[2];
  assert.ok(vector !== undefined);
  for (let at = 0; at <= vector.input.length; at++) {
    const hash = new Sha256();
    hash.update(vector.input.subarray(0, at));
    hash.update(vector.input.subarray(at));
    assert.equal(hash.hex(), vector.digest, `split at ${at}`);
  }
  const each = new Sha256();
  for (let i = 0; i < vector.input.length; i++) {
    each.update(vector.input.subarray(i, i + 1));
  }
  assert.equal(each.hex(), vector.digest);
});

test("one million 'a' in uneven chunks", () => {
  const vector = VECTORS[3];
  assert.ok(vector !== undefined);
  const hash = new Sha256();
  for (let at = 0; at < vector.input.length; at += 997) {
    hash.update(vector.input.subarray(at, at + 997));
  }
  assert.equal(hash.hex(), vector.digest);
});

test("a finished hash refuses another update or digest", () => {
  const hash = new Sha256();
  hash.update(bytes("abc"));
  hash.hex();
  assert.throws(() => hash.update(bytes("d")));
  assert.throws(() => hash.hex());
});
