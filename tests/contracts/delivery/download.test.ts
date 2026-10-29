// Axis V: the delivery suite (tests/README.md), the download.
//
// Contract locked (app/src/worker/delivery/download.ts): the pinned file is
// kept only when it arrives whole, as many bytes as the pin says, with the
// pin's SHA-256; otherwise it is refused by name and nothing is kept. A
// copy already kept is not downloaded again.
//
// Inputs, made here: a few kilobytes standing in for the model, with their
// real SHA-256 as the pin's. The server and the store are in-memory
// stand-ins at the interfaces download.ts uses (OpenDownload in fetch.ts,
// Store in store.ts), labelled as such: what they cannot show, the real
// fetch and OPFS, the browser scenario covers (tests/README.md).

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";

import {
  DownloadError,
  type Pin,
  ensureModel,
  keyOf,
} from "../../../app/src/worker/delivery/download.ts";
import type {
  Download,
  OpenDownload,
} from "../../../app/src/worker/delivery/fetch.ts";
import type {
  Kept,
  Store,
  Writer,
} from "../../../app/src/worker/delivery/store.ts";

const FILE = new Uint8Array(5000).map((_, i) => (i * 31 + 7) % 251);
const PIN: Pin = {
  url: "https://example.test/model.gguf",
  sizeBytes: FILE.length,
  sha256: createHash("sha256").update(FILE).digest("hex"),
};

// Test stand-in for the OPFS store: what was written, committed and
// discarded, in memory.
class MemoryStore implements Store {
  readonly kept_ = new Map<string, { bytes: Uint8Array; record: Kept }>();
  discarded = 0;
  async writer(key: string): Promise<Writer> {
    const parts: Uint8Array[] = [];
    return {
      write: async (bytes) => {
        parts.push(bytes.slice());
      },
      commit: async (record) => {
        const whole = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
        let at = 0;
        for (const p of parts) {
          whole.set(p, at);
          at += p.length;
        }
        this.kept_.set(key, { bytes: whole, record });
      },
      discard: async () => {
        this.discarded++;
      },
    };
  }
  async kept(key: string): Promise<Kept | null> {
    return this.kept_.get(key)?.record ?? null;
  }
}

// Test stand-in for the server: `bytes` in chunks of 1000, stating
// `statedSize`, and counting how often it is asked.
function server(bytes: Uint8Array, statedSize: number | null) {
  const asked = { count: 0 };
  const open: OpenDownload = async (url): Promise<Download> => {
    assert.equal(url, PIN.url);
    asked.count++;
    return {
      totalSize: statedSize,
      chunks: (async function* () {
        for (let at = 0; at < bytes.length; at += 1000) {
          yield bytes.subarray(at, at + 1000);
        }
      })(),
    };
  };
  return { open, asked };
}

const refusedAs = (pattern: RegExp) => (error: unknown) =>
  error instanceof DownloadError && pattern.test(error.message);

test("the pinned file is kept, whole, under its hash", async () => {
  const store = new MemoryStore();
  const { open } = server(FILE, FILE.length);
  const progress: number[] = [];
  const key = await ensureModel(PIN, open, store, (done) =>
    progress.push(done),
  );
  assert.equal(key, keyOf(PIN));
  assert.equal(key, `sha256-${PIN.sha256}`);
  const kept = store.kept_.get(key);
  assert.ok(kept !== undefined);
  assert.deepEqual(kept.bytes, FILE);
  assert.deepEqual(kept.record, {
    url: PIN.url,
    sizeBytes: PIN.sizeBytes,
    sha256: PIN.sha256,
  });
  assert.equal(progress.at(-1), FILE.length);
  assert.equal(store.discarded, 0);
});

test("a kept copy is not downloaded again", async () => {
  const store = new MemoryStore();
  const first = server(FILE, FILE.length);
  await ensureModel(PIN, first.open, store);
  const second = server(FILE, FILE.length);
  assert.equal(await ensureModel(PIN, second.open, store), keyOf(PIN));
  assert.equal(second.asked.count, 0);
});

test("a server stating another size is refused, nothing kept", async () => {
  const store = new MemoryStore();
  const { open } = server(FILE, FILE.length + 1);
  await assert.rejects(ensureModel(PIN, open, store), refusedAs(/size|bytes/));
  assert.equal(store.kept_.size, 0);
});

test("a download that ends early is refused, nothing kept", async () => {
  const store = new MemoryStore();
  const { open } = server(FILE.subarray(0, 4000), null);
  await assert.rejects(ensureModel(PIN, open, store), refusedAs(/ended|short/));
  assert.equal(store.kept_.size, 0);
  assert.equal(store.discarded, 1);
});

test("a download that runs long is refused, nothing kept", async () => {
  const store = new MemoryStore();
  const longer = new Uint8Array(FILE.length + 10);
  longer.set(FILE);
  const { open } = server(longer, null);
  await assert.rejects(ensureModel(PIN, open, store), refusedAs(/more|long/));
  assert.equal(store.kept_.size, 0);
  assert.equal(store.discarded, 1);
});

test("bytes whose hash is not the pin's are refused, nothing kept", async () => {
  const store = new MemoryStore();
  const altered = FILE.slice();
  altered[2500] = (altered[2500] ?? 0) ^ 1;
  const { open } = server(altered, altered.length);
  await assert.rejects(ensureModel(PIN, open, store), refusedAs(/SHA-256/));
  assert.equal(store.kept_.size, 0);
  assert.equal(store.discarded, 1);
});

test("a download aborted partway keeps nothing, and discards what it wrote", async () => {
  const store = new MemoryStore();
  const { open, asked } = server(FILE, FILE.length);
  const controller = new AbortController();
  // Abort once the first chunk is written: the download is under way, with
  // part of the file in the store.
  const aborted = ensureModel(
    PIN,
    open,
    store,
    () => controller.abort(),
    controller.signal,
  );
  await assert.rejects(aborted, (error: unknown) => {
    assert.equal(error, controller.signal.reason);
    return true;
  });
  assert.equal(asked.count, 1);
  assert.equal(store.kept_.size, 0);
  assert.equal(store.discarded, 1);
});
