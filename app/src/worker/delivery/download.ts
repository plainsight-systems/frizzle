// Axis L: how the model arrives. Downloads the pinned file into the store,
// hashing it as it streams, and keeps it only when it is the pinned file.
//
// Contract
//   - ensureModel(pin, open, store, onProgress, signal): resolves with the
//     store key of a kept copy of the pinned file, downloading it first if
//     the store has none. pin is config/model.json's {url, sizeBytes,
//     sha256}; the key is "sha256-" and the pin's hash, so a copy is kept
//     under what it is, not where it came from.
//   - A download is kept only when every byte has arrived, as many as the
//     pin says, and their SHA-256 is the pin's. Otherwise it is refused, a
//     DownloadError naming why, and nothing is kept (the writer is
//     discarded): the server states another size; the download ends early
//     or runs long; the bytes' hash is not the pin's. A fetch's own failure
//     (FetchError) and an abort also discard what was written.
//   - onProgress(done, total) follows the bytes received, total the pin's.
//
// Algorithm
//   1. kept(key) and the record's sha256 the pin's: done.
//   2. open the download; refuse a stated size other than the pin's.
//   3. Each chunk: refuse more bytes than the pin's; hash it; write it.
//   4. Refuse a short download, then a hash other than the pin's; commit
//      the record.
//
// Design and cost
//   - Ported from Charlotte e003048's web/download.js, for one pinned model.
//   - One pass over the bytes: each is hashed and written once, and no more
//     than one chunk is held.

import type { OpenDownload } from "./fetch.ts";
import { Sha256 } from "./sha256.ts";
import type { Store } from "./store.ts";

export interface Pin {
  readonly url: string;
  readonly sizeBytes: number;
  readonly sha256: string;
}

export class DownloadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DownloadError";
  }
}

export const keyOf = (pin: Pin): string => `sha256-${pin.sha256}`;

export async function ensureModel(
  pin: Pin,
  open: OpenDownload,
  store: Store,
  onProgress?: (done: number, total: number) => void,
  signal?: AbortSignal,
): Promise<string> {
  const key = keyOf(pin);
  const kept = await store.kept(key);
  if (kept !== null && kept.sha256 === pin.sha256) return key;

  signal?.throwIfAborted();
  const { totalSize, chunks } = await open(pin.url, signal);
  if (totalSize !== null && totalSize !== pin.sizeBytes) {
    throw new DownloadError(
      `the server's file is ${totalSize} bytes, but the pin says ${pin.sizeBytes}`,
    );
  }
  const writer = await store.writer(key);
  try {
    const hash = new Sha256();
    let received = 0;
    for await (const chunk of chunks) {
      signal?.throwIfAborted();
      if (received + chunk.length > pin.sizeBytes) {
        throw new DownloadError(
          `the server sent more than the pin's ${pin.sizeBytes} bytes`,
        );
      }
      hash.update(chunk);
      await writer.write(chunk);
      received += chunk.length;
      onProgress?.(received, pin.sizeBytes);
    }
    if (received !== pin.sizeBytes) {
      throw new DownloadError(
        `the download ended after ${received} of the pin's ${pin.sizeBytes} bytes`,
      );
    }
    const sha256 = hash.hex();
    if (sha256 !== pin.sha256) {
      throw new DownloadError(
        "the downloaded file does not match the pin's SHA-256",
      );
    }
    await writer.commit({ url: pin.url, sizeBytes: received, sha256 });
    return key;
  } catch (error) {
    await writer.discard();
    throw error;
  }
}
