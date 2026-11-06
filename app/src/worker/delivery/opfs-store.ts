// Axis L: the browser's storage. Keeps the verified model in the Origin
// Private File System (OPFS), so it is downloaded once.
//
// Contract
//   - ModelStore is a Store (store.ts), and keeps its contract.
//   - On disk, a key is `<key>.gguf`, its bytes, and `<key>.json`, its
//     record ({url, sizeBytes, sha256}), written last.
//   - writer(key) starts `<key>.partial`; commit(record) closes it, moves it
//     to `<key>.gguf`, then writes the record; discard() removes the
//     partial. A partial left by a page closed mid-download is removed when
//     the store opens.
//   - kept(key): the record when both files exist, otherwise null.
//   - open(key): the complete file, by a synchronous access handle, which
//     only a worker may hold, to stream it into the engine.
//
// Design
//   - Built for one pinned model.
//   - A kept file is trusted by its record, which is written only after the
//     download matched its pinned size and SHA-256 (download.ts); it is not
//     hashed again at every load, about 2 s of the worker's time for 722 MB.
//   - move() is given the directory and the name: WebKit has only that
//     form.

import type { Kept, Store, Writer } from "./store.ts";

export class ModelStore implements Store {
  readonly #directory: FileSystemDirectoryHandle;

  static async open(): Promise<ModelStore> {
    const store = new ModelStore(await navigator.storage.getDirectory());
    await store.#removePartials();
    return store;
  }

  private constructor(directory: FileSystemDirectoryHandle) {
    this.#directory = directory;
  }

  async writer(key: string): Promise<Writer> {
    const partial = await this.#directory.getFileHandle(`${key}.partial`, {
      create: true,
    });
    const stream = await partial.createWritable();
    return {
      write: (bytes) => stream.write(bytes as Uint8Array<ArrayBuffer>),
      commit: async (record) => {
        await stream.close();
        await (partial as MovableHandle).move(this.#directory, this.name(key));
        const json = await this.#directory.getFileHandle(`${key}.json`, {
          create: true,
        });
        const out = await json.createWritable();
        await out.write(JSON.stringify(record));
        await out.close();
      },
      discard: async () => {
        await stream.abort();
        await this.#directory.removeEntry(`${key}.partial`);
      },
    };
  }

  async kept(key: string): Promise<Kept | null> {
    try {
      const handle = await this.#directory.getFileHandle(`${key}.json`);
      const record = JSON.parse(await (await handle.getFile()).text()) as Kept;
      await this.#directory.getFileHandle(this.name(key)); // the bytes too
      return record;
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return null;
      }
      throw error;
    }
  }

  // Opens the kept file for reading into the engine: a handle only a
  // worker may hold, closed by the caller however the load ends.
  async open(key: string): Promise<FileSystemSyncAccessHandle> {
    const handle = await this.#directory.getFileHandle(this.name(key));
    return handle.createSyncAccessHandle();
  }

  name(key: string): string {
    return `${key}.gguf`;
  }

  async #removePartials(): Promise<void> {
    const partials: string[] = [];
    for await (const name of this.#directory.keys()) {
      if (name.endsWith(".partial")) partials.push(name);
    }
    for (const name of partials) await this.#directory.removeEntry(name);
  }
}

// What the platform has that TypeScript's worker types do not yet declare:
// moving a file handle (Chrome, Safari).
interface MovableHandle extends FileSystemFileHandle {
  move(directory: FileSystemDirectoryHandle, name: string): Promise<void>;
}
