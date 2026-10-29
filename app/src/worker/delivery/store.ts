// Axis L: where the verified model is kept. The store's interface, with no
// platform behind it: opfs-store.ts is the browser's.
//
// Contract
//   - A store holds at most one file per key. A key is complete when its
//     record is kept, and the record is written only after the bytes: a key
//     whose record exists holds the whole, verified file; nothing else does.
//   - writer(key) starts a new file for key. write(bytes) appends, in order.
//     commit(record) makes the file the key's and then keeps the record.
//     discard() removes what was written, and the key stays as it was.
//   - kept(key): the key's record, or null when the key is not complete.
//
// Design
//   - Kept apart from opfs-store.ts so that code using a store
//     (download.ts) depends on what a store does, not on the browser's
//     storage types.

// A kept file's record: where it came from and what it is.
export interface Kept {
  readonly url: string;
  readonly sizeBytes: number;
  readonly sha256: string;
}

export interface Writer {
  write(bytes: Uint8Array): Promise<void>;
  commit(record: Kept): Promise<void>;
  discard(): Promise<void>;
}

export interface Store {
  writer(key: string): Promise<Writer>;
  kept(key: string): Promise<Kept | null>;
}
