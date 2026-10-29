// Axis L: SHA-256 over a stream, so the downloaded file is verified as it
// arrives. The browser's crypto.subtle.digest takes its whole input at
// once, which for the model would mean holding 722 MB in memory.
//
// Contract
//   - new Sha256(), then update(bytes) any number of times, in order, then
//     hex(): the digest of every byte given, lowercase hex (FIPS 180-4). The
//     same bytes give the same digest however they are split into updates.
//   - After hex(), the hash is finished: update() or hex() again throws,
//     rather than returning a digest of something else.
//
// Algorithm
//   update tops up a partial 64-byte block first, then compresses whole
//   blocks straight from the input, and keeps the rest. hex pads with 0x80,
//   zeros and the message's length in bits as a big-endian 64-bit number,
//   then compresses the last blocks.
//
// Design and cost
//   - Ported from Charlotte e003048's web/sha256.js. One block's compression
//     is the whole cost of hashing a file; it uses locals and one reused
//     64-word schedule, allocating nothing an update.
//   - A byte count past 2^53 would lose precision; the model is 722 MB.

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));

export class Sha256 {
  readonly #state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
    0x1f83d9ab, 0x5be0cd19,
  ]);
  readonly #block = new Uint8Array(64);
  readonly #words = new Uint32Array(64);
  #blockLength = 0;
  #totalBytes = 0;
  #finished = false;

  update(bytes: Uint8Array): this {
    if (this.#finished) throw new Error("this SHA-256 is already finished");
    this.#take(bytes);
    return this;
  }

  hex(): string {
    if (this.#finished) throw new Error("this SHA-256 is already finished");
    const bits = this.#totalBytes * 8;
    const padding = new Uint8Array(
      (this.#blockLength < 56 ? 56 : 120) - this.#blockLength + 8,
    );
    padding[0] = 0x80;
    const view = new DataView(padding.buffer);
    view.setUint32(padding.length - 8, Math.floor(bits / 2 ** 32));
    view.setUint32(padding.length - 4, bits >>> 0);
    this.#take(padding);
    this.#finished = true;
    return Array.from(this.#state, (word) =>
      word.toString(16).padStart(8, "0"),
    ).join("");
  }

  // Top up a partial block, compress whole blocks straight from the input,
  // keep the rest. Padding counts no bytes toward the length: #take is
  // called with it only after the length is read.
  #take(bytes: Uint8Array): void {
    let offset = 0;
    this.#totalBytes += bytes.length;
    if (this.#blockLength > 0) {
      const take = Math.min(64 - this.#blockLength, bytes.length);
      this.#block.set(bytes.subarray(0, take), this.#blockLength);
      this.#blockLength += take;
      offset = take;
      if (this.#blockLength < 64) return;
      this.#compress(this.#block, 0);
      this.#blockLength = 0;
    }
    for (; offset + 64 <= bytes.length; offset += 64) {
      this.#compress(bytes, offset);
    }
    this.#block.set(bytes.subarray(offset), 0);
    this.#blockLength = bytes.length - offset;
  }

  // One 64-byte block; the whole cost of hashing a file.
  #compress(bytes: Uint8Array, offset: number): void {
    const w = this.#words;
    const s = this.#state;
    for (let i = 0; i < 16; i++) {
      const j = offset + i * 4;
      w[i] =
        ((bytes[j] ?? 0) << 24) |
        ((bytes[j + 1] ?? 0) << 16) |
        ((bytes[j + 2] ?? 0) << 8) |
        (bytes[j + 3] ?? 0);
    }
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15] ?? 0;
      const y = w[i - 2] ?? 0;
      const s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >>> 3);
      const s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >>> 10);
      w[i] = ((w[i - 16] ?? 0) + s0 + (w[i - 7] ?? 0) + s1) | 0;
    }
    let a = s[0] ?? 0;
    let b = s[1] ?? 0;
    let c = s[2] ?? 0;
    let d = s[3] ?? 0;
    let e = s[4] ?? 0;
    let f = s[5] ?? 0;
    let g = s[6] ?? 0;
    let h = s[7] ?? 0;
    for (let i = 0; i < 64; i++) {
      const t1 =
        (h +
          (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
          ((e & f) ^ (~e & g)) +
          (K[i] ?? 0) +
          (w[i] ?? 0)) |
        0;
      const t2 =
        ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
          ((a & b) ^ (a & c) ^ (b & c))) |
        0;
      h = g;
      g = f;
      f = e;
      e = (d + t1) | 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) | 0;
    }
    s[0] = (s[0] ?? 0) + a;
    s[1] = (s[1] ?? 0) + b;
    s[2] = (s[2] ?? 0) + c;
    s[3] = (s[3] ?? 0) + d;
    s[4] = (s[4] ?? 0) + e;
    s[5] = (s[5] ?? 0) + f;
    s[6] = (s[6] ?? 0) + g;
    s[7] = (s[7] ?? 0) + h;
  }
}
