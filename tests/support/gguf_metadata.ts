// Axis V: reading a GGUF's metadata from a test. Test support only.
//
// Contract
//   - chatTemplateOf(bytes): the chat template a GGUF carries and its BOS
//     and EOS tokens' text: tokenizer.chat_template, and the entries of
//     tokenizer.ggml.tokens at tokenizer.ggml.bos_token_id and
//     eos_token_id. A key missing or of another type, or a file that ends
//     inside its metadata, throws, naming it; nothing is guessed.
//   - Reads only the metadata, the GGUF v3 layout: "GGUF", a u32 version, a
//     u64 tensor count and a u64 entry count, then each entry as a key (a
//     u64 length and its bytes), a u32 type and the value, little-endian.
//
// Why here: the engine reads the model file in C++ (src/core/gguf/); the
// page gets the template from the engine (slice 3, step 4). The TypeScript
// suites need it from the committed tokenizer fixture, and this reads just
// that, rather than committing a copy of the publisher's template.

import type { ChatTemplate } from "../../app/src/worker/template/template.ts";

const STRING = 8;
const ARRAY = 9;
const U32 = 4;
// Bytes of each fixed-size value type, by type number.
const FIXED: ReadonlyMap<number, number> = new Map([
  [0, 1],
  [1, 1],
  [2, 2],
  [3, 2],
  [4, 4],
  [5, 4],
  [6, 4],
  [7, 1],
  [10, 8],
  [11, 8],
  [12, 8],
]);

type Value = string | number | readonly string[] | null;

class Reader {
  private at = 0;
  private readonly view: DataView;
  private readonly bytes: Uint8Array;

  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  private need(count: number): number {
    if (this.at + count > this.bytes.byteLength) {
      throw new Error("the GGUF ends inside its metadata");
    }
    const start = this.at;
    this.at += count;
    return start;
  }

  u32(): number {
    return this.view.getUint32(this.need(4), true);
  }

  u64(): number {
    const value = this.view.getBigUint64(this.need(8), true);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("a GGUF length past what a test reads");
    }
    return Number(value);
  }

  string(): string {
    const length = this.u64();
    const start = this.need(length);
    return new TextDecoder("utf-8", { fatal: true }).decode(
      this.bytes.subarray(start, start + length),
    );
  }

  skip(count: number): void {
    this.need(count);
  }

  // A value of `type`: strings, u32s and string arrays read; the rest
  // skipped, as null.
  value(type: number): Value {
    if (type === STRING) return this.string();
    if (type === U32) return this.u32();
    if (type === ARRAY) {
      const element = this.u32();
      const count = this.u64();
      if (element === STRING) {
        const out: string[] = [];
        for (let i = 0; i < count; i++) out.push(this.string());
        return out;
      }
      const size = FIXED.get(element);
      if (size === undefined)
        throw new Error(`an array of GGUF type ${element}`);
      this.skip(size * count);
      return null;
    }
    const size = FIXED.get(type);
    if (size === undefined) throw new Error(`GGUF value type ${type}`);
    this.skip(size);
    return null;
  }
}

function metadata(bytes: Uint8Array): Map<string, Value> {
  const read = new Reader(bytes);
  const magic = new TextDecoder().decode(bytes.subarray(0, 4));
  if (magic !== "GGUF") throw new Error("not a GGUF file");
  read.skip(4);
  read.u32(); // version
  read.u64(); // tensors
  const entries = read.u64();
  const out = new Map<string, Value>();
  for (let i = 0; i < entries; i++) {
    const key = read.string();
    out.set(key, read.value(read.u32()));
  }
  return out;
}

function stringOf(map: Map<string, Value>, key: string): string {
  const value = map.get(key);
  if (typeof value !== "string")
    throw new Error(`${key} missing or not a string`);
  return value;
}

function tokenText(map: Map<string, Value>, key: string): string {
  const id = map.get(key);
  const tokens = map.get("tokenizer.ggml.tokens");
  if (typeof id !== "number") throw new Error(`${key} missing or not a u32`);
  if (!Array.isArray(tokens)) throw new Error("tokenizer.ggml.tokens missing");
  const text: unknown = tokens[id];
  if (typeof text !== "string")
    throw new Error(`${key} ${id} is past the tokens`);
  return text;
}

export function chatTemplateOf(bytes: Uint8Array): ChatTemplate {
  const map = metadata(bytes);
  return {
    template: stringOf(map, "tokenizer.chat_template"),
    bosToken: tokenText(map, "tokenizer.ggml.bos_token_id"),
    eosToken: tokenText(map, "tokenizer.ggml.eos_token_id"),
  };
}
