// Axis R: the engine, behind the operations the worker offers. Every call
// into the module goes through here.
//
// Contract
//   - start(onDevice): instantiates the module and asks it for the device
//     (frizzle_start); onDevice receives the outcome once. Resolves with the
//     engine once the module is instantiated.
//   - load(handle, onProgress): streams a kept model file through the
//     module: begin with the file's first 16 MiB, read again at the length
//     the module asks for when that is short (its "bytesNeeded"); then each
//     16 MiB chunk in order, read from the file straight into the module's
//     chunk buffer; then finish. Resolves with {contextTokens, chat
//     template}; rejects with an EngineError carrying the module's stage
//     and words. The handle is the caller's, and is not closed here. Once
//     begun, a load runs to its end: the engine has no way to abandon one
//     half made, and a later load would be refused as busy.
//   - generate(text, kept, edits, seed, maxTokens, onToken): one turn over
//     the rendered text, then the kept reply tokens (a branch; none for a
//     run from the reply's start), with the ranged edits (edits.ts,
//     RangedEdit); maxTokens 0 inspects. onToken receives each emitted
//     token, its identifier and the text it completes, possibly ""; resolves
//     with the module's answer (messages.ts's Generated), the reply's
//     identifiers, its tail (bindings.cpp) and its captures included:
//     "taken" with each layer's slots, keys and query heads and the input's
//     position from the answer, and the store's words that came in the same
//     call (bindings.cpp); "not-taken"; or "unread" with the module's
//     reason. A "taken" answer rejects as a broken module, at stage
//     "generate", rather than resolving with captures that are not there,
//     when: no words came; their count is not the answer's "words"; it has
//     no layers; a layer lacks one of its six slots, or its keys or query
//     heads; or a slot is not whole numbers or ends past the words. Rejects
//     with its refusal or failure otherwise. One turn at a time, as the
//     engine allows.
//   - cancel(): the running turn ends, answered "cancelled".
//
// Algorithm, a call
//   Each call into the module gets a request number and a pending promise;
//   frizzleOnAnswer(request, json, words?) settles it, with the store's
//   words when the module sent them; frizzleOnText(request, text, token) streams
//   into it. Bytes for a call are copied into the module's memory
//   through a view made just then, and freed when the call returns, the
//   module having read them (bindings.cpp).
//
// Design and cost
//   - Edits cross as six 32-bit words each, [layer, neuron, kind, value's
//     bits, from, until], kind set 0, scale 1, zero 2, until 0xFFFFFFFF for
//     an open range (null in edits.ts); a layer, neuron, value or range the
//     engine cannot take is its to refuse (edit::check). Kept identifiers
//     cross as one 32-bit word each; one that names no token is the
//     engine's to refuse (Kept).
//   - A load is 2 + ceil(size / 16 MiB) calls, plus one if the front is
//     short; nothing crosses between page and worker but progress.
//   - The store arrives as a Float32Array the module sliced out of its
//     heap for this answer (bindings.cpp): this file's to hand on, not a
//     view of the heap, so it survives the heap growing. Checking its slots
//     is 26 × 6 comparisons a run; nothing is copied.
//   - Not tested: the crossing from bindings.cpp to here has no end-to-end
//     test since the browser scenario was dropped (tests/README.md); the
//     store's layout is locked in the engine (the kernels suite), and the
//     check above refuses an answer whose slots do not fit its words, but
//     nothing checks that each slot the module names is the one layout()
//     made.
//   - The module and a promise per call: request 0 is the device's answer;
//     every other call takes the next number.
//   - Only what Gemma 3 uses: no preflight, media, companions, embedding
//     or diagnostic passes.

import createModule, { type FrizzleModule } from "@frizzle/engine";

import type { Edit, RangedEdit } from "../../engine/edits.ts";
import type {
  Captures,
  Generated,
  LayerSlots,
  Slot,
} from "../../protocol/messages.ts";
import type { ChatTemplate } from "../template/template.ts";

const FRONT_BYTES = 16 * 2 ** 20;
const TO_THE_END = 0xffffffff; // src/core/edit/edit.h, kToTheEnd
const EDIT_WORDS = 6;
const START = 0;
const KIND: Readonly<Record<Edit["kind"], number>> = {
  set: 0,
  scale: 1,
  zero: 2,
};

// What the module answers, as bindings.cpp writes it.
interface Answer {
  readonly ok: boolean;
  readonly stage?: string;
  readonly error?: string;
  readonly subject?: string;
  readonly bytesNeeded?: number;
  readonly [field: string]: unknown;
}

declare global {
  // The module calls these by name (bindings.cpp); a run's store rides
  // with its answer.
  var frizzleOnAnswer: (
    request: number,
    answer: Answer,
    words?: Float32Array,
  ) => void;
  var frizzleOnText: (request: number, text: string, token: number) => void;
}

// An answer, and the store's words the module sent with it, if any.
interface Answered {
  readonly answer: Answer;
  readonly words: Float32Array | undefined;
}

export class EngineError extends Error {
  readonly stage: string;
  readonly code: string | undefined;
  constructor(stage: string, message: string, code?: string) {
    super(message);
    this.name = "EngineError";
    this.stage = stage;
    this.code = code;
  }
}

export interface LoadedModel {
  readonly contextTokens: number;
  readonly chat: ChatTemplate | null; // null: the file carries no template
}

export type DeviceOutcome =
  | { readonly ok: true; readonly adapter: string }
  | { readonly ok: false; readonly error: string };

export class Engine {
  readonly #module: FrizzleModule;
  readonly #answers = new Map<number, (answer: Answered) => void>();
  readonly #texts = new Map<number, (token: number, text: string) => void>();
  #nextRequest = START + 1;

  private constructor(module: FrizzleModule) {
    this.#module = module;
  }

  static async start(
    onDevice: (outcome: DeviceOutcome) => void,
  ): Promise<Engine> {
    const engine = new Engine(await createModule());
    // The module answers through these, by name (bindings.cpp); it calls
    // neither before frizzle_start.
    globalThis.frizzleOnAnswer = (request, answer, words) => {
      const settle = engine.#answers.get(request);
      engine.#answers.delete(request);
      settle?.({ answer, words });
    };
    globalThis.frizzleOnText = (request, text, token) =>
      engine.#texts.get(request)?.(token, text);
    engine.#answers.set(START, ({ answer }) =>
      onDevice(
        answer.ok
          ? { ok: true, adapter: String(answer["adapter"] ?? "") }
          : { ok: false, error: describe(answer) },
      ),
    );
    engine.#module._frizzle_start();
    return engine;
  }

  async load(
    handle: FileSystemSyncAccessHandle,
    onProgress?: (done: number, total: number) => void,
  ): Promise<LoadedModel> {
    const size = handle.getSize();
    let front = Math.min(size, FRONT_BYTES);
    let begun: Answer;
    for (;;) {
      begun = await this.#withBytes(read(handle, 0, front), (pointer) =>
        this.#ask((request) =>
          this.#module._frizzle_load_begin(request, pointer, front, size),
        ),
      );
      const needed = begun.bytesNeeded;
      if (
        begun.ok ||
        needed === undefined ||
        needed <= front ||
        needed > size
      ) {
        break;
      }
      front = needed;
    }
    settled("load", begun);
    const chunk = Number(begun["chunk"]);
    const chunkBytes = Number(begun["chunkBytes"]);
    for (let offset = 0; offset < size; offset += chunkBytes) {
      const length = Math.min(chunkBytes, size - offset);
      // A view made now: growing the heap detaches any view kept (WASM.1).
      const into = this.#module.HEAPU8.subarray(chunk, chunk + length);
      if (handle.read(into, { at: offset }) !== length) {
        throw new EngineError(
          "load",
          `the kept file ended before byte ${offset + length}`,
        );
      }
      settled(
        "load",
        await this.#ask((request) =>
          this.#module._frizzle_load_chunk(request, offset, length),
        ),
      );
      onProgress?.(offset + length, size);
    }
    const finished = settled(
      "load",
      await this.#ask((request) => this.#module._frizzle_load_finish(request)),
    );
    const chat = begun["chat"] as
      { template: string; bos: string; eos: string } | null | undefined;
    return {
      contextTokens: Number(finished["context"]),
      chat:
        chat === null || chat === undefined
          ? null
          : { template: chat.template, bosToken: chat.bos, eosToken: chat.eos },
    };
  }

  async generate(
    text: string,
    kept: readonly number[],
    edits: readonly RangedEdit[],
    seed: number,
    maxTokens: number,
    onToken: (token: number, text: string) => void,
  ): Promise<Generated> {
    const bytes = new TextEncoder().encode(text);
    const keptWords = Uint32Array.from(kept);
    const words = new Uint32Array(edits.length * EDIT_WORDS);
    const value = new DataView(new ArrayBuffer(4));
    edits.forEach((e, i) => {
      value.setFloat32(0, e.value, true);
      words.set(
        [
          e.layer,
          e.neuron,
          KIND[e.kind],
          value.getUint32(0, true),
          e.from,
          e.until ?? TO_THE_END,
        ],
        i * EDIT_WORDS,
      );
    });
    const { answer, words: store } = await this.#withBytes(
      bytes,
      (textPointer) =>
        this.#withBytes(new Uint8Array(keptWords.buffer), (keptPointer) =>
          this.#withBytes(new Uint8Array(words.buffer), (editsPointer) =>
            this.#call(
              (request) =>
                this.#module._frizzle_generate(
                  request,
                  textPointer,
                  bytes.length,
                  keptPointer,
                  kept.length,
                  editsPointer,
                  edits.length,
                  seed,
                  maxTokens,
                ),
              onToken,
            ),
          ),
        ),
    );
    const done = settled("generate", answer);
    return {
      end: done["end"] as Generated["end"],
      tokens: Number(done["tokens"]),
      promptTokens: Number(done["promptTokens"]),
      reusedTokens: Number(done["reusedTokens"]),
      reply: replyOf(done["reply"], Number(done["tokens"])),
      tail: String(done["tail"] ?? ""),
      captures: capturesOf(done["captures"], store),
    };
  }

  cancel(): void {
    this.#module._frizzle_cancel();
  }

  // One call: its number, its answer, and its text while it runs. The
  // module may answer before `start` returns, so the answer is awaited
  // through a promise registered first.
  #call(
    start: (request: number) => void,
    onText?: (token: number, text: string) => void,
  ): Promise<Answered> {
    const request = this.#nextRequest++;
    const answered = new Promise<Answered>((resolve) =>
      this.#answers.set(request, resolve),
    );
    if (onText !== undefined) this.#texts.set(request, onText);
    start(request);
    return answered.finally(() => this.#texts.delete(request));
  }

  // A call whose answer carries no store: a load stage's.
  async #ask(start: (request: number) => void): Promise<Answer> {
    return (await this.#call(start)).answer;
  }

  // `bytes` copied into the module's memory for `use`, and freed once the
  // call it makes has returned: the module reads them during that call.
  async #withBytes<T>(
    bytes: Uint8Array,
    use: (pointer: number) => Promise<T>,
  ): Promise<T> {
    const pointer = this.#module._malloc(Math.max(1, bytes.length));
    if (pointer === 0) {
      throw new EngineError(
        "memory",
        `${bytes.length} bytes would not fit in the module`,
      );
    }
    this.#module.HEAPU8.set(bytes, pointer);
    const pending = use(pointer);
    this.#module._free(pointer);
    return pending;
  }
}

function read(
  handle: FileSystemSyncAccessHandle,
  at: number,
  length: number,
): Uint8Array {
  const bytes = new Uint8Array(length);
  if (handle.read(bytes, { at }) !== length) {
    throw new EngineError(
      "load",
      `the kept file is shorter than ${at + length} bytes`,
    );
  }
  return bytes;
}

// The answer's "captures", with the store's words that came beside it, as
// messages.ts's Captures; a "taken" whose slots do not fit its words is a
// broken module, refused by name, never captures that are not there (the
// header's generate).
function capturesOf(field: unknown, words: Float32Array | undefined): Captures {
  const c = field as
    | {
        status?: string;
        words?: unknown;
        position?: unknown;
        layers?: unknown;
        reason?: string;
      }
    | undefined;
  switch (c?.status) {
    case "not-taken":
      return { kind: "not-taken" };
    case "unread":
      return { kind: "unread", reason: String(c.reason ?? "") };
    case "taken": {
      const broken = (why: string): EngineError =>
        new EngineError(
          "generate",
          `the module said its captures were taken but ${why}`,
        );
      if (words === undefined) throw broken("sent no words");
      if (words.length !== c.words) {
        throw broken(`sent ${words.length} words for ${String(c.words)}`);
      }
      if (!Array.isArray(c.layers) || c.layers.length === 0) {
        throw broken("named no layers");
      }
      const position = c.position;
      if (!isWhole(position)) {
        throw broken(`named no position (${String(position)})`);
      }
      const layers = c.layers.map((layer: unknown, l) =>
        layerOf(layer, words.length, (why) => broken(`layer ${l} ${why}`)),
      );
      return { kind: "taken", layers, words, position };
    }
    default:
      throw new EngineError(
        "generate",
        `the module's answer has no captures it names (${String(c?.status)})`,
      );
  }
}

// One layer of the answer's "captures": its six slots, each a pair [first,
// width] of whole numbers ending within the store's `size` words, and its
// keys and query heads, whole numbers; anything else is `broken`.
function layerOf(
  field: unknown,
  size: number,
  broken: (why: string) => EngineError,
): LayerSlots {
  const layer = (field ?? {}) as Record<string, unknown>;
  const slot = (
    name: Exclude<keyof LayerSlots, "keys" | "queryHeads">,
  ): Slot => {
    const pair = layer[name];
    if (!Array.isArray(pair) || pair.length !== 2) {
      throw broken(`has no ${name} slot`);
    }
    const [first, width] = pair as unknown[];
    if (!isWhole(first) || !isWhole(width)) {
      throw broken(`has a ${name} slot that is not whole numbers`);
    }
    if (first + width > size) {
      throw broken(`has a ${name} slot ending at ${first + width} of ${size}`);
    }
    return { first, width };
  };
  const count = (name: "keys" | "queryHeads"): number => {
    const value = layer[name];
    if (!isWhole(value)) throw broken(`has no ${name}`);
    return value;
  };
  return {
    neurons: slot("neurons"),
    residual: slot("residual"),
    attentionAdded: slot("attentionAdded"),
    mlpAdded: slot("mlpAdded"),
    weights: slot("weights"),
    stats: slot("stats"),
    keys: count("keys"),
    queryHeads: count("queryHeads"),
  };
}

function isWhole(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

// The answer's "reply": one identifier an emitted token. One missing, or a
// count that is not the answer's "tokens", is a broken module, refused by
// name.
function replyOf(field: unknown, tokens: number): readonly number[] {
  if (!Array.isArray(field) || field.length !== tokens) {
    throw new EngineError(
      "generate",
      `the module's answer has ${
        Array.isArray(field) ? `${field.length} reply identifiers` : "no reply"
      } for ${tokens} tokens`,
    );
  }
  return field.map(Number);
}

function describe(answer: Answer): string {
  return [answer.error, answer.subject].filter(Boolean).join(": ");
}

// The answer, or its refusal as an EngineError.
function settled(stage: string, answer: Answer): Answer {
  if (answer.ok) return answer;
  throw new EngineError(answer.stage ?? stage, describe(answer), answer.error);
}
