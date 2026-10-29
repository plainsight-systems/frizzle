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
//   - generate(text, edits, seed, maxTokens, onText): one turn over the
//     rendered text; onText receives each piece as it completes; resolves
//     with the module's answer (messages.ts's Generated), or rejects with
//     its refusal or failure. One turn at a time, as the engine allows.
//   - cancel(): the running turn ends, answered "cancelled".
//
// Algorithm, a call
//   Each call into the module gets a request number and a pending promise;
//   frizzleOnAnswer(request, json) settles it, frizzleOnText(request, text)
//   streams into it. Bytes for a call are copied into the module's memory
//   through a view made just then, and freed when the call returns, the
//   module having read them (bindings.cpp).
//
// Design and cost
//   - Edits cross as four 32-bit words each, [layer, neuron, kind, value's
//     bits], kind set 0, scale 1, zero 2; a layer, neuron or value the
//     engine cannot take is its to refuse (edit::check).
//   - A load is 2 + ceil(size / 16 MiB) calls, plus one if the front is
//     short; nothing crosses between page and worker but progress.
//   - The module and a promise per call: request 0 is the device's answer;
//     every other call takes the next number.
//   - Ported from Charlotte e003048's web/wasm_runtime.js and web/load.js,
//     without what Gemma 3 does not use: preflight, media, companions,
//     embedding, the diagnostic passes.

import createModule, { type FrizzleModule } from "@frizzle/engine";

import type { Edit } from "../../engine/edits.ts";
import type { Generated } from "../../protocol/messages.ts";
import type { ChatTemplate } from "../template/template.ts";

const FRONT_BYTES = 16 * 2 ** 20;
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
  // The module calls these by name (bindings.cpp).
  var frizzleOnAnswer: (request: number, answer: Answer) => void;
  var frizzleOnText: (request: number, text: string) => void;
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
  readonly #answers = new Map<number, (answer: Answer) => void>();
  readonly #texts = new Map<number, (text: string) => void>();
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
    globalThis.frizzleOnAnswer = (request, answer) => {
      const settle = engine.#answers.get(request);
      engine.#answers.delete(request);
      settle?.(answer);
    };
    globalThis.frizzleOnText = (request, text) =>
      engine.#texts.get(request)?.(text);
    engine.#answers.set(START, (answer) =>
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
        this.#call((request) =>
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
        await this.#call((request) =>
          this.#module._frizzle_load_chunk(request, offset, length),
        ),
      );
      onProgress?.(offset + length, size);
    }
    const finished = settled(
      "load",
      await this.#call((request) => this.#module._frizzle_load_finish(request)),
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
    edits: readonly Edit[],
    seed: number,
    maxTokens: number,
    onText: (text: string) => void,
  ): Promise<Generated> {
    const bytes = new TextEncoder().encode(text);
    const words = new Uint32Array(edits.length * 4);
    const value = new DataView(new ArrayBuffer(4));
    edits.forEach((e, i) => {
      value.setFloat32(0, e.value, true);
      words.set(
        [e.layer, e.neuron, KIND[e.kind], value.getUint32(0, true)],
        i * 4,
      );
    });
    const answer = await this.#withBytes(bytes, (textPointer) =>
      this.#withBytes(new Uint8Array(words.buffer), (editsPointer) =>
        this.#call(
          (request) =>
            this.#module._frizzle_generate(
              request,
              textPointer,
              bytes.length,
              editsPointer,
              edits.length,
              seed,
              maxTokens,
            ),
          onText,
        ),
      ),
    );
    const done = settled("generate", answer);
    return {
      end: done["end"] as Generated["end"],
      tokens: Number(done["tokens"]),
      promptTokens: Number(done["promptTokens"]),
      reusedTokens: Number(done["reusedTokens"]),
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
    onText?: (text: string) => void,
  ): Promise<Answer> {
    const request = this.#nextRequest++;
    const answered = new Promise<Answer>((resolve) =>
      this.#answers.set(request, resolve),
    );
    if (onText !== undefined) this.#texts.set(request, onText);
    start(request);
    return answered.finally(() => this.#texts.delete(request));
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

function describe(answer: Answer): string {
  return [answer.error, answer.subject].filter(Boolean).join(": ");
}

// The answer, or its refusal as an EngineError.
function settled(stage: string, answer: Answer): Answer {
  if (answer.ok) return answer;
  throw new EngineError(answer.stage ?? stage, describe(answer), answer.error);
}
