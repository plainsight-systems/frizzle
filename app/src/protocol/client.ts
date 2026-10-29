// Axis R: the page's side of the protocol (messages.ts).
//
// Contract
//   - new EngineClient(port, onDevice): `port` is the worker, or any
//     MessagePort, which delivers messages alike; onDevice receives the
//     "device" notice.
//   - load(onProgress) and generate(request, onText) send a request and
//     resolve with its "done" value, or reject with a WorkerError carrying
//     the failure, once. onProgress and onText receive the request's
//     progress and text, in order, before it settles.
//   - The latest generation wins (logical-overview.md, invariant 7):
//     calling generate while an earlier generation is pending sends a
//     cancel naming it, then the new request, and settles the earlier at
//     once, rejected with a Superseded error; its text and its answer, when
//     they arrive, are dropped, as is the cancel's own answer. So the page only ever shows the answer to its latest
//     request.
//   - A reply naming an id this client never sent is a broken protocol, and
//     throws, rather than being ignored.
//
// Design
//   - Ids count from 1, per client; one map of pending requests.
//   - No timers: a request settles when the worker answers, and the worker
//     answers every request (messages.ts).

import type { Edit } from "../engine/edits.ts";
import type {
  FromWorker,
  Generated,
  Loaded,
  Message,
  Progress,
  Request,
  WorkerFailure,
} from "./messages.ts";

export interface Port {
  postMessage(message: Request): void;
  addEventListener(
    type: "message",
    listener: (event: { readonly data: FromWorker }) => void,
  ): void;
  start?(): void;
}

export class WorkerError extends Error {
  readonly stage: string;
  readonly code: string | undefined;
  constructor(failure: WorkerFailure) {
    super(failure.message);
    this.name = "WorkerError";
    this.stage = failure.stage;
    this.code = failure.code;
  }
}

// An earlier generation, settled because a later one was sent.
export class Superseded extends Error {
  constructor() {
    super("a later generation was sent");
    this.name = "Superseded";
  }
}

export interface GenerateArguments {
  readonly messages: readonly Message[];
  readonly edits: readonly Edit[];
  readonly seed: number;
  readonly maxTokens: number;
}

interface Pending {
  readonly resolve: (value: unknown) => void;
  readonly reject: (error: Error) => void;
  readonly onProgress: ((progress: Progress) => void) | undefined;
  readonly onText: ((text: string) => void) | undefined;
}

export class EngineClient {
  readonly #port: Port;
  readonly #pending = new Map<number, Pending>();
  // Ids this client settled before the worker answered: their replies are
  // dropped, not a broken protocol.
  readonly #settled = new Set<number>();
  #nextId = 1;
  #generation: number | null = null;

  constructor(
    port: Port,
    onDevice: (device: FromWorker & { kind: "device" }) => void,
  ) {
    this.#port = port;
    port.addEventListener("message", ({ data }) => {
      if (data.kind === "device") onDevice(data);
      else this.#receive(data);
    });
    port.start?.();
  }

  load(onProgress?: (progress: Progress) => void): Promise<Loaded> {
    const id = this.#nextId++;
    const reply = this.#await<Loaded>(id, onProgress, undefined);
    this.#port.postMessage({ kind: "load", id });
    return reply;
  }

  generate(
    request: GenerateArguments,
    onText?: (text: string) => void,
  ): Promise<Generated> {
    const earlier = this.#generation;
    if (earlier !== null && this.#pending.has(earlier)) {
      this.#port.postMessage({
        kind: "cancel",
        id: this.#forget(this.#nextId++),
        target: earlier,
      });
      this.#settle(earlier)?.reject(new Superseded());
    }
    const id = this.#nextId++;
    this.#generation = id;
    const reply = this.#await<Generated>(id, undefined, onText);
    this.#port.postMessage({
      kind: "generate",
      id,
      messages: request.messages,
      edits: request.edits,
      seed: request.seed,
      maxTokens: request.maxTokens,
    });
    return reply;
  }

  #await<T>(
    id: number,
    onProgress: ((progress: Progress) => void) | undefined,
    onText: ((text: string) => void) | undefined,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.#pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        onProgress,
        onText,
      });
    });
  }

  // A request whose answer nobody waits for: its replies are dropped.
  #forget(id: number): number {
    this.#settled.add(id);
    return id;
  }

  #settle(id: number): Pending | undefined {
    const pending = this.#pending.get(id);
    this.#pending.delete(id);
    this.#settled.add(id);
    return pending;
  }

  #receive(reply: Exclude<FromWorker, { kind: "device" }>): void {
    const pending = this.#pending.get(reply.id);
    if (pending === undefined) {
      if (this.#settled.has(reply.id)) {
        if (reply.kind === "done" || reply.kind === "failed") {
          this.#settled.delete(reply.id);
        }
        return;
      }
      throw new Error(`the worker answered request ${reply.id}, never sent`);
    }
    switch (reply.kind) {
      case "progress":
        pending.onProgress?.(reply.progress);
        return;
      case "text":
        pending.onText?.(reply.text);
        return;
      case "done":
        this.#settle(reply.id);
        this.#settled.delete(reply.id);
        pending.resolve(reply.value);
        return;
      case "failed":
        this.#settle(reply.id);
        this.#settled.delete(reply.id);
        pending.reject(new WorkerError(reply.failure));
        return;
    }
  }
}
