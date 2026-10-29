// Axis H (session): what the page is doing: the GPU, the model, and the
// run it shows. Plain TypeScript; the view reads it.
//
// Contract
//   - new Session(start): start(onDevice) is startEngine
//     (protocol/start.ts), called once, here; the session holds the client.
//   - snapshot(): the state, a new object on every change and the same one
//     between changes, so React's useSyncExternalStore can read it.
//     subscribe(listener) calls the listener after each change and
//     returns its unsubscribe.
//   - device: "waiting" until the worker's notice; then "ready" with the
//     adapter, or "failed" with the browser's or the engine's words.
//   - model: "idle" until load(); "starting" until the worker's first
//     progress; "downloading" or "loading" with bytes done and total, as
//     the worker reports them; then "ready", with the
//     context in tokens and `downloaded`, the bytes this load downloaded
//     (null when the file was already kept), or "failed" with the
//     failure's stage and message.
//   - load(): only from "idle" or "failed", once the device is ready;
//     otherwise it throws, a page bug, never a state.
//   - run(message, edits, seed): only once the model is ready; otherwise it
//     throws. The run replaces the one shown. The original reply runs
//     first, with no edits; then, when edits were given and the original
//     is done, the edited reply, with the same message and seed. With no
//     edits there is no edited reply (`edited` is null). Each reply streams
//     its text as it comes, then is "done" with the engine's answer (how it
//     ended, its tokens), or "failed" with the stage and message. The
//     edited reply does not run after a failed original.
//   - A run started while one is in flight supersedes it: the client
//     cancels the earlier generation in the worker (protocol/client.ts),
//     and the earlier run's edited reply never starts.
//   - A device that fails after it was ready means the worker has stopped
//     (protocol/start.ts reports its error event so), and it will answer
//     nothing more: the model, however far its load had come (a ready
//     model was held by that worker), and every reply running or waiting,
//     become "failed" at stage "worker" with the device's error, rather
//     than showing progress that will never come or offering a Run nothing
//     would answer.
//   - The page sets no limit of its own on a reply: maxTokens is NO_LIMIT,
//     so a reply ends when the model ends its turn ("stop") or the context
//     is full ("context"), as the model gives it. A long reply is ended by
//     running again, which supersedes it.
//
// Design
//   - The two replies run one after the other: the engine runs one turn at
//     a time, and the client's latest-wins rule would cancel the original
//     if the edited were sent beside it.
//   - The conversation is one user message: the plain page holds no
//     history (slice 4's workbench decides more).
//   - A rejection that is not the worker's (WorkerError) or a supersede
//     shows as failed at stage "page", with its message, never silently.

import type { Edit } from "../engine/edits.ts";
import {
  type EngineClient,
  Superseded,
  WorkerError,
} from "../protocol/client.ts";
import type { DeviceNotice, Generated } from "../protocol/messages.ts";

// The engine's own "no other limit" (policy.h, TurnPolicy::max_tokens): the
// largest count a 32-bit word carries, which the context bounds first.
export const NO_LIMIT = 2 ** 32 - 1;

export interface Failure {
  readonly stage: string;
  readonly message: string;
}

export type DeviceState =
  | { readonly kind: "waiting" }
  | { readonly kind: "ready"; readonly adapter: string }
  | { readonly kind: "failed"; readonly error: string };

export type ModelState =
  | { readonly kind: "idle" }
  | { readonly kind: "starting" }
  | {
      readonly kind: "downloading" | "loading";
      readonly done: number;
      readonly total: number;
    }
  | {
      readonly kind: "ready";
      readonly contextTokens: number;
      readonly downloaded: number | null;
    }
  | { readonly kind: "failed"; readonly failure: Failure };

export type ReplyState =
  | { readonly kind: "waiting" } // the edited reply, behind the original
  | { readonly kind: "running"; readonly text: string }
  | { readonly kind: "done"; readonly text: string; readonly answer: Generated }
  | {
      readonly kind: "failed";
      readonly text: string;
      readonly failure: Failure;
    };

export interface Run {
  readonly message: string;
  readonly seed: number;
  readonly edits: readonly Edit[];
  readonly original: ReplyState;
  readonly edited: ReplyState | null;
}

export interface SessionState {
  readonly device: DeviceState;
  readonly model: ModelState;
  readonly run: Run | null;
}

type Which = "original" | "edited";

export class Session {
  readonly #client: EngineClient;
  readonly #listeners = new Set<() => void>();
  #state: SessionState = {
    device: { kind: "waiting" },
    model: { kind: "idle" },
    run: null,
  };
  // The run shown, by number: replies of an earlier run change nothing.
  #runNumber = 0;

  constructor(
    start: (onDevice: (notice: DeviceNotice) => void) => EngineClient,
  ) {
    this.#client = start((notice) => this.#onDevice(notice));
  }

  // Arrow properties, so React may hold them unbound.
  readonly snapshot = (): SessionState => this.#state;

  readonly subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  load(): void {
    const { device, model } = this.#state;
    if (
      device.kind !== "ready" ||
      !(model.kind === "idle" || model.kind === "failed")
    ) {
      throw new Error(
        `load() with the device ${device.kind} and the model ${model.kind}`,
      );
    }
    let downloaded: number | null = null;
    this.#set({ model: { kind: "starting" } });
    this.#client
      .load(({ phase, done, total }) => {
        if (phase === "download") downloaded = done;
        this.#setModel({
          kind: phase === "download" ? "downloading" : "loading",
          done,
          total,
        });
      })
      .then(
        ({ contextTokens }) =>
          this.#setModel({ kind: "ready", contextTokens, downloaded }),
        (error: unknown) =>
          this.#setModel({ kind: "failed", failure: failureOf(error) }),
      );
  }

  run(message: string, edits: readonly Edit[], seed: number): void {
    if (this.#state.model.kind !== "ready") {
      throw new Error(`run() with the model ${this.#state.model.kind}`);
    }
    const number = ++this.#runNumber;
    this.#set({
      run: {
        message,
        seed,
        edits,
        original: { kind: "running", text: "" },
        edited: edits.length > 0 ? { kind: "waiting" } : null,
      },
    });
    void this.#replies(number, message, edits, seed);
  }

  async #replies(
    number: number,
    message: string,
    edits: readonly Edit[],
    seed: number,
  ): Promise<void> {
    const original = await this.#reply(number, "original", message, [], seed);
    if (original !== "done" || edits.length === 0) return;
    if (number !== this.#runNumber) return;
    this.#setReply(number, "edited", { kind: "running", text: "" });
    await this.#reply(number, "edited", message, edits, seed);
  }

  // One reply, streamed into the run `number`; how it settled.
  async #reply(
    number: number,
    which: Which,
    message: string,
    edits: readonly Edit[],
    seed: number,
  ): Promise<"done" | "failed" | "superseded"> {
    let text = "";
    try {
      const answer = await this.#client.generate(
        {
          messages: [{ role: "user", content: message }],
          edits,
          seed,
          maxTokens: NO_LIMIT,
        },
        (piece) => {
          text += piece;
          this.#setReply(number, which, { kind: "running", text });
        },
      );
      this.#setReply(number, which, { kind: "done", text, answer });
      return "done";
    } catch (error) {
      if (error instanceof Superseded) return "superseded";
      this.#setReply(number, which, {
        kind: "failed",
        text,
        failure: failureOf(error),
      });
      return "failed";
    }
  }

  #onDevice(notice: DeviceNotice): void {
    if (notice.ok) {
      this.#set({ device: { kind: "ready", adapter: notice.adapter } });
      return;
    }
    // The worker answers nothing more: what waits on it fails, by name.
    const failure: Failure = { stage: "worker", message: notice.error };
    const { model, run } = this.#state;
    const stopped = (reply: ReplyState | null): ReplyState | null =>
      reply === null || reply.kind === "done" || reply.kind === "failed"
        ? reply
        : {
            kind: "failed",
            text: reply.kind === "running" ? reply.text : "",
            failure,
          };
    this.#set({
      device: { kind: "failed", error: notice.error },
      model:
        model.kind === "idle" || model.kind === "failed"
          ? model
          : { kind: "failed", failure },
      run:
        run === null
          ? null
          : {
              ...run,
              original: stopped(run.original) as ReplyState,
              edited: stopped(run.edited),
            },
    });
  }

  // A load's news, unless the device's failure has already ended it.
  #setModel(model: ModelState): void {
    if (this.#state.device.kind === "failed") return;
    this.#set({ model });
  }

  #setReply(number: number, which: Which, reply: ReplyState): void {
    const run = this.#state.run;
    if (run === null || number !== this.#runNumber) return;
    if (this.#state.device.kind === "failed") return;
    this.#set({ run: { ...run, [which]: reply } });
  }

  #set(change: Partial<SessionState>): void {
    this.#state = { ...this.#state, ...change };
    for (const listener of this.#listeners) listener();
  }
}

function failureOf(error: unknown): Failure {
  if (error instanceof WorkerError) {
    return { stage: error.stage, message: error.message };
  }
  return {
    stage: "page",
    message: error instanceof Error ? error.message : String(error),
  };
}
