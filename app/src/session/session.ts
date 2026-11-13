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
//   - pins: the edit set held for the next run, at most one edit a neuron,
//     in the order first pinned (pins.ts). pinAll(edits) adds them, each
//     replacing the edit on its layer and neuron in place, as one change;
//     unpinAll(layer, neurons) removes theirs, as one change; pin(edit) and
//     unpin(layer, neuron) are the same for one. A change that leaves the
//     set as it was notifies no one. Changing the pins changes nothing
//     shown of the run already made: a reply keeps the edits it was sent.
//     Neither a run nor an accept changes them: pins are kept throughout.
//   - The run shown: one message and seed, and two columns.
//       reply     the reply to steer from (left): streamed as text while it
//                 runs, with its count of tokens so far, then done with
//                 its tokens (branch.ts, ReplyToken,
//                 the last token's text ending with the answer's tail), how
//                 it ended, and its captures; its history, the ranged edits
//                 its tokens were made under; and `replyAt`, the step its
//                 own captures hold.
//       point     the branch point k (branch.ts), 0 to the reply's length.
//       inspection  the reply's neurons at k when k is not replyAt: an
//                 inspection's captures (maxTokens 0, the reply's first k
//                 tokens kept, its history as sent, branch.ts
//                 inspectEdits), running, done or failed; null when k is
//                 replyAt, where the reply's own captures are those neurons.
//       branch    the steered reply (right), from step `from`: the pins held
//                 when it was run, the edits it sent (branch.ts,
//                 branchEdits), and its own reply state, as the reply's;
//                 null until there is one.
//   - run(message, seed): only once the model is ready; otherwise it
//     throws. With a new message or seed, or no reply done to steer from, a
//     new reply runs, from the start with no edits, kept nothing (point 0,
//     replyAt 0, its history empty); then, when pins are held and it is
//     done, a branch from 0 with them, as the steered reply (waiting until
//     then). With the shown reply done and the same message and seed, a
//     branch runs from the point k: the reply's first k identifiers kept,
//     branchEdits(history, pins, k) sent, the steered reply replaced; it
//     waits for an inspection in flight first, so that inspection's answer
//     is not superseded. It runs with no pins too: the reply's own edits
//     cut at k, so after an accept it shows the reply with its pins
//     dropped from the point on, and before one, the reply again. Each reply streams its text as it comes, then is
//     "done", or "failed" with the stage and message. The branch does not
//     run after a failed reply.
//   - branchAt(k): sets the point, while the reply is done and no reply or
//     branch runs; otherwise, or for k not a whole number from 0 to the
//     reply's length, it throws, a page bug. When k is replyAt and the
//     reply's own captures were taken, no inspection; otherwise one runs,
//     superseding an earlier inspection.
//     The steered reply stays as it is, from its own `from`.
//   - accept(): only with the steered reply done, stopped included;
//     otherwise it throws. The
//     reply becomes branch.ts's accepted(reply, from, branch): its tokens,
//     the branch's ending and captures, its history the branch's edits,
//     replyAt and the point the branch's `from`; no branch. An inspection
//     in flight is dropped. When the steered reply's captures were not
//     taken (a stopped one has none, "not-taken"), its neurons at `from`
//     are read by an inspection there, as branchAt(from) reads them;
//     otherwise no inspection. The pins stay as they are.
//   - stop(): only while a reply or the steered reply runs or waits
//     (running(run)); otherwise it throws. What runs is ended through the
//     client's cancel and settles as "done" with end "cancelled", its
//     tokens and text until then, its captures "not-taken": a stopped
//     reply can be steered from, a stopped steered reply accepted. A
//     stopped reply from the start has no neurons at point 0 until a word
//     (or the start) is chosen, which inspects (branchAt); an accepted
//     stopped steered reply is inspected at once (accept). A
//     steered reply still waiting (behind the reply, or behind an
//     inspection) never starts: the branch is withdrawn (null). An
//     inspection in flight runs on: it computes no reply.
//   - A device that fails after it was ready means the worker has stopped
//     (protocol/start.ts reports its error event so), and it will answer
//     nothing more: the model, however far its load had come (a ready
//     model was held by that worker), and every reply or inspection running
//     or waiting, become "failed" at stage "worker" with the device's
//     error, rather
//     than showing progress that will never come or offering a Run nothing
//     would answer.
//   - The page sets no limit of its own on a reply or a branch: maxTokens
//     is NO_LIMIT,
//     so a reply ends when the model ends its turn ("stop") or the context
//     is full ("context"), as the model gives it. A long reply is ended by
//     stop(), or by running again, which supersedes it.
//
// Design
//   - Requests run one at a time: the engine runs one turn at a time, and
//     the client's latest-wins rule would cancel one sent beside another.
//     So the branch follows the reply, and a branch waits for an
//     inspection in flight; a word cannot be chosen while a reply runs.
//   - The conversation is one user message: the plain page holds no
//     history (slice 4's workbench decides more).
//   - The page holds every reply's identifiers and history, and sends them
//     whole with each run (branch.ts, Design); the engine reuses its cache
//     to the first position whose tokens or edits differ, so an inspection
//     or a branch computes only past that.
//   - A reply's tokens are collected as they stream, its text shown while
//     it runs; the tokens enter the state once, when it is done, so a long
//     reply does not copy its token list on every token.
//   - Captures: the reply's own, an inspection's and the branch's, the
//     whole store, 3.2 MiB each for Gemma 3 1B, 9.6 MiB at most; a new run
//     replaces all three.
//   - A rejection that is not the worker's (WorkerError) or a supersede
//     shows as failed at stage "page", with its message, never silently.

import type { Edit, RangedEdit } from "../engine/edits.ts";
import {
  type EngineClient,
  Superseded,
  WorkerError,
} from "../protocol/client.ts";
import type {
  Captures,
  DeviceNotice,
  Generated,
  Message,
} from "../protocol/messages.ts";
import {
  type ReplyToken,
  accepted,
  branchEdits,
  inspectEdits,
} from "./branch.ts";
import { withPins, withoutPins } from "./pins.ts";

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
  | { readonly kind: "waiting" } // the branch, behind the reply
  | {
      readonly kind: "running";
      readonly text: string;
      readonly tokens: number;
    }
  | {
      readonly kind: "done";
      readonly text: string;
      readonly tokens: readonly ReplyToken[];
      readonly end: Generated["end"];
      readonly captures: Captures;
    }
  | {
      readonly kind: "failed";
      readonly text: string;
      readonly failure: Failure;
    };

export interface Inspection {
  readonly at: number;
  readonly state:
    | { readonly kind: "running" }
    | { readonly kind: "done"; readonly captures: Captures }
    | { readonly kind: "failed"; readonly failure: Failure };
}

export interface Branch {
  readonly from: number;
  readonly pins: readonly Edit[];
  readonly edits: readonly RangedEdit[];
  readonly reply: ReplyState;
}

export interface Run {
  readonly message: string;
  readonly seed: number;
  readonly reply: ReplyState;
  readonly replyEdits: readonly RangedEdit[];
  readonly replyAt: number;
  readonly point: number;
  readonly inspection: Inspection | null;
  readonly branch: Branch | null;
}

export interface SessionState {
  readonly device: DeviceState;
  readonly model: ModelState;
  readonly pins: readonly Edit[];
  readonly run: Run | null;
}

// Whether a reply or a branch of the run is still to finish.
export function running(run: Run | null): boolean {
  if (run === null) return false;
  const live = (r: ReplyState | undefined) =>
    r !== undefined && (r.kind === "running" || r.kind === "waiting");
  return live(run.reply) || live(run.branch?.reply);
}

type Settled = "done" | "failed" | "superseded";

export class Session {
  readonly #client: EngineClient;
  readonly #listeners = new Set<() => void>();
  #state: SessionState = {
    device: { kind: "waiting" },
    model: { kind: "idle" },
    pins: [],
    run: null,
  };
  // The run shown, by number: answers to an earlier run change nothing.
  #runNumber = 0;
  // The branch shown and the inspection shown, by number, alike.
  #branchNumber = 0;
  #inspectionNumber = 0;
  // The inspection in flight, which a branch waits for.
  #inspecting: Promise<void> | null = null;

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

  pin(edit: Edit): void {
    this.pinAll([edit]);
  }

  pinAll(edits: readonly Edit[]): void {
    const { pins } = this.#state;
    const next = withPins(pins, edits);
    if (next !== pins) this.#set({ pins: next });
  }

  unpin(layer: number, neuron: number): void {
    this.unpinAll(layer, [neuron]);
  }

  unpinAll(layer: number, neurons: readonly number[]): void {
    const { pins } = this.#state;
    const kept = withoutPins(pins, layer, neurons);
    if (kept !== pins) this.#set({ pins: kept });
  }

  run(message: string, seed: number): void {
    if (this.#state.model.kind !== "ready") {
      throw new Error(`run() with the model ${this.#state.model.kind}`);
    }
    const pins = this.#state.pins;
    const shown = this.#state.run;
    if (
      shown !== null &&
      shown.message === message &&
      shown.seed === seed &&
      shown.reply.kind === "done"
    ) {
      void this.#branch(this.#runNumber, shown, shown.reply.tokens, pins);
      return;
    }
    const number = ++this.#runNumber;
    this.#inspecting = null;
    this.#set({
      run: {
        message,
        seed,
        reply: { kind: "running", text: "", tokens: 0 },
        replyEdits: [],
        replyAt: 0,
        point: 0,
        inspection: null,
        branch:
          pins.length === 0
            ? null
            : {
                from: 0,
                pins,
                edits: branchEdits([], pins, 0),
                reply: { kind: "waiting" },
              },
      },
    });
    void this.#fresh(number, message, seed, pins);
  }

  branchAt(k: number): void {
    const run = this.#state.run;
    if (run === null || run.reply.kind !== "done" || running(run)) {
      throw new Error("branchAt() with no finished reply, or one running");
    }
    const { tokens } = run.reply;
    if (!Number.isInteger(k) || k < 0 || k > tokens.length) {
      throw new Error(
        `branchAt(${k}) outside the reply's ${tokens.length} tokens`,
      );
    }
    const inspection = ++this.#inspectionNumber;
    if (k === run.replyAt && run.reply.captures.kind === "taken") {
      this.#inspecting = null;
      this.#set({ run: { ...run, point: k, inspection: null } });
      return;
    }
    this.#set({
      run: {
        ...run,
        point: k,
        inspection: { at: k, state: { kind: "running" } },
      },
    });
    this.#inspecting = this.#inspect(
      this.#runNumber,
      inspection,
      run,
      tokens,
      k,
    );
  }

  accept(): void {
    const run = this.#state.run;
    const branch = run?.branch;
    if (
      run === null ||
      branch === null ||
      branch === undefined ||
      branch.reply.kind !== "done" ||
      run.reply.kind !== "done"
    ) {
      throw new Error("accept() with no finished steered reply");
    }
    const line = accepted(
      { tokens: run.reply.tokens, edits: run.replyEdits },
      branch.from,
      { tokens: branch.reply.tokens, edits: branch.edits },
    );
    ++this.#inspectionNumber;
    ++this.#branchNumber;
    this.#inspecting = null;
    this.#set({
      run: {
        ...run,
        reply: {
          kind: "done",
          text: textOf(line.tokens),
          tokens: line.tokens,
          end: branch.reply.end,
          captures: branch.reply.captures,
        },
        replyEdits: line.edits,
        replyAt: branch.from,
        point: branch.from,
        inspection: null,
        branch: null,
      },
    });
    if (branch.reply.captures.kind !== "taken") this.branchAt(branch.from);
  }

  stop(): void {
    const run = this.#state.run;
    if (run === null || !running(run)) {
      throw new Error("stop() with no reply running or waiting");
    }
    const generating =
      run.reply.kind === "running" || run.branch?.reply.kind === "running";
    if (run.branch?.reply.kind === "waiting") {
      // The waiting branch never starts: #branch and #fresh find it gone.
      ++this.#branchNumber;
      this.#set({ run: { ...run, branch: null } });
    }
    if (generating && !this.#client.cancel()) {
      throw new Error("stop(): a reply runs with no generation pending");
    }
  }

  // A new reply from the start, then the branch from 0 when pins are held.
  async #fresh(
    number: number,
    message: string,
    seed: number,
    pins: readonly Edit[],
  ): Promise<void> {
    const settled = await this.#generate(message, seed, [], [], (reply) =>
      this.#setRun(number, (run) => ({ ...run, reply })),
    );
    if (settled !== "done") return;
    const run = this.#state.run;
    if (run === null || number !== this.#runNumber) return;
    // No branch waits when no pins were held, or once stop() withdrew it.
    if (run.reply.kind !== "done" || run.branch === null) return;
    await this.#branch(number, run, run.reply.tokens, pins);
  }

  // A branch from the run's point, replacing the steered reply.
  async #branch(
    number: number,
    run: Run,
    tokens: readonly ReplyToken[],
    pins: readonly Edit[],
  ): Promise<void> {
    const branchNumber = ++this.#branchNumber;
    const from = run.point;
    const edits = branchEdits(run.replyEdits, pins, from);
    const kept = tokens.slice(0, from).map((t) => t.token);
    const setBranch = (reply: ReplyState) =>
      this.#setRun(number, (shown) =>
        branchNumber === this.#branchNumber
          ? { ...shown, branch: { from, pins, edits, reply } }
          : shown,
      );
    setBranch({ kind: "waiting" });
    if (this.#inspecting !== null) await this.#inspecting;
    if (number !== this.#runNumber || branchNumber !== this.#branchNumber) {
      return;
    }
    await this.#generate(run.message, run.seed, kept, edits, setBranch);
  }

  // The reply's neurons at step k, its first k tokens kept.
  async #inspect(
    number: number,
    inspection: number,
    run: Run,
    tokens: readonly ReplyToken[],
    k: number,
  ): Promise<void> {
    const setState = (state: Inspection["state"]) =>
      this.#setRun(number, (shown) =>
        inspection === this.#inspectionNumber
          ? { ...shown, inspection: { at: k, state } }
          : shown,
      );
    try {
      const answer = await this.#client.generate({
        messages: messagesOf(run.message),
        kept: tokens.slice(0, k).map((t) => t.token),
        edits: inspectEdits(run.replyEdits),
        seed: run.seed,
        maxTokens: 0,
      });
      setState({ kind: "done", captures: answer.captures });
    } catch (error) {
      if (!(error instanceof Superseded)) {
        setState({ kind: "failed", failure: failureOf(error) });
      }
    }
  }

  // One generation, its state passed to `show` as it streams and settles.
  async #generate(
    message: string,
    seed: number,
    kept: readonly number[],
    edits: readonly RangedEdit[],
    show: (reply: ReplyState) => void,
  ): Promise<Settled> {
    const texts: string[] = [];
    let text = "";
    show({ kind: "running", text, tokens: 0 });
    try {
      const answer = await this.#client.generate(
        {
          messages: messagesOf(message),
          kept,
          edits,
          seed,
          maxTokens: NO_LIMIT,
        },
        (piece) => {
          texts.push(piece);
          text += piece;
          show({ kind: "running", text, tokens: texts.length });
        },
      );
      show(doneOf(answer, texts));
      return "done";
    } catch (error) {
      if (error instanceof Superseded) return "superseded";
      show({ kind: "failed", text, failure: failureOf(error) });
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
    const stopped = (reply: ReplyState): ReplyState =>
      reply.kind === "done" || reply.kind === "failed"
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
              reply: stopped(run.reply),
              inspection:
                run.inspection?.state.kind === "running"
                  ? {
                      at: run.inspection.at,
                      state: { kind: "failed", failure },
                    }
                  : run.inspection,
              branch:
                run.branch === null
                  ? null
                  : { ...run.branch, reply: stopped(run.branch.reply) },
            },
    });
  }

  // A load's news, unless the device's failure has already ended it.
  #setModel(model: ModelState): void {
    if (this.#state.device.kind === "failed") return;
    this.#set({ model });
  }

  // A change to the run `number`, while it is shown and the worker lives.
  #setRun(number: number, change: (run: Run) => Run): void {
    const run = this.#state.run;
    if (run === null || number !== this.#runNumber) return;
    if (this.#state.device.kind === "failed") return;
    const next = change(run);
    if (next !== run) this.#set({ run: next });
  }

  #set(change: Partial<SessionState>): void {
    this.#state = { ...this.#state, ...change };
    for (const listener of this.#listeners) listener();
  }
}

function messagesOf(message: string): readonly Message[] {
  return [{ role: "user", content: message }];
}

export function textOf(tokens: readonly ReplyToken[]): string {
  return tokens.map((t) => t.text).join("");
}

// A finished reply: each emitted identifier with the text it streamed, the
// tail on the last. A count that differs is the protocol broken, and throws.
function doneOf(answer: Generated, texts: readonly string[]): ReplyState {
  if (answer.reply.length !== texts.length) {
    throw new Error(
      `the answer names ${answer.reply.length} tokens, ${texts.length} streamed`,
    );
  }
  if (answer.tail !== "" && texts.length === 0) {
    throw new Error("the answer has a tail and no token to own it");
  }
  const last = texts.length - 1;
  const tokens = answer.reply.map((token, i) => ({
    token,
    text: i === last ? `${texts[i] ?? ""}${answer.tail}` : (texts[i] ?? ""),
  }));
  return {
    kind: "done",
    text: textOf(tokens),
    tokens,
    end: answer.end,
    captures: answer.captures,
  };
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
