// Axis R: the engine's worker. Owns the engine and the model's file, and
// answers the page's requests (protocol/messages.ts). Runs off the page's
// thread, so the page stays responsive while the model loads and decodes.
//
// Contract
//   - On start: Engine.start, and the "device" notice with its outcome.
//   - load: ensureModel (delivery/download.ts) against config/model.json's
//     pin, progress "download"; open the kept file with a synchronous
//     access handle, Engine.load, progress "load"; close the handle however
//     the load ends; compile the chat template the engine reports. Done:
//     {contextTokens}. A file without a chat template is refused by name.
//   - generate: refused, "failed" at stage "generate", before load; render
//     the messages (worker/template/template.ts), refusing a TemplateError
//     by name at stage "template"; Engine.generate with the edits, the seed
//     and maxTokens, streaming "text"; done with its answer.
//   - cancel {target}: a load still downloading is aborted, answered
//     "failed" at stage "cancelled"; once it streams into the engine it
//     runs to its end (engine/engine.ts). A generation running is ended
//     through Engine.cancel, answered "done" with end "cancelled"; one not
//     yet started is answered so without running. Done: whether there was
//     one to end.
//   - Every request is answered once, finally (messages.ts); a failure's
//     stage and message cross with it, never a bare "failed".
//
// Design
//   - A plain Web Worker: no SharedArrayBuffer, so no cross-origin
//     isolation, which a static host such as GitHub Pages cannot set.
//   - Requests run one after another, in the order sent, each once the one
//     before has finished, since the engine runs one operation at a time;
//     a cancel acts at once. So a generation sent after a cancel waits for
//     the cancelled turn to end, rather than finding the engine busy.
//   - Ported from Charlotte e003048's web/worker.js, for Frizzle's three
//     requests.

import pin from "../../../config/model.json" with { type: "json" };
import type {
  CancelRequest,
  FromWorker,
  GenerateRequest,
  Generated,
  Request,
  WorkerFailure,
} from "../protocol/messages.ts";
import { DownloadError, ensureModel } from "./delivery/download.ts";
import { FetchError, openDownload } from "./delivery/fetch.ts";
import { ModelStore } from "./delivery/opfs-store.ts";
import { Engine, EngineError } from "./engine/engine.ts";
import {
  type Render,
  TemplateError,
  compileTemplate,
} from "./template/template.ts";

const post = (message: FromWorker): void => self.postMessage(message);

const engine = Engine.start((outcome) =>
  post(
    outcome.ok
      ? { kind: "device", ok: true, adapter: outcome.adapter }
      : { kind: "device", ok: false, error: outcome.error },
  ),
);

let render: Render | null = null;
let downloading: { id: number; controller: AbortController } | null = null;
let generating: number | null = null;
// Requests sent and not yet started, and those of them cancelled.
const queued = new Set<number>();
const cancelled = new Set<number>();
let queue: Promise<void> = Promise.resolve();

self.addEventListener("message", (event: MessageEvent<Request>) => {
  const request = event.data;
  if (request.kind === "cancel") {
    cancel(request);
    return;
  }
  queued.add(request.id);
  queue = queue.then(() => answer(request));
});

async function answer(request: Exclude<Request, CancelRequest>): Promise<void> {
  queued.delete(request.id);
  if (cancelled.delete(request.id)) {
    post(
      request.kind === "load"
        ? {
            kind: "failed",
            id: request.id,
            failure: { stage: "cancelled", message: "the load was cancelled" },
          }
        : {
            kind: "done",
            id: request.id,
            value: {
              end: "cancelled",
              tokens: 0,
              promptTokens: 0,
              reusedTokens: 0,
            },
          },
    );
    return;
  }
  try {
    const value =
      request.kind === "load"
        ? await load(request.id)
        : await generate(request);
    post({ kind: "done", id: request.id, value });
  } catch (error) {
    post({
      kind: "failed",
      id: request.id,
      failure: failureOf(request.kind, error),
    });
  }
}

async function load(id: number): Promise<{ contextTokens: number }> {
  const store = await ModelStore.open();
  const controller = new AbortController();
  downloading = { id, controller };
  let key: string;
  try {
    key = await ensureModel(
      pin,
      openDownload,
      store,
      (done, total) =>
        post({
          kind: "progress",
          id,
          progress: { phase: "download", done, total },
        }),
      controller.signal,
    );
  } finally {
    downloading = null;
  }
  const handle = await store.open(key);
  try {
    const loaded = await (
      await engine
    ).load(handle, (done, total) =>
      post({ kind: "progress", id, progress: { phase: "load", done, total } }),
    );
    if (loaded.chat === null) {
      throw new TemplateError("the model file carries no chat template");
    }
    render = compileTemplate(loaded.chat);
    return { contextTokens: loaded.contextTokens };
  } finally {
    handle.close();
  }
}

async function generate(request: GenerateRequest): Promise<Generated> {
  if (render === null) {
    throw new EngineError("generate", "no model is loaded: send load first");
  }
  const text = render(request.messages);
  generating = request.id;
  try {
    return await (
      await engine
    ).generate(text, request.edits, request.seed, request.maxTokens, (piece) =>
      post({ kind: "text", id: request.id, text: piece }),
    );
  } finally {
    generating = null;
  }
}

function cancel(request: CancelRequest): void {
  let ended = false;
  if (downloading?.id === request.target) {
    downloading.controller.abort(
      new EngineError("cancelled", "the load was cancelled"),
    );
    ended = true;
  } else if (generating === request.target) {
    void engine.then((e) => e.cancel());
    ended = true;
  } else if (queued.has(request.target)) {
    // Not yet started: answered "cancelled" when its turn comes.
    cancelled.add(request.target);
    ended = true;
  }
  post({ kind: "done", id: request.id, value: ended });
}

function failureOf(kind: string, error: unknown): WorkerFailure {
  if (error instanceof EngineError) {
    return error.code === undefined
      ? { stage: error.stage, message: error.message }
      : { stage: error.stage, message: error.message, code: error.code };
  }
  if (error instanceof DownloadError)
    return { stage: "download", message: error.message };
  if (error instanceof FetchError) {
    return { stage: "download", message: error.message, code: error.reason };
  }
  if (error instanceof TemplateError)
    return { stage: "template", message: error.message };
  return {
    stage: kind,
    message: error instanceof Error ? error.message : String(error),
  };
}
