// Axis V: the protocol suite (tests/README.md), the page's client.
//
// Contract locked (app/src/protocol/client.ts, messages.ts): every request
// settles once, with its result or a named refusal, after its progress or
// text in order, each token's text with its identifier; and the latest
// generation wins: sending one settles the
// earlier as Superseded, cancels it in the worker, and drops whatever the
// worker still sends for it, so the page only shows its latest request.
// A cancel ends the latest generation early with one cancel request, and
// the generation still settles with the worker's answer and its text.
//
// Inputs: a real MessageChannel (node:worker_threads), the client on one
// end. The test plays the worker on the other, answering as messages.ts
// says a worker answers: it stands in for the worker only at the port.

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { MessageChannel, type MessagePort } from "node:worker_threads";

import {
  EngineClient,
  type Port,
  Superseded,
  WorkerError,
} from "../../../app/src/protocol/client.ts";
import type {
  FromWorker,
  Request,
} from "../../../app/src/protocol/messages.ts";

// Every port a test opens is closed after it, passed or failed, so a client
// that never answers fails its test rather than holding the run open.
const open: MessagePort[] = [];
afterEach(() => {
  for (const port of open.splice(0)) port.close();
});

// The worker's end: the requests it has received, in order, and a way to
// answer.
function channel() {
  const { port1, port2 } = new MessageChannel();
  open.push(port1, port2);
  const received: Request[] = [];
  const waiting: ((request: Request) => void)[] = [];
  port2.on("message", (request: Request) => {
    const next = waiting.shift();
    if (next !== undefined) next(request);
    else received.push(request);
  });
  const nextRequest = (): Promise<Request> =>
    received.length > 0
      ? Promise.resolve(received.shift() as Request)
      : new Promise((resolve) => waiting.push(resolve));
  const send = (message: FromWorker): void => port2.postMessage(message);
  return { port: port1 as unknown as Port, nextRequest, send };
}

const conversation = {
  messages: [{ role: "user", content: "Hi" }] as const,
  kept: [11, 12],
  edits: [],
  seed: 7,
  maxTokens: 40,
};

test("a load's progress arrives in order, then its value", async () => {
  const { port, nextRequest, send } = channel();
  const client = new EngineClient(port, () => {});
  const seen: number[] = [];
  const loaded = client.load((p) => seen.push(p.done));
  const request = await nextRequest();
  assert.equal(request.kind, "load");
  send({
    kind: "progress",
    id: request.id,
    progress: { phase: "download", done: 1, total: 3 },
  });
  send({
    kind: "progress",
    id: request.id,
    progress: { phase: "load", done: 3, total: 3 },
  });
  send({ kind: "done", id: request.id, value: { contextTokens: 32768 } });
  assert.deepEqual(await loaded, { contextTokens: 32768 });
  assert.deepEqual(seen, [1, 3]);
});

test("a failure settles as a WorkerError naming its stage", async () => {
  const { port, nextRequest, send } = channel();
  const client = new EngineClient(port, () => {});
  const loaded = client.load();
  const request = await nextRequest();
  send({
    kind: "failed",
    id: request.id,
    failure: {
      stage: "download",
      message: "the bytes do not match",
      code: "sha256",
    },
  });
  await assert.rejects(
    loaded,
    (error: unknown) =>
      error instanceof WorkerError &&
      error.stage === "download" &&
      error.code === "sha256" &&
      error.message === "the bytes do not match",
  );
});

test("a generation's text streams in order, then its answer", async () => {
  const { port, nextRequest, send } = channel();
  const client = new EngineClient(port, () => {});
  let text = "";
  const tokens: number[] = [];
  const generated = client.generate(conversation, (piece, token) => {
    text += piece;
    tokens.push(token);
  });
  const request = await nextRequest();
  assert.equal(request.kind, "generate");
  assert.ok(request.kind === "generate");
  assert.deepEqual(request.messages, conversation.messages);
  assert.deepEqual(request.kept, conversation.kept);
  assert.equal(request.seed, 7);
  send({ kind: "text", id: request.id, text: "Hel", token: 101 });
  send({ kind: "text", id: request.id, text: "", token: 102 });
  send({ kind: "text", id: request.id, text: "lo.", token: 103 });
  const answer = { end: "stop", tokens: 3, promptTokens: 9, reusedTokens: 0 };
  send({ kind: "done", id: request.id, value: answer });
  assert.deepEqual(await generated, answer);
  assert.equal(text, "Hello.");
  assert.deepEqual(tokens, [101, 102, 103]);
});

test("a later generation supersedes, cancels and silences the earlier", async () => {
  const { port, nextRequest, send } = channel();
  const client = new EngineClient(port, () => {});
  let earlierText = "";
  let delivered: () => void = () => {};
  const firstPiece = new Promise<void>((resolve) => (delivered = resolve));
  const earlier = client.generate(conversation, (p) => {
    earlierText += p;
    delivered();
  });
  const first = await nextRequest();
  send({ kind: "text", id: first.id, text: "old ", token: 1 });
  // The client has shown the earlier's first piece before it is superseded.
  await firstPiece;

  let laterText = "";
  const later = client.generate(conversation, (p) => (laterText += p));
  await assert.rejects(
    earlier,
    (error: unknown) => error instanceof Superseded,
  );
  const cancel = await nextRequest();
  assert.equal(cancel.kind, "cancel");
  assert.ok(cancel.kind === "cancel");
  assert.equal(cancel.target, first.id);
  const second = await nextRequest();
  assert.equal(second.kind, "generate");

  // What the worker still sends for the earlier is dropped.
  send({ kind: "text", id: first.id, text: "stale", token: 2 });
  send({
    kind: "done",
    id: first.id,
    value: { end: "cancelled", tokens: 1, promptTokens: 9, reusedTokens: 8 },
  });
  send({ kind: "done", id: cancel.id, value: true });
  send({ kind: "text", id: second.id, text: "new", token: 3 });
  const answer = { end: "stop", tokens: 1, promptTokens: 9, reusedTokens: 8 };
  send({ kind: "done", id: second.id, value: answer });
  assert.deepEqual(await later, answer);
  assert.equal(earlierText, "old ");
  assert.equal(laterText, "new");
});

test("a cancelled generation settles with the worker's answer, its text kept", async () => {
  const { port, nextRequest, send } = channel();
  const client = new EngineClient(port, () => {});
  let text = "";
  let delivered: () => void = () => {};
  const firstPiece = new Promise<void>((resolve) => (delivered = resolve));
  const generated = client.generate(conversation, (p) => {
    text += p;
    delivered();
  });
  const request = await nextRequest();
  send({ kind: "text", id: request.id, text: "so far", token: 5 });
  await firstPiece;

  // Asked twice, one cancel is sent, naming the generation.
  assert.equal(client.cancel(), true);
  assert.equal(client.cancel(), true);
  const cancel = await nextRequest();
  assert.equal(cancel.kind, "cancel");
  assert.ok(cancel.kind === "cancel");
  assert.equal(cancel.target, request.id);

  const answer = {
    end: "cancelled",
    tokens: 1,
    promptTokens: 9,
    reusedTokens: 0,
  };
  send({ kind: "done", id: request.id, value: answer });
  // The cancel's own answer is dropped, not taken for a stray reply.
  send({ kind: "done", id: cancel.id, value: true });
  assert.deepEqual(await generated, answer);
  assert.equal(text, "so far");

  // Nothing pending: nothing is sent. The next request the worker sees is
  // the load, not a second cancel.
  assert.equal(client.cancel(), false);
  void client.load();
  assert.equal((await nextRequest()).kind, "load");
});

test("the device notice reaches onDevice", async () => {
  const { port, send } = channel();
  const notice = new Promise<FromWorker>((resolve) => {
    new EngineClient(port, resolve);
  });
  send({ kind: "device", ok: true, adapter: "Apple M3 Max" });
  assert.deepEqual(await notice, {
    kind: "device",
    ok: true,
    adapter: "Apple M3 Max",
  });
});
