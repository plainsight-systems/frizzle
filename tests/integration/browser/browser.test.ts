// Axis V: integration scenario 3 of 5, Browser, end to end (tests/README.md).
//
// Contract locked: in a real browser with WebGPU, the app as it ships
// fetches and verifies the pinned file, generates a reply, and, with
// neurons pinned in three layers, gives a steered reply from the same seed
// that differs from the reply shown beside it; and each pinned neuron,
// read back from the page, holds in the steered run exactly what its pin
// sent: every edit crosses the page, the worker and the module as sent, and
// every run's captures come back from the engine. Then, steering from a
// word: the steered reply keeps the reply's text to the word, each pin
// holds there what it sent, read against the reply's neurons at that word;
// and accepting it makes it the reply, its neurons at the point the
// steered reply's, bit for bit. Last, a steered reply that would run to the
// end of the context is stopped, and the stopped reply accepted: never a
// dead end.
//
// Inputs
//   - The app as it ships: app/ built by its own `npm run build` (with the
//     engine's module from CMake's wasm-release preset) into a temporary
//     directory, served from 127.0.0.1 by this file as static files, as a
//     static host serves them: no dev server.
//   - The pinned file from the local copy the oracle reads
//     (.cache/models/, scripts/fetch_model.py). The page asks for the
//     pin's URL, as it always does; the driver answers that one request
//     from a second local server, with the copy's bytes, and blocks every
//     other request that leaves 127.0.0.1. So the app's own fetch, OPFS,
//     hash check, worker, module and page all run, and nothing touches the
//     network: Hugging Face's CDN is not this scenario's to lock. A fresh
//     browser profile every run, so nothing is kept and the whole file is
//     fetched.
//   - Chrome for Testing at package.json's config.chrome, the build
//     puppeteer-core's version is made for (a bump of one is a bump of
//     both), headless, installed by `npm run install-browser` into
//     .cache/browsers/.
//   - The replay scenario's message and seed, "Name three colours of the
//     rainbow." and 7, and three pins, one of each kind, chosen in step 2's
//     code by running them natively (tools/frizzle with --pin) against two
//     criteria: they change the reply, and the steered reply still ends by
//     itself, "stop", well within the deadline. The captures arrive with a
//     run's end (runtime/engine.h), so the steered reply must end; the
//     replay's two set pins make the model repeat itself to the context
//     (32,768 tokens), and are not used. The scale pin is in the earliest
//     pinned layer: nothing edited runs before it, so its input in the
//     steered run is the reply's, bit for bit (the kernels are batch-
//     invariant, graph/graph.h), and its steered value is exactly
//     Math.fround(reply × k). Its k is not 1, which the edit kernel
//     leaves alone (edit/edit_kernel.h). If the model or its settings
//     change and the pins stop meeting the criteria, others are chosen,
//     said so in the commit.
//
// What is asserted, in order
//   1. The WebGPU status reads "Ready". A device failure fails here, in
//      the page's words.
//   2. After Load model, the model status reads "Ready", and the bytes it
//      downloaded are the pin's sizeBytes, asked for once. The worker keeps
//      and loads a file only once its size and SHA-256 match the pin
//      (worker/delivery/download.ts), so a ready model is a verified one.
//   3. After Run, the reply is done, ended "stop" (the model ended its
//      turn: the page sets no limit of its own), with text. Then the
//      steered reply, from the start and the same seed, is done, ended
//      "stop", with text other than the reply's.
//   3a. The pins are made as a person makes them without a mouse: Open
//      neurons, Go to the layer and neuron, then the inspector's Set,
//      Scale × or Zero (views/deck/Inspector.tsx), and Run from the
//      workbench's bar, which closes it; the deck's click selects through
//      the same card, and is not driven here. Nor are a dragged box, Set
//      on many, zoom and pan: they reach the engine as the same per-neuron
//      edits, in one change of the pin set, which the Pins suite locks
//      (tests/contracts/pins/); the box's cells, the Shape suite. A box
//      that maps the wrong cells on screen would still pass all three.
//   4. Each pin, read back from the page: Open neurons, Go to it, then the
//      inspector's "Neuron reply value" and "Neuron steered value", the
//      exact f32 in each one's <data value>:
//        - in the reply, a number: the run's captures came back;
//        - set to v: the steered value is exactly v, as the page writes it;
//        - zero: the steered value is exactly 0;
//        - scale by k: the steered value is exactly Math.fround(reply ×
//          k), the reply's being the value the page shows for it.
//   5. Steering from a word: the reply's WORD-th word pressed (its branch
//      point, after it), then Run from the workbench. The steered reply is
//      done, ended "stop"; its text begins with the reply's text up to
//      that word and is not the reply's. Each pin read back as in 4, at
//      that word: the reply's value is an inspection's (maxTokens 0, the
//      reply's tokens to the word kept), the steered value the branch's.
//      The scale pin's layer is the earliest pinned, and the tokens before
//      the point are the reply's own, so its input at the point is the
//      reply's there, bit for bit: Math.fround(reply × k) holds only if
//      the kept identifiers crossed whole, the edits started at the point
//      (not at step 0, nor later), and both runs captured at the point.
//   6. Accept: the reply's text becomes the steered reply's, the steered
//      column reads "Not steered", and each pin's reply value at the point
//      is exactly the steered value read in 5: the accepted reply's
//      neurons are the run that made it.
//   7. Stop: the replay's strong pin added (layer 12, neuron 4031, set
//      100), which makes the model repeat itself to the context, then Run
//      from the same point. While it runs the ending counts its tokens;
//      once it has run RUNAWAY tokens, Stop. The steered
//      reply is done, ended "cancelled", its text beginning with the
//      reply's to the point, and Accept is enabled. Accept: the reply's
//      text becomes the stopped reply's, the steered column reads "Not
//      steered", and the neurons at the point, read by the inspection the
//      page runs since a stopped reply has none of its own, hold each pin
//      as sent: the three of step 5 their values there (the scale pin's
//      layer is before the strong pin's, so its input is unchanged), the
//      strong pin 100. A pin that ends the reply by itself before RUNAWAY
//      tokens fails by name: another is chosen.
//   8. Nothing was asked of any host but the two local servers.
//
//   What the read-back checks, and what it does not. The capture's unit
//   test locks the copy and the reader on a synthetic plan; the edit
//   kernel's, that an edit writes its value; the oracle, that the model's
//   activations are right. What only this scenario runs is the whole way,
//   on the real model: the page packs each pin, the worker and engine.ts
//   pass it as four words, bindings.cpp reads it back, the engine applies
//   it and captures the layer after the edit, reads the store back, and the
//   neurons cross the module, the worker and the page to the value shown.
//   So these fail here, by name: a pin's kind, value, layer or neuron
//   garbled on the way in; a kept identifier dropped or reordered, or a
//   range's start misplaced (step 5's scale pin); the engine reading the wrong buffer back, or
//   none; a layer's slot misplaced in the module or the page; captures
//   dropped or not transferred. What still passes:
//     - a wrong value at a neuron nobody pinned: only pinned neurons are
//       read back;
//     - a capture taken at the wrong step: a set or zero pin holds at
//       every scoring step, so its value would match there too; only the
//       scale pin, whose reply value it is tied to, would likely fail;
//     - the reply's values themselves: nothing independent says what
//       they should be, so they are checked as numbers, and the scale
//       pin's reply value only through its steered value;
//     - a NaN's payload: the page writes every NaN as "NaN";
//     - a range's end: after one accept nothing branches again, so no
//       edit with an end crosses the boundary here; the replay suite locks
//       the engine's reading of ends, and both ends cross as two words of
//       one edit, packed alike;
//     - which words the page offers, and the mark and fading drawn: the
//       Branch suite locks the words, and nothing locks the drawing;
//     - where step 7's stop lands: the reply is stopped after at least
//       RUNAWAY tokens, at whatever token the cancel reaches the engine,
//       so its length is not checked; nor that a reply stopped before its
//       first token reads alike (the worker answers that case without
//       running, worker.ts);
//     - step 6 recomputes nothing: the accepted reply's neurons at the
//       point are the steered run's captures, carried over by the page
//       (session.ts, accept), so step 6 checks that bookkeeping, not that
//       an inspection of the accepted reply there would give them; the
//       replay suite's cases 6 and 7 lock the engine's side.
//   That these pins change this reply and let it end is a fact observed
//   for this file, message and seed, not a law: if the model, its
//   settings or the kernels change and they no longer do, this fails by
//   name, and others are chosen.
//
// Design
//   - Runs before every push, as the pre-push hook (scripts/pre_push.py),
//     and by hand; not in CI, which has no GPU.
//   - The page is driven by role and accessible name, as a person reads it
//     (App.tsx, views/prompt/PromptView.tsx and views/deck/Inspector.tsx
//     list them); it has no test hooks. The word pressed is found among
//     the buttons in status "Reply", where the page lists each word.
//   - No sleeps: each wait is on the page's state, within one deadline for
//     the whole run; a page that stops changing fails at the deadline.
//   - A missing prerequisite fails by name, and nothing is installed or
//     built behind the reader's back but the app's own build: the engine's
//     module (build/wasm-release/web/), app/node_modules, the model's local
//     copy, the browser at its pin.
//   - Its own npm project (package.json here), because Node and
//     TypeScript resolve a package from the importing file's directory
//     upward, and the app's packages are app/'s: the driver's packages
//     never enter the app's lockfile.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { type Server, createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { extname, join, normalize, sep } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { Browser, computeExecutablePath } from "@puppeteer/browsers";
import puppeteer, { type ElementHandle, type Page } from "puppeteer-core";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const APP = join(ROOT, "app");
const ENGINE = join(ROOT, "build", "wasm-release", "web", "frizzle.mjs");
const BROWSERS = join(ROOT, ".cache", "browsers");

const scenario = JSON.parse(
  readFileSync(new URL("package.json", import.meta.url), "utf8"),
) as { readonly config: { readonly chrome: string } };
const pin = JSON.parse(
  readFileSync(join(ROOT, "config", "model.json"), "utf8"),
) as { readonly id: string; readonly url: string; readonly sizeBytes: number };
const MODEL = join(ROOT, ".cache", "models", `${pin.id}.gguf`);

const MESSAGE = "Name three colours of the rainbow.";
const SEED = "7";
// One of each kind, the scale pin in the earliest layer (the header's
// Inputs). Natively, at this seed, they change the reply, which still ends
// "stop": 24 tokens, where the reply without them is 37.
const KIND_BUTTON = { set: "Set", scale: "Scale ×", zero: "Zero" } as const;
// The reply's word steered from in step 5, counting from 1.
const WORD = 4;
// Step 7's pin, the replay suite's strong one, and the tokens the steered
// reply must have run before Stop is pressed: enough that it is running,
// and well past where these pins' replies end by themselves.
const RUNAWAY_PIN = { layer: "12", neuron: "4031", value: "100" } as const;
const RUNAWAY = 60;
const PINS = [
  { layer: "2", neuron: "1000", kind: "scale", value: "-4" },
  { layer: "16", neuron: "3000", kind: "set", value: "8" },
  { layer: "24", neuron: "100", kind: "zero", value: null },
] as const;

// The whole run, under a minute on an Apple M3 Max. The runner's own
// timeout (package.json, 6 minutes) is longer, so a run that misses this
// deadline fails here, and the browser, the servers and the temporary
// directories are still cleaned up.
const DEADLINE_MS = 5 * 60 * 1000;

const TYPES: Readonly<Record<string, string>> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
};

function prerequisites(): string {
  if (!existsSync(join(APP, "node_modules"))) {
    assert.fail("app/node_modules is missing: run `npm ci` in app/.");
  }
  if (!existsSync(MODEL)) {
    assert.fail(
      `the model's local copy is missing (.cache/models/${pin.id}.gguf): ` +
        "run `python3 scripts/fetch_model.py`.",
    );
  }
  if (!existsSync(ENGINE)) {
    assert.fail(
      "the engine's module is not built (build/wasm-release/web/frizzle.mjs): " +
        "build CMake's wasm-release preset, target frizzle_web.",
    );
  }
  const chrome = computeExecutablePath({
    browser: Browser.CHROME,
    buildId: scenario.config.chrome,
    cacheDir: BROWSERS,
  });
  if (!existsSync(chrome)) {
    assert.fail(
      `Chrome for Testing ${scenario.config.chrome} is not installed: ` +
        "run `npm run install-browser` in tests/integration/browser/.",
    );
  }
  return chrome;
}

// The built app as static files: a path outside `root` is not served.
function serve(root: string): Promise<Server> {
  const server = createServer((request, response) => {
    const path = decodeURIComponent(
      new URL(request.url ?? "/", "http://host").pathname,
    );
    const file = normalize(join(root, path === "/" ? "index.html" : path));
    if (!file.startsWith(root + sep)) {
      response.writeHead(403).end();
      return;
    }
    readFile(file).then(
      (bytes) => {
        response.writeHead(200, {
          "Content-Type": TYPES[extname(file)] ?? "application/octet-stream",
        });
        response.end(bytes);
      },
      () => response.writeHead(404).end(),
    );
  });
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => resolve(server)),
  );
}

// The model's local copy, for the one request the driver sends here: its
// own bytes and length, whatever they are, so the app's checks judge them.
// It counts the requests it answers. Any origin may read it, as the pin's
// host allows.
function serveModel(): Promise<{
  readonly server: Server;
  readonly asked: { count: number };
}> {
  const asked = { count: 0 };
  const server = createServer((_request, response) => {
    asked.count++;
    response.writeHead(200, {
      "Content-Type": "application/octet-stream",
      "Content-Length": String(statSync(MODEL).size),
      "Access-Control-Allow-Origin": "*",
    });
    createReadStream(MODEL).pipe(response);
  });
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => resolve({ server, asked })),
  );
}

const aria = (role: string, name: string): string =>
  `::-p-aria([role="${role}"][name="${name}"])`;

function remaining(start: number): number {
  return Math.max(1, DEADLINE_MS - (Date.now() - start));
}

// The status named `name`, once its text matches `settled`; its text.
async function settledStatus(
  page: Page,
  name: string,
  settled: RegExp,
  start: number,
): Promise<{ readonly element: ElementHandle; readonly text: string }> {
  const element = await page.waitForSelector(aria("status", name), {
    timeout: remaining(start),
  });
  assert.ok(element !== null, `no status named "${name}"`);
  await page.waitForFunction(
    (el, pattern) => new RegExp(pattern).test(el.textContent ?? ""),
    { timeout: remaining(start), polling: "mutation" },
    element,
    settled.source,
  );
  return {
    element,
    text: await element.evaluate((el) => el.textContent ?? ""),
  };
}

// The inspector's neuron card for that neuron, by Go to.
async function goTo(page: Page, layer: string, neuron: string): Promise<void> {
  await page.locator(aria("textbox", "Go to layer")).fill(layer);
  await page.locator(aria("textbox", "Go to neuron")).fill(neuron);
  await page.locator(aria("button", "Go")).click();
  await page.waitForSelector(
    aria("heading", `Neuron #${neuron} · layer ${layer}`),
  );
}

// A value's exact text: its <data value>, or the status's words when it
// has none.
async function exactOf(page: Page, name: string): Promise<string> {
  return page.$eval(
    aria("status", name),
    (el) =>
      el.querySelector("data")?.getAttribute("value") ?? el.textContent ?? "",
  );
}

async function textOf(page: Page, name: string): Promise<string> {
  return page.$eval(aria("status", name), (el) => el.textContent ?? "");
}

// Each pin, read back in the workbench (the header's step 4): the reply's
// value a number, the steered value what the pin sent. The steered values,
// as the page writes them, in PINS' order.
async function readBack(page: Page, where: string): Promise<string[]> {
  const sent: string[] = [];
  for (const [index, p] of PINS.entries()) {
    const n = index + 1;
    await goTo(page, p.layer, p.neuron);
    const before = await exactOf(page, "Neuron reply value");
    const after = await exactOf(page, "Neuron steered value");
    const value = Number(before);
    assert.ok(
      before.trim() !== "" && Number.isFinite(value),
      `pin ${n}, ${where}: the reply's neurons did not come back (${before})`,
    );
    const expected =
      p.kind === "set"
        ? Number(p.value)
        : p.kind === "zero"
          ? 0
          : Math.fround(value * Number(p.value));
    assert.equal(
      after,
      Object.is(expected, -0) ? "-0" : String(expected), // as the page writes it
      `pin ${n} (layer ${p.layer}, neuron ${p.neuron}, ${p.kind}), ${where}: ` +
        `the steered run used ${after}, not what the pin sent`,
    );
    sent.push(after);
  }
  return sent;
}

test("the pinned model is fetched, runs, and pins steer its reply, from a word too", async () => {
  const start = Date.now();
  const chrome = prerequisites();
  const out = await mkdtemp(join(tmpdir(), "frizzle-app-"));
  const profile = await mkdtemp(join(tmpdir(), "frizzle-profile-"));
  execFileSync(
    "npm",
    ["run", "--silent", "build", "--", "--outDir", out, "--emptyOutDir"],
    { cwd: APP, stdio: "inherit" },
  );
  const server = await serve(out);
  const model = await serveModel();
  const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    userDataDir: profile,
    // A wait is one call into the page, which lasts as long as the page
    // takes: the deadline bounds it, not Puppeteer's default of 3 minutes.
    protocolTimeout: DEADLINE_MS,
  });
  try {
    const page = await browser.newPage();
    const app = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`;
    const copy = `http://127.0.0.1:${(model.server.address() as AddressInfo).port}/`;
    const elsewhere: string[] = [];
    await page.setRequestInterception(true);
    page.on("request", (request) => {
      const url = request.url();
      if (url === pin.url) void request.continue({ url: copy });
      else if (url.startsWith(app)) void request.continue();
      else {
        elsewhere.push(url);
        void request.abort("blockedbyclient");
      }
    });
    await page.goto(app);

    // 1. The GPU.
    const device = await settledStatus(
      page,
      "WebGPU",
      /^(Ready|Failed)/,
      start,
    );
    assert.match(device.text, /^Ready: /, device.text);

    // 2. The model: downloaded whole, verified, loaded.
    await page.locator(aria("button", "Load model")).click();
    const loaded = await settledStatus(page, "Model", /^(Ready|Failed)/, start);
    assert.match(loaded.text, /^Ready: downloaded /, loaded.text);
    const downloaded = await loaded.element.$eval("data", (d) =>
      d.getAttribute("value"),
    );
    assert.equal(Number(downloaded), pin.sizeBytes);
    assert.equal(model.asked.count, 1);

    // 3. One message, one seed, three pins in three layers, made in the
    // inspector and held on the prompt.
    await page.locator(aria("textbox", "Message")).fill(MESSAGE);
    await page.locator(aria("textbox", "Seed")).fill(SEED);
    await page.locator(aria("button", "Open neurons")).click();
    for (const p of PINS) {
      await goTo(page, p.layer, p.neuron);
      // A zero takes no value.
      if (p.value !== null) {
        await page.locator(aria("textbox", "Pin value")).fill(p.value);
      }
      await page.locator(aria("button", KIND_BUTTON[p.kind])).click();
      await page.waitForSelector(
        aria("button", `Unpin layer ${p.layer} neuron ${p.neuron}`),
        { timeout: remaining(start) },
      );
    }
    await page.locator(aria("button", "Run")).click();

    const stopped = /^Done: stop, [\d,]+ tokens$/;
    const reply = await settledStatus(
      page,
      "Reply ending",
      /^(Done|Failed)/,
      start,
    );
    assert.match(reply.text, stopped, reply.text);
    const replyText = await textOf(page, "Reply");
    assert.notEqual(replyText.trim(), "", "the reply is empty");

    // 3. The steered reply ends by itself, as other text.
    const steered = await settledStatus(
      page,
      "Steered reply ending",
      /^(Done|Failed)/,
      start,
    );
    assert.match(steered.text, stopped, steered.text);
    assert.notEqual(
      await textOf(page, "Steered reply"),
      replyText,
      "the pins no longer change this reply at this seed; choose others",
    );

    // 4. Each pin, read back: what its pin sent, in the steered run.
    await page.locator(aria("button", "Open neurons")).click();
    await readBack(page, "from the start");
    await page.locator(aria("button", "Close")).click();

    // 5. Steering from a word.
    const replyPane = await page.$(aria("status", "Reply"));
    assert.ok(replyPane !== null, 'no status named "Reply"');
    const word = (await replyPane.$$("button"))[WORD - 1];
    assert.ok(word !== undefined, `the reply has fewer than ${WORD} words`);
    const kept = await page.$eval(
      aria("status", "Reply"),
      (out, button) => {
        const range = document.createRange();
        range.setStart(out, 0);
        range.setEndAfter(button);
        return range.toString();
      },
      word,
    );
    await word.click();
    await page.locator(aria("button", "Open neurons")).click();
    await page.locator(aria("button", "Run")).click();
    const fromWord = await settledStatus(
      page,
      "Steered reply ending",
      /^(Done|Failed)/,
      start,
    );
    assert.match(fromWord.text, stopped, fromWord.text);
    const steeredText = await textOf(page, "Steered reply");
    assert.ok(
      steeredText.startsWith(kept),
      `the steered reply does not keep the reply's text to word ${WORD}: ` +
        JSON.stringify({ kept, steeredText }),
    );
    assert.notEqual(
      steeredText,
      replyText,
      `the pins no longer change this reply from word ${WORD}; choose others`,
    );
    await page.locator(aria("button", "Open neurons")).click();
    const sentAtWord = await readBack(page, `from word ${WORD}`);
    await page.locator(aria("button", "Close")).click();

    // 6. Accept: the steered reply becomes the reply, its neurons with it.
    await page.locator(aria("button", "Accept")).click();
    await settledStatus(page, "Steered reply ending", /^Not steered$/, start);
    assert.equal(await textOf(page, "Reply"), steeredText);
    await page.locator(aria("button", "Open neurons")).click();
    for (const [index, p] of PINS.entries()) {
      await goTo(page, p.layer, p.neuron);
      assert.equal(
        await exactOf(page, "Neuron reply value"),
        sentAtWord[index],
        `pin ${index + 1}: the accepted reply's neurons at the point are ` +
          "not the steered run's",
      );
    }

    // 7. Stop: a steered reply that would run to the context, stopped and
    // accepted.
    await page.locator(aria("button", "Close")).click();
    const keptAtPoint = await page.$eval(aria("status", "Reply"), (out) => {
      const mark = out.querySelector(".point");
      if (mark === null) return null;
      const range = document.createRange();
      range.setStart(out, 0);
      range.setEndBefore(mark);
      return range.toString();
    });
    assert.ok(keptAtPoint !== null, "the reply shows no branch point");
    await page.locator(aria("button", "Open neurons")).click();
    await goTo(page, RUNAWAY_PIN.layer, RUNAWAY_PIN.neuron);
    await page.locator(aria("textbox", "Pin value")).fill(RUNAWAY_PIN.value);
    await page.locator(aria("button", "Set")).click();
    await page.waitForSelector(
      aria(
        "button",
        `Unpin layer ${RUNAWAY_PIN.layer} neuron ${RUNAWAY_PIN.neuron}`,
      ),
      { timeout: remaining(start) },
    );
    await page.locator(aria("button", "Run")).click();
    const running = await settledStatus(
      page,
      "Steered reply ending",
      /^(Running|Done|Failed)/,
      start,
    );
    const past = await page.waitForFunction(
      (el, least) => {
        const text = el.textContent ?? "";
        const count = /^Running: ([\d,]+) tokens$/.exec(text)?.[1];
        if (count === undefined)
          return text.startsWith("Running") ? false : text;
        return Number(count.replaceAll(",", "")) >= least ? text : false;
      },
      { timeout: remaining(start), polling: "mutation" },
      running.element,
      RUNAWAY,
    );
    const runningText = String(await past.jsonValue());
    assert.match(
      runningText,
      /^Running: /,
      `the strong pin's reply ended by itself before ${RUNAWAY} tokens ` +
        `(${runningText}); choose another`,
    );
    await page.locator(aria("button", "Stop")).click();
    const halted = await settledStatus(
      page,
      "Steered reply ending",
      /^(Done|Failed)/,
      start,
    );
    assert.match(halted.text, /^Done: cancelled, [\d,]+ tokens$/, halted.text);
    const haltedText = await textOf(page, "Steered reply");
    assert.ok(
      haltedText.startsWith(keptAtPoint),
      "the stopped reply does not keep the reply's text to the point: " +
        JSON.stringify({ keptAtPoint, haltedText }),
    );
    await page.locator(aria("button", "Accept")).click();
    await settledStatus(page, "Steered reply ending", /^Not steered$/, start);
    assert.equal(await textOf(page, "Reply"), haltedText);
    await page.locator(aria("button", "Open neurons")).click();
    for (const [index, p] of PINS.entries()) {
      await goTo(page, p.layer, p.neuron);
      // The inspection the page runs at the point, once it has answered.
      await settledStatus(page, "Neuron reply value", /^(?!Waiting$)/, start);
      assert.equal(
        await exactOf(page, "Neuron reply value"),
        sentAtWord[index],
        `pin ${index + 1}: after accepting the stopped reply, its neurons ` +
          "at the point do not hold what the pin sent",
      );
    }
    await goTo(page, RUNAWAY_PIN.layer, RUNAWAY_PIN.neuron);
    assert.equal(
      await exactOf(page, "Neuron reply value"),
      RUNAWAY_PIN.value,
      "after accepting the stopped reply, the strong pin does not hold there",
    );
    await page.locator(aria("button", "Close")).click();

    assert.deepEqual(elsewhere, [], "the page asked another host");
  } finally {
    await browser.close();
    server.close();
    model.server.close();
    await rm(out, { recursive: true, force: true });
    await rm(profile, { recursive: true, force: true });
  }
});
