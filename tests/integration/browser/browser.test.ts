// Axis V: integration scenario 3 of 5, Browser, end to end (tests/README.md).
//
// Contract locked: in a real browser with WebGPU, the app as it ships
// fetches and verifies the pinned file, generates a reply, and, with
// neurons pinned in three layers, gives an edited reply from the same seed
// that differs from the original shown beside it; and each pinned neuron,
// read back from the page, holds in the edited run exactly what its pin
// sent: every edit crosses the page, the worker and the module as sent, and
// every run's captures come back from the engine.
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
//     criteria: they change the reply, and the edited reply still ends by
//     itself, "stop", well within the deadline. The captures arrive with a
//     run's end (runtime/engine.h), so the edited reply must end; the
//     replay's two set pins make the model repeat itself to the context
//     (32,768 tokens), and are not used. The scale pin is in the earliest
//     pinned layer: nothing edited runs before it, so its input in the
//     edited run is the original run's, bit for bit (the kernels are batch-
//     invariant, graph/graph.h), and its edited value is exactly
//     Math.fround(original × k). Its k is not 1, which the edit kernel
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
//   3. After Run, the original reply is done, ended "stop" (the model
//      ended its turn: the page sets no limit of its own), with text. Then
//      the edited reply, from the same seed, is done, ended "stop", with
//      text other than the original's.
//   4. Each pin, read back from the page (PromptView.tsx's "Pin n original
//      value" and "Pin n edited value"):
//        - in the original, a number: the run's captures came back;
//        - set to v: the edited value is exactly v, as the page writes it;
//        - zero: the edited value is exactly 0;
//        - scale by k: the edited value is exactly Math.fround(original ×
//          k), the original being the value the page shows for it.
//   5. Nothing was asked of any host but the two local servers.
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
//   garbled on the way in; the engine reading the wrong buffer back, or
//   none; a layer's slot misplaced in the module or the page; captures
//   dropped or not transferred. What still passes:
//     - a wrong value at a neuron nobody pinned: only pinned neurons are
//       read back;
//     - a capture taken at the wrong step: a set or zero pin holds at
//       every scoring step, so its value would match there too; only the
//       scale pin, whose original it is tied to, would likely fail;
//     - the original's values themselves: nothing independent says what
//       they should be, so they are checked as numbers, and the scale
//       pin's original only through its edited value;
//     - a NaN's payload: the page writes every NaN as "NaN".
//   That these pins change this reply and let it end is a fact observed
//   for this file, message and seed, not a law: if the model, its
//   settings or the kernels change and they no longer do, this fails by
//   name, and others are chosen.
//
// Design
//   - Runs before every push, as the pre-push hook (scripts/pre_push.py),
//     and by hand; not in CI, which has no GPU.
//   - The page is driven by role and accessible name, as a person reads it
//     (views/prompt/PromptView.tsx lists them); it has no test hooks.
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
// "stop": 24 tokens, where the original is 37.
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

async function textOf(page: Page, name: string): Promise<string> {
  return page.$eval(aria("status", name), (el) => el.textContent ?? "");
}

test("the pinned model is fetched, runs, and an edit changes its reply", async () => {
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

    // 3. One message, one seed, two pins in two layers.
    await page.locator(aria("textbox", "Message")).fill(MESSAGE);
    await page.locator(aria("textbox", "Seed")).fill(SEED);
    for (const [index, p] of PINS.entries()) {
      const n = index + 1;
      await page.locator(aria("button", "Add pin")).click();
      await page.locator(aria("textbox", `Pin ${n} layer`)).fill(p.layer);
      await page.locator(aria("textbox", `Pin ${n} neuron`)).fill(p.neuron);
      await page.locator(aria("combobox", `Pin ${n} kind`)).fill(p.kind);
      // A zero takes no value: its field is disabled.
      if (p.value !== null) {
        await page.locator(aria("textbox", `Pin ${n} value`)).fill(p.value);
      }
    }
    await page.locator(aria("button", "Run")).click();

    const stopped = /^Done: stop, [\d,]+ tokens$/;
    const original = await settledStatus(
      page,
      "Original reply ending",
      /^(Done|Failed)/,
      start,
    );
    assert.match(original.text, stopped, original.text);
    const originalText = await textOf(page, "Original reply");
    assert.notEqual(originalText.trim(), "", "the original reply is empty");

    // 3. The edited reply ends by itself, as other text.
    const edited = await settledStatus(
      page,
      "Edited reply ending",
      /^(Done|Failed)/,
      start,
    );
    assert.match(edited.text, stopped, edited.text);
    assert.notEqual(
      await textOf(page, "Edited reply"),
      originalText,
      "the pins no longer change this reply at this seed; choose others",
    );

    // 4. Each pin, read back: what its pin sent, in the edited run.
    for (const [index, p] of PINS.entries()) {
      const n = index + 1;
      const before = await textOf(page, `Pin ${n} original value`);
      const after = await textOf(page, `Pin ${n} edited value`);
      const original = Number(before);
      assert.ok(
        before.trim() !== "" && Number.isFinite(original),
        `pin ${n}: the original run's captures did not come back (${before})`,
      );
      const sent =
        p.kind === "set"
          ? Number(p.value)
          : p.kind === "zero"
            ? 0
            : Math.fround(original * Number(p.value));
      assert.equal(
        after,
        Object.is(sent, -0) ? "-0" : String(sent), // as the page writes it
        `pin ${n} (layer ${p.layer}, neuron ${p.neuron}, ${p.kind}): the ` +
          `edited run used ${after}, not what the pin sent`,
      );
    }
    assert.deepEqual(elsewhere, [], "the page asked another host");
  } finally {
    await browser.close();
    server.close();
    model.server.close();
    await rm(out, { recursive: true, force: true });
    await rm(profile, { recursive: true, force: true });
  }
});
