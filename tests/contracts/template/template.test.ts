// Axis V: the chat template suite (tests/README.md).
//
// Contract locked (app/src/worker/template/template.ts): Gemma 3's chat
// template, from the model file, renders a conversation as the model was
// trained to read it, byte for byte:
//   - each of four conversations as Python's jinja2, configured as Hugging
//     Face transformers configures it, renders it (the frozen reference);
//   - one user message as <bos><start_of_turn>user\n{message, trimmed}
//     <end_of_turn>\n<start_of_turn>model\n;
//   - an assistant turn as the model's turn;
//   - a system message as the first user turn's opening, then a blank line;
// and refuses, as a TemplateError naming why: turns that do not alternate
// (the template's own message), and a template that does not parse.
//
// Inputs: the template and its BOS and EOS text from the committed
// tokenizer fixture (tests/fixtures/tokenizer/, the pinned model's
// tokenizer entries); the reference hashes in tests/fixtures/template/,
// frozen from Charlotte. No model file, no network.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  type ChatTemplate,
  type Message,
  TemplateError,
  compileTemplate,
} from "../../../app/src/worker/template/template.ts";
import { chatTemplateOf } from "../../support/gguf_metadata.ts";

const fixtures = new URL("../../fixtures/", import.meta.url);

const gemma: ChatTemplate = chatTemplateOf(
  readFileSync(new URL("tokenizer/gemma-3-1b-it-q4_0.gguf", fixtures)),
);

interface Case {
  readonly name: string;
  readonly messages: readonly Message[];
  readonly sha256: string;
}
const reference = JSON.parse(
  readFileSync(new URL("template/gemma-3-1b-it-q4_0.json", fixtures), "utf8"),
) as { readonly cases: readonly Case[] };

const sha256 = (text: string): string =>
  createHash("sha256").update(text, "utf8").digest("hex");

test("the fixture is Gemma 3's: <bos>, and <end_of_turn> as its end", () => {
  assert.equal(gemma.bosToken, "<bos>");
  assert.equal(gemma.eosToken, "<end_of_turn>");
});

test("every reference conversation renders as jinja2 renders it", () => {
  const render = compileTemplate(gemma);
  assert.equal(reference.cases.length, 4);
  for (const { name, messages, sha256: expected } of reference.cases) {
    const rendered = render(messages);
    assert.equal(sha256(rendered), expected, `${name} rendered:\n${rendered}`);
  }
});

test("one user message: <bos>, the turn trimmed, then the model's turn opens", () => {
  const render = compileTemplate(gemma);
  assert.equal(
    render([
      { role: "user", content: "  Name three colours of the rainbow.\n" },
    ]),
    "<bos><start_of_turn>user\nName three colours of the rainbow.<end_of_turn>\n" +
      "<start_of_turn>model\n",
  );
});

test("an assistant turn is the model's turn", () => {
  const render = compileTemplate(gemma);
  assert.equal(
    render([
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello. " },
      { role: "user", content: "Again?" },
    ]),
    "<bos><start_of_turn>user\nHi<end_of_turn>\n" +
      "<start_of_turn>model\nHello.<end_of_turn>\n" +
      "<start_of_turn>user\nAgain?<end_of_turn>\n" +
      "<start_of_turn>model\n",
  );
});

test("a system message opens the first user turn, then a blank line", () => {
  const render = compileTemplate(gemma);
  assert.equal(
    render([
      { role: "system", content: "Be brief." },
      { role: "user", content: "Why is the sky blue?" },
    ]),
    "<bos><start_of_turn>user\nBe brief.\n\nWhy is the sky blue?<end_of_turn>\n" +
      "<start_of_turn>model\n",
  );
});

test("turns that do not alternate are refused with the template's message", () => {
  const render = compileTemplate(gemma);
  assert.throws(
    () =>
      render([
        { role: "user", content: "a" },
        { role: "user", content: "b" },
      ]),
    (error: unknown) =>
      error instanceof TemplateError && /alternate/.test(error.message),
  );
});

test("a template that does not parse is refused by name", () => {
  assert.throws(
    () => compileTemplate({ ...gemma, template: "{% if %}" }),
    (error: unknown) =>
      error instanceof TemplateError && /does not parse/.test(error.message),
  );
});
