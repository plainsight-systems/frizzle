// Axis V: the branch suite (tests/README.md).
//
// Contract locked (app/src/session/branch.ts): a reply reads as words, each
// with the branch point after it; a branch sends the reply's own edits cut
// at its point and the pins held now from it, joined where one pin holds
// on unchanged; accepting a branch makes the reply the base's kept tokens
// and the branch's, with the branch's edits as its history:
//   - every token is in one word, in order, and the words' texts join to
//     the reply's text; a word begins at the first token, at each token
//     whose text begins with white space, and after a token whose text
//     ends with a line break, tokens with "" text going with the token
//     after them; a word's branch point is its end;
//   - a history edit is cut to end at k, dropped when it begins at or after
//     k; pins hold from k to the end; an edit ending at k and a pin of the
//     same neuron, kind and value become one range; -0 and 0 are apart; no
//     two results hold one neuron at one step;
//   - accepting keeps the base's first k tokens, then the branch's, and
//     takes the branch's edits, leaving the base as it was;
//   - a point inside a word, where an accepted branch went on without a
//     space, reads as the word's text up to it;
//   - a branch point outside the reply is refused, never sent or kept.
//
// Not locked: which words the view offers, or how it draws them; the
// engine's reading of the ranges (the edit specs and replay suites). Plain
// arrays, no browser.

import assert from "node:assert/strict";
import { test } from "node:test";

import type { Edit, RangedEdit } from "../../../app/src/engine/edits.ts";
import {
  type ReplyToken,
  accepted,
  beforePoint,
  branchEdits,
  wordsOf,
} from "../../../app/src/session/branch.ts";

const tokens = (...texts: string[]): ReplyToken[] =>
  texts.map((text, i) => ({ token: 1000 + i, text }));

const pin = (layer: number, neuron: number, value: number): Edit => ({
  layer,
  neuron,
  kind: "set",
  value,
});
const ranged = (
  edit: Edit,
  from: number,
  until: number | null,
): RangedEdit => ({ ...edit, from, until });

test("a reply reads as words, each with the branch point after it", () => {
  const reply = tokens("The", " light", "house", " keeper", ",", "\n\n", "He");
  const words = wordsOf(reply);
  assert.deepEqual(words, [
    { text: "The", start: 0, end: 1 },
    { text: " lighthouse", start: 1, end: 3 },
    { text: " keeper,", start: 3, end: 5 },
    { text: "\n\n", start: 5, end: 6 },
    { text: "He", start: 6, end: 7 },
  ]);
  assert.equal(
    words.map((w) => w.text).join(""),
    reply.map((t) => t.text).join(""),
  );
});

test("a token that completes no character goes with the word it falls in", () => {
  // An emoji in two tokens: the first completes nothing, the second owns it.
  const reply = tokens("Hi", " ", "", "🙂", " there");
  assert.deepEqual(wordsOf(reply), [
    { text: "Hi", start: 0, end: 1 },
    { text: " 🙂", start: 1, end: 4 },
    { text: " there", start: 4, end: 5 },
  ]);
  assert.deepEqual(wordsOf(tokens("", "é", " ok")), [
    { text: "é", start: 0, end: 2 },
    { text: " ok", start: 2, end: 3 },
  ]);
  assert.deepEqual(wordsOf([]), []);
});

test("a branch sends the reply's edits cut at its point, then the pins from it", () => {
  const a = pin(2, 10, 8);
  const b = pin(5, 7, 0);
  const history = [
    ranged(a, 0, null),
    ranged(b, 9, null),
    ranged(pin(6, 1, 1), 2, 4),
  ];
  assert.deepEqual(branchEdits(history, [pin(3, 3, -2)], 6), [
    ranged(a, 0, 6),
    ranged(pin(6, 1, 1), 2, 4),
    ranged(pin(3, 3, -2), 6, null),
  ]);
  assert.deepEqual(branchEdits(history, [], 0), []);
  assert.deepEqual(branchEdits([], [a], 0), [ranged(a, 0, null)]);
});

test("a pin held unchanged across the point stays one range", () => {
  const a = pin(2, 10, 8);
  const history = [ranged(a, 3, null)];
  assert.deepEqual(branchEdits(history, [a], 7), [ranged(a, 3, null)]);
  const changed = pin(2, 10, 9);
  assert.deepEqual(branchEdits(history, [changed], 7), [
    ranged(a, 3, 7),
    ranged(changed, 7, null),
  ]);
  const zero = pin(2, 10, 0);
  const negativeZero = pin(2, 10, -0);
  const apart = branchEdits([ranged(zero, 0, null)], [negativeZero], 4);
  assert.equal(apart.length, 2);
  assert.ok(Object.is(apart[1]?.value, -0));
});

test("no two edits of a branch hold one neuron at one step", () => {
  const history = [
    ranged(pin(1, 1, 1), 0, 5),
    ranged(pin(1, 1, 2), 5, null),
    ranged(pin(1, 2, 3), 4, null),
  ];
  const pins = [pin(1, 1, 7), pin(1, 2, 3)];
  const sent = branchEdits(history, pins, 5);
  for (let step = 0; step < 20; step++) {
    const held = new Set<string>();
    for (const e of sent) {
      if (e.from <= step && (e.until === null || step < e.until)) {
        const key = `${e.layer}:${e.neuron}`;
        assert.ok(!held.has(key), `${key} twice at step ${step}`);
        held.add(key);
      }
    }
  }
  assert.deepEqual(sent, [
    ranged(pin(1, 1, 1), 0, 5),
    ranged(pin(1, 2, 3), 4, null),
    ranged(pin(1, 1, 7), 5, null),
  ]);
});

test("accepting keeps the base's tokens to k, then the branch's, with its edits", () => {
  const base = { tokens: tokens("A", " b", " c", " d"), edits: [] };
  const branch = {
    tokens: tokens(" x", " y"),
    edits: [ranged(pin(4, 4, 4), 2, null)],
  };
  const line = accepted(base, 2, branch);
  assert.deepEqual(line.tokens, [...base.tokens.slice(0, 2), ...branch.tokens]);
  assert.deepEqual(line.edits, branch.edits);
  assert.equal(base.tokens.length, 4, "the base is not changed");
  assert.deepEqual(
    accepted(base, 4, { tokens: [], edits: [] }).tokens,
    base.tokens,
  );
});

test("a point reads as its word's text up to it, inside a word too", () => {
  const reply = tokens("The", " keeper", ",", " went");
  assert.equal(beforePoint(reply, 1), "The");
  assert.equal(beforePoint(reply, 2), " keeper");
  assert.equal(beforePoint(reply, 3), " keeper,");
  assert.equal(beforePoint(reply, 4), " went");
  assert.throws(() => beforePoint(reply, 0));
  assert.throws(() => beforePoint(reply, 5));
});

test("a branch point outside the reply is refused", () => {
  const base = { tokens: tokens("A", " b"), edits: [] };
  const none = { tokens: [], edits: [] };
  assert.throws(() => accepted(base, 3, none), /outside/u);
  assert.throws(() => accepted(base, -1, none), /outside/u);
  assert.throws(() => accepted(base, 1.5, none), /outside/u);
  assert.throws(() => branchEdits([], [], -1), /not a step/u);
  assert.throws(() => branchEdits([], [], 0.5), /not a step/u);
});
