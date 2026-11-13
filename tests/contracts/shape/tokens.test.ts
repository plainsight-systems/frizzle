// Axis V: the shape suite (tests/README.md), the input's tokens as a person
// reads them one at a time.
//
// Contract locked (app/src/shape/tokens.ts): every token of the input is in
// exactly one span, in order, and no token reads as a blank or as its
// position alone:
//   - a control token is its own span, named by its spelling, "<control>"
//     when it has none, and read as "the control token {name}";
//   - a token with text is its own span, shown with white space made
//     visible (a space ▁, a line break ↵, other characters below U+0020 and
//     U+007F as Control Pictures, every other character that draws nothing
//     or only space, \p{Cc}, \p{Cf} or \p{Z}, as its code point);
//   - tokens with no text before a token with text that is not a control
//     token are one span with it, shown once as its text, each read as
//     "part i of k of “{text}”";
//   - tokens with no text that complete nothing (before a control token,
//     or at the end) are spans of their own, "(no text)", read as "a token
//     with no text of its own";
//   - a span continues the word before it when letters, digits or marks
//     run across its start, after a text span; never first, never after a
//     control or empty span; punctuation after a word does not;
//   - spanAt finds the span holding a token; an index outside the input,
//     or one not whole, is refused, as is a reading of a token outside its
//     span.
//
// Not locked: how the pass map places or styles a label. Plain arrays, no
// browser.

import assert from "node:assert/strict";
import { test } from "node:test";

import type {
  InputToken,
  TokenType,
} from "../../../app/src/protocol/messages.ts";
import {
  type Span,
  labelOf,
  readingOf,
  spanAt,
  spansOf,
} from "../../../app/src/shape/tokens.ts";

const token = (text: string, type: TokenType = "normal"): InputToken => ({
  id: 0,
  type,
  text,
});
const control = (text: string): InputToken => token(text, "control");
const byte = (text: string): InputToken => token(text, "byte");

// Each span as [first, count, label, fragment or the kind].
const outline = (spans: readonly Span[]) =>
  spans.map((s) => [
    s.first,
    s.count,
    labelOf(s),
    s.shown.kind === "text" ? s.shown.fragment : s.shown.kind,
  ]);

test("each token with text is its own span, white space made visible", () => {
  const spans = spansOf([
    control("<bos>"),
    token("What"),
    token(" is"),
    token("\n\n"),
    token("\t x"),
    token("\u0001\u007f\r"),
  ]);
  assert.deepEqual(outline(spans), [
    [0, 1, "<bos>", "control"],
    [1, 1, "What", false],
    [2, 1, "▁is", false],
    [3, 1, "↵↵", false],
    [4, 1, "␉▁x", false],
    [5, 1, "␁␡␍", false],
  ]);
});

test("a character that draws nothing or only space is shown as its code point", () => {
  const spans = spansOf([
    token(" "),
    token("　x"),
    token("​"),
    token("﻿"),
    token(" "),
    token("\u0085"),
    token("é́"),
  ]);
  assert.deepEqual(spans.map(labelOf), [
    "⟨U+00A0⟩",
    "⟨U+3000⟩x",
    "⟨U+200B⟩",
    "⟨U+FEFF⟩",
    "⟨U+2028⟩",
    "⟨U+0085⟩",
    "é́",
  ]);
});

test("a span continues the word before it only when letters run across its start", () => {
  const spans = spansOf([
    token(" run"),
    token("ning"),
    token("?"),
    token(" é"),
    token("t"),
    token("12"),
    token("34"),
    token("e"),
    token("́"),
    token(" ("),
    token("x"),
  ]);
  assert.deepEqual(
    spans.map((s) => (s.shown.kind === "text" ? s.shown.fragment : null)),
    [false, true, false, false, true, true, true, true, true, false, false],
  );
});

test("a span after a control token never continues a word", () => {
  const spans = spansOf([control("<start_of_turn>"), token("user")]);
  assert.deepEqual(outline(spans), [
    [0, 1, "<start_of_turn>", "control"],
    [1, 1, "user", false],
  ]);
});

test("tokens with no text join the token that completes their character, shown once", () => {
  const input = [token(" a"), byte(""), byte(""), byte("☕"), token(" b")];
  const spans = spansOf(input);
  assert.deepEqual(outline(spans), [
    [0, 1, "▁a", false],
    [1, 3, "☕", false],
    [4, 1, "▁b", false],
  ]);
  const cup = spanAt(spans, 2);
  assert.equal(readingOf(cup, 1), "part 1 of 3 of “☕”");
  assert.equal(readingOf(cup, 2), "part 2 of 3 of “☕”");
  assert.equal(readingOf(cup, 3), "part 3 of 3 of “☕”");
});

test("a character split across tokens inside a word is a fragment, and so is what follows it", () => {
  const spans = spansOf([token(" caf"), byte(""), byte("é"), token("s")]);
  assert.deepEqual(outline(spans), [
    [0, 1, "▁caf", false],
    [1, 2, "é", true],
    [3, 1, "s", true],
  ]);
  assert.equal(
    readingOf(spanAt(spans, 1), 1),
    "part 1 of 2 of “é”, part of a word",
  );
});

test("tokens with no text that complete nothing are empty spans of their own", () => {
  const spans = spansOf([
    token("ab"),
    byte(""),
    byte(""),
    control("<end_of_turn>"),
    byte(""),
  ]);
  assert.deepEqual(outline(spans), [
    [0, 1, "ab", false],
    [1, 1, "(no text)", "empty"],
    [2, 1, "(no text)", "empty"],
    [3, 1, "<end_of_turn>", "control"],
    [4, 1, "(no text)", "empty"],
  ]);
  assert.equal(
    readingOf(spanAt(spans, 4), 4),
    "a token with no text of its own",
  );
});

test("a control token is read by its name, and named even without a spelling", () => {
  const spans = spansOf([control("<bos>"), control("")]);
  assert.equal(readingOf(spanAt(spans, 0), 0), "the control token <bos>");
  assert.equal(labelOf(spanAt(spans, 1)), "<control>");
  assert.equal(readingOf(spanAt(spans, 1), 1), "the control token <control>");
});

test("a token alone reads as its visible text, a fragment says so", () => {
  const spans = spansOf([token(" France"), token(" run"), token("ning")]);
  assert.equal(readingOf(spanAt(spans, 0), 0), "“▁France”");
  assert.equal(readingOf(spanAt(spans, 2), 2), "“ning”, part of a word");
});

test("every token is in exactly one span, in order, and spanAt finds it", () => {
  const input = [
    control("<bos>"),
    token("Hi"),
    byte(""),
    byte("☕"),
    byte(""),
    control("<end_of_turn>"),
    token("\n"),
    byte(""),
  ];
  const spans = spansOf(input);
  let next = 0;
  for (const span of spans) {
    assert.equal(span.first, next);
    assert.ok(span.count >= 1);
    next += span.count;
  }
  assert.equal(next, input.length);
  for (let i = 0; i < input.length; i++) {
    const span = spanAt(spans, i);
    assert.ok(span.first <= i && i < span.first + span.count);
    assert.notEqual(readingOf(span, i), "");
    assert.notEqual(labelOf(span), "");
  }
});

test("an index outside the input or its span, or not whole, is refused", () => {
  const spans = spansOf([token("a"), byte(""), byte("☕")]);
  assert.throws(() => spanAt(spans, -1), RangeError);
  assert.throws(() => spanAt(spans, 3), RangeError);
  assert.throws(() => spanAt(spans, 1.5), RangeError);
  assert.throws(() => readingOf(spanAt(spans, 1), 0), RangeError);
  assert.equal(spansOf([]).length, 0);
});
