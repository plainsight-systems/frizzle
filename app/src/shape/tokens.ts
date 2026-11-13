// Axis H (shape): the input's tokens as a person reads them one at a time,
// out of the text they sit in, so an attention key is named by its token,
// never by its position alone (research/plan.md, "Showing tokens as
// text"). Plain TypeScript, no React, no DOM.
//
// Contract
//   - spansOf(input): the input as spans, in order, every token in exactly
//     one, each span one thing to show:
//       control  a control token (type "control"), alone. Its name is its
//                text, the template's spelling ("<bos>",
//                "<start_of_turn>"), shown as a name and never as text; a
//                control token with no text is named "<control>".
//       text     a token with text, alone, or a run of tokens with no text
//                of their own ("") and the token after them, when that
//                token is not a control token and has text: the run began
//                a character the last token completes (a character split
//                across byte tokens), and the span is shown once, as the
//                last token's text. Its text is shown with white space
//                made visible: a space "▁", a line break "↵", every other
//                character below U+0020 and U+007F as its Control Pictures
//                sign (U+2400 + code, U+2421); every other character that
//                draws nothing or only space (the other controls, \p{Cc};
//                format characters, \p{Cf}, such as U+200B and U+FEFF;
//                separators, \p{Z}, such as U+00A0, U+3000 and U+2028) as
//                its code point, "⟨U+00A0⟩"; the rest as they are.
//                `fragment` when the span continues the word before it:
//                its text begins with a letter, digit or mark (\p{L},
//                \p{N}, \p{M}) and the span before it is a text span whose
//                text ends with one. The first span, and a span after a
//                control or empty span, never continue a word.
//       empty    a token with no text of its own that completes nothing: a
//                run of such tokens followed by a control token, or ending
//                the input, one span a token.
//     Other types ("normal", "byte", "unknown", "user-defined", "unused",
//     "undefined") are read by their text alone.
//   - spanAt(spans, index): the span holding token `index`. An index that
//     is not a whole number inside the spans throws a RangeError.
//   - labelOf(span): the short text drawn beside a key: a control token's
//     name; a text span's visible text; "(no text)" for an empty span.
//     Never blank.
//   - readingOf(span, index): what the reading says the token `index` is:
//       control           "the control token <bos>"
//       text, one token   "“▁France”"; a fragment "“ning”, part of a word"
//       text, k tokens    "part 2 of 3 of “☕”" (the token's place in the
//                         run, from 1), with ", part of a word" after it
//                         when the span is a fragment
//       empty             "a token with no text of its own"
//     An index outside the span throws a RangeError.
//
// Algorithm
//   1. spansOf: walk the input once, holding the first index of a pending
//      run of tokens with no text. A control token closes the run as empty
//      spans, one a token, then is its own span. A token with text closes
//      the run into its own text span. At the end, a pending run becomes
//      empty spans. Then each text span's `fragment` from the span before
//      it, by the rule above.
//   2. spanAt: binary search on the spans' first indices.
//
// Design and cost
//   - Spans, not one label a token: a character split across byte tokens
//     is drawn once (the plan's rule), and its tokens read as parts of it,
//     so no token reads as a blank. The detokenizer gives the text to the
//     token that completes the character (tokenizer/detokenizer.h, texts),
//     so the run is always the tokens before the one with text.
//   - Control tokens by name, never as text: the plan's rule keeps them out
//     of the prompt as people read it, but a key on <bos> is a fact a
//     person reading attention needs (a head resting on the first token is
//     the commonest thing attention does), so the reading names it.
//   - The fragment rule reads only the token's own text and its
//     neighbour's, no vocabulary: a token is part of a word when letters
//     run across its start. Punctuation after a word is not a fragment.
//   - Cost: one pass over the input when the captures change, a span each
//     token at most; a binary search a reading.

import type { InputToken } from "../protocol/messages.ts";

export type Shown =
  | { readonly kind: "control"; readonly name: string }
  | {
      readonly kind: "text";
      readonly text: string; // white space made visible
      readonly fragment: boolean;
    }
  | { readonly kind: "empty" };

// Tokens first .. first + count − 1 of the input, shown as one.
export interface Span {
  readonly first: number;
  readonly count: number;
  readonly shown: Shown;
}

export function spansOf(input: readonly InputToken[]): readonly Span[] {
  const spans: Span[] = [];
  // Raw text of each text span, for the fragment rule (step 1).
  const raw: (string | null)[] = [];
  // The first token of a pending run with no text, or -1.
  let run = -1;
  const closeAsEmpty = (end: number) => {
    if (run === -1) return;
    for (let i = run; i < end; i++) {
      spans.push({ first: i, count: 1, shown: { kind: "empty" } });
      raw.push(null);
    }
    run = -1;
  };
  input.forEach((token, i) => {
    if (token.type === "control") {
      closeAsEmpty(i);
      const name = token.text === "" ? "<control>" : token.text;
      spans.push({ first: i, count: 1, shown: { kind: "control", name } });
      raw.push(null);
    } else if (token.text === "") {
      if (run === -1) run = i;
    } else {
      const first = run === -1 ? i : run;
      run = -1;
      const before = raw.at(-1) ?? null;
      const fragment =
        before !== null && WORD_END.test(before) && WORD_START.test(token.text);
      spans.push({
        first,
        count: i - first + 1,
        shown: { kind: "text", text: visible(token.text), fragment },
      });
      raw.push(token.text);
    }
  });
  closeAsEmpty(input.length);
  return spans;
}

export function spanAt(spans: readonly Span[], index: number): Span {
  const last = spans.at(-1);
  const end = last === undefined ? 0 : last.first + last.count;
  if (!Number.isInteger(index) || index < 0 || index >= end) {
    throw new RangeError(`token ${index}; the input has ${end}, from 0`);
  }
  // The last span whose first is at most index (step 2).
  let low = 0;
  let high = spans.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if ((spans[middle]?.first ?? 0) <= index) low = middle;
    else high = middle - 1;
  }
  const span = spans[low];
  if (span === undefined) throw new RangeError(`token ${index}; no span`);
  return span;
}

export function labelOf(span: Span): string {
  switch (span.shown.kind) {
    case "control":
      return span.shown.name;
    case "text":
      return span.shown.text;
    case "empty":
      return "(no text)";
  }
}

export function readingOf(span: Span, index: number): string {
  if (
    !Number.isInteger(index) ||
    index < span.first ||
    index >= span.first + span.count
  ) {
    throw new RangeError(
      `token ${index} is outside the span of tokens ${span.first}–${span.first + span.count - 1}`,
    );
  }
  const { shown } = span;
  switch (shown.kind) {
    case "control":
      return `the control token ${shown.name}`;
    case "empty":
      return "a token with no text of its own";
    case "text": {
      const quoted = `“${shown.text}”`;
      const said =
        span.count === 1
          ? quoted
          : `part ${index - span.first + 1} of ${span.count} of ${quoted}`;
      return shown.fragment ? `${said}, part of a word` : said;
    }
  }
}

// Letters, digits or marks run across a span's start (the header's rule).
const WORD_START = /^[\p{L}\p{N}\p{M}]/u;
const WORD_END = /[\p{L}\p{N}\p{M}]$/u;

// What draws nothing, or only space, past the cases visible() names first.
const UNSEEN = /^[\p{Cc}\p{Cf}\p{Z}]$/u;

// White space made visible: a space ▁, a line break ↵, every other
// character below U+0020 and U+007F as its Control Pictures sign, every
// other unseen character as its code point.
function visible(text: string): string {
  let out = "";
  for (const character of text) {
    const code = character.codePointAt(0) ?? 0;
    if (character === " ") out += "▁";
    else if (character === "\n") out += "↵";
    else if (code < 0x20) out += String.fromCodePoint(0x2400 + code);
    else if (code === 0x7f) out += "␡";
    else if (UNSEEN.test(character)) {
      out += `⟨U+${code.toString(16).toUpperCase().padStart(4, "0")}⟩`;
    } else out += character;
  }
  return out;
}
