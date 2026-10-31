// Axis H (session): a reply as words and branch points, and the edits a
// run sends from one. Plain TypeScript, no React: the session holds the
// replies and the pins, the views show what this file makes of them.
//
// Contract
//   - A reply is its tokens, each an identifier and the text it completes,
//     possibly "" (protocol/messages.ts, "text"), in order.
//   - wordsOf(tokens): the reply as words, in order, every token in exactly
//     one, their texts joined giving the reply's text. A word begins at the
//     first token, at each token whose text begins with white space, and at
//     each token after one whose text ends with a line break, reading a
//     token whose text is "" as the token after it (it completes no
//     character, and belongs with the token that does). Each word: its text, the
//     tokens it spans [start, end), and its branch point, `end`: the number
//     of the reply's tokens a branch after it keeps. A word whose text is
//     white space alone (a line break) is a word, so a branch may follow a
//     paragraph; the view decides which words it offers.
//   - The branch point k, from 0 (before the first word) to the reply's
//     length (after the last), is the run's step k: the position whose
//     input is the k-th kept token, which draws reply token k
//     (src/core/edit/edit.h). Pins at k change the neurons there, where the
//     model has read the whole word and chooses what follows.
//   - branchEdits(history, pins, k): the edits a branch from a reply sends.
//     `history` is the reply's own edits, ranged (the edits its run sent);
//     each is cut to end at k, and dropped when it begins at or after k.
//     Then each pin held now, from k to the end. A history edit that ends
//     at k and a pin of the same neuron, kind and value are joined into one
//     range, so a pin held unchanged across accepted replies stays one
//     edit. The result: the cut history in its order, a joined edit in its
//     history edit's place; then the other pins in their order. No two of
//     it hold one neuron at one step.
//   - inspectEdits(history): the edits an inspection of the reply sends:
//     its history, as it is. An inspection computes the reply's own step k
//     (maxTokens 0), so the neurons are the reply's as generated, under the
//     edits that were in force there.
//   - beforePoint(tokens, k), 0 < k <= the reply's length: the text of the
//     word the point k falls in, from the word's start to k. A point is a
//     word's end unless an accepted branch went on without a space, which
//     leaves it inside a word. Outside that range it throws, a page bug.
//   - accepted(base, k, branch): what the reply becomes when a branch from
//     k is accepted: the base's first k tokens, then the tokens the branch
//     emitted, and the edits the branch sent, its history from now on. The
//     base is not changed.
//
// Design
//   - Words, not tokens: a person reads and steers meaning, and Gemma's
//     tokenizer puts a word's leading space in its first token, so a word
//     is whole tokens and its punctuation stays with it. A word split
//     across tokens is branched after its last, never inside it.
//   - The page holds every reply's identifiers and edits, and sends them
//     whole with each run, as it sends the pins: the engine holds no
//     conversation (logical-overview.md, invariant 4).
//   - A branch point outside [0, the reply's length], or not a whole
//     number, is a page bug: branchEdits and accepted throw, naming it,
//     rather than send or keep a range the reply does not have.
//
// Algorithm
//   - wordsOf: group each run of "" tokens with the token after it (a run
//     at the end stays with the word before); a group starts a word when it
//     is the first, when its text begins with white space, or when the
//     group before ends with a line break. One pass, O(tokens).
//   - branchEdits: each history edit beginning before k, its end made
//     min(until, k); one that then ends at k is joined with the pin, held
//     now, of its layer, neuron, kind and value (Object.is, so -0 and 0 are
//     apart), its end becoming null, and that pin is spent. Then each pin
//     not spent, from k. A neuron's history ranges never share a step
//     (edit.h), so at most one of them ends at k. O(history + pins).

import type { Edit, RangedEdit } from "../engine/edits.ts";

export interface ReplyToken {
  readonly token: number;
  readonly text: string;
}

export interface Word {
  readonly text: string;
  readonly start: number; // the first token's index
  readonly end: number; // past the last token: the branch point after it
}

export interface Line {
  readonly tokens: readonly ReplyToken[];
  readonly edits: readonly RangedEdit[];
}

export function wordsOf(tokens: readonly ReplyToken[]): readonly Word[] {
  const words: Word[] = [];
  let text = "";
  let start = 0;
  let group = 0; // the first token of the group being read
  let groupText = "";
  let previous = ""; // the text of the group before
  const close = (end: number) => {
    if (end > start) words.push({ text, start, end });
  };
  for (let i = 0; i < tokens.length; i++) {
    groupText += (tokens[i] as ReplyToken).text;
    const last = i === tokens.length - 1;
    if (groupText === "" && !last) continue;
    const begins =
      group === 0 || /^\s/u.test(groupText) || /[\n\r]$/u.test(previous);
    if (begins && groupText !== "") {
      close(group);
      text = "";
      start = group;
    }
    text += groupText;
    previous = groupText === "" ? previous : groupText;
    groupText = "";
    group = i + 1;
  }
  close(tokens.length);
  return words;
}

export function branchEdits(
  history: readonly RangedEdit[],
  pins: readonly Edit[],
  k: number,
): readonly RangedEdit[] {
  if (!Number.isInteger(k) || k < 0) {
    throw new Error(`branchEdits: the branch point ${k} is not a step`);
  }
  const spent = new Set<number>(); // indices of pins joined to history
  const sent: RangedEdit[] = [];
  for (const edit of history) {
    if (edit.from >= k) continue;
    const until = edit.until === null || edit.until > k ? k : edit.until;
    const joined =
      until === k
        ? pins.findIndex(
            (p, i) =>
              !spent.has(i) &&
              p.layer === edit.layer &&
              p.neuron === edit.neuron &&
              p.kind === edit.kind &&
              Object.is(p.value, edit.value),
          )
        : -1;
    if (joined >= 0) spent.add(joined);
    sent.push({ ...edit, until: joined >= 0 ? null : until });
  }
  pins.forEach((p, i) => {
    if (!spent.has(i)) sent.push({ ...p, from: k, until: null });
  });
  return sent;
}

export function inspectEdits(
  history: readonly RangedEdit[],
): readonly RangedEdit[] {
  return history;
}

export function accepted(
  base: Line,
  k: number,
  branch: {
    readonly tokens: readonly ReplyToken[];
    readonly edits: readonly RangedEdit[];
  },
): Line {
  if (!Number.isInteger(k) || k < 0 || k > base.tokens.length) {
    throw new Error(
      `accepted: the branch point ${k} is outside the reply's ${base.tokens.length} tokens`,
    );
  }
  return {
    tokens: [...base.tokens.slice(0, k), ...branch.tokens],
    edits: branch.edits,
  };
}

export function beforePoint(tokens: readonly ReplyToken[], k: number): string {
  const word = wordsOf(tokens).find((w) => w.start < k && k <= w.end);
  if (word === undefined) {
    throw new Error(
      `beforePoint: the point ${k} is outside the reply's ${tokens.length} tokens`,
    );
  }
  return tokens
    .slice(word.start, k)
    .map((t) => t.text)
    .join("");
}
