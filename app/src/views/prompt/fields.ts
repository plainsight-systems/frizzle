// Axis H (view): reading the page's number fields, typed as text: the
// seed, a neuron's layer and number, and a pin's value.
//
// Contract
//   - parseSeed(text): a whole number from 0 to 2^53 - 1, written in
//     decimal digits (spaces around it ignored). The seed crosses to the
//     engine as a double, which holds every such integer exactly.
//   - parseWord(text, field): a whole number from 0 to 2^32 - 1, in decimal
//     digits, the range of the 32-bit words a layer and a neuron cross as.
//   - parseValue(text, field): a decimal number (sign, fraction and
//     exponent allowed: "-200", "0.5", "1e3").
//   - Either the value, or the problem, naming its field by the label the
//     view gives it ("Seed", "Pin value") and why.
//
// Design
//   - This is parsing, not judging: whether layer 40 exists, a neuron is
//     in range, or a value is finite is the engine's to say (edit::check),
//     by name, so there is one judge (engine/edits.ts). What is checked
//     here is only that the text is a number the crossing carries
//     unchanged: a fraction, a number past 2^32 or an empty field would
//     otherwise reach the engine as another number.
//   - Decimal digits only, not JavaScript's Number(): "0x10", "" and " "
//     would read as 16, 0 and 0.

export interface Problem {
  readonly field: string;
  readonly reason: string;
}

export type Parsed<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly problem: Problem };

const WHOLE = /^\d+$/;
const DECIMAL = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
const SEED_LIMIT = 2 ** 53 - 1;
const WORD_LIMIT = 2 ** 32 - 1;

// A whole number from 0 to `limit`, or why not.
function whole(text: string, field: string, limit: number): Parsed<number> {
  const trimmed = text.trim();
  if (!WHOLE.test(trimmed)) {
    return { ok: false, problem: { field, reason: "not a whole number" } };
  }
  const value = Number(trimmed);
  if (value > limit) {
    return { ok: false, problem: { field, reason: `past ${limit}` } };
  }
  return { ok: true, value };
}

export function parseSeed(text: string): Parsed<number> {
  return whole(text, "Seed", SEED_LIMIT);
}

export function parseWord(text: string, field: string): Parsed<number> {
  return whole(text, field, WORD_LIMIT);
}

export function parseValue(text: string, field: string): Parsed<number> {
  const trimmed = text.trim();
  return DECIMAL.test(trimmed)
    ? { ok: true, value: Number(trimmed) }
    : { ok: false, problem: { field, reason: "not a number" } };
}
