// Axis H (view): reading the prompt view's number fields: the seed and
// each pin's layer, neuron and value, typed as text.
//
// Contract
//   - parseSeed(text): a whole number from 0 to 2^53 - 1, written in
//     decimal digits (spaces around it ignored). The seed crosses to the
//     engine as a double, which holds every such integer exactly.
//   - parsePins(rows): each row's layer and neuron are whole numbers from
//     0 to 2^32 - 1, in decimal digits, the range of the 32-bit words they
//     cross as; a "set" or "scale" row's value is a decimal number (sign,
//     fraction and exponent allowed: "-200", "0.5", "1e3"); a "zero" row
//     takes no value, and sends 0.
//   - Either the value, or every problem found, each naming its field by
//     the label the view gives it ("Seed", "Pin 2 layer") and why.
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

import type { Edit, EditKind } from "../../engine/edits.ts";

export interface PinFields {
  readonly layer: string;
  readonly neuron: string;
  readonly kind: EditKind;
  readonly value: string;
}

export interface Problem {
  readonly field: string;
  readonly reason: string;
}

export type Parsed<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly problems: readonly Problem[] };

const WHOLE = /^\d+$/;
const DECIMAL = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
const SEED_LIMIT = 2 ** 53 - 1;
const WORD_LIMIT = 2 ** 32 - 1;

// A whole number from 0 to `limit`, or why not.
function whole(text: string, field: string, limit: number): number | Problem {
  const trimmed = text.trim();
  if (!WHOLE.test(trimmed)) return { field, reason: "not a whole number" };
  const value = Number(trimmed);
  if (value > limit) return { field, reason: `past ${limit}` };
  return value;
}

export function parseSeed(text: string): Parsed<number> {
  const seed = whole(text, "Seed", SEED_LIMIT);
  return typeof seed === "number"
    ? { ok: true, value: seed }
    : { ok: false, problems: [seed] };
}

export function parsePins(rows: readonly PinFields[]): Parsed<readonly Edit[]> {
  const edits: Edit[] = [];
  const problems: Problem[] = [];
  rows.forEach((row, index) => {
    const name = `Pin ${index + 1}`;
    const layer = whole(row.layer, `${name} layer`, WORD_LIMIT);
    const neuron = whole(row.neuron, `${name} neuron`, WORD_LIMIT);
    let value: number | Problem = 0;
    if (row.kind !== "zero") {
      const trimmed = row.value.trim();
      value = DECIMAL.test(trimmed)
        ? Number(trimmed)
        : { field: `${name} value`, reason: "not a number" };
    }
    for (const part of [layer, neuron, value]) {
      if (typeof part !== "number") problems.push(part);
    }
    if (
      typeof layer === "number" &&
      typeof neuron === "number" &&
      typeof value === "number"
    ) {
      edits.push({ layer, neuron, kind: row.kind, value });
    }
  });
  return problems.length === 0
    ? { ok: true, value: edits }
    : { ok: false, problems };
}
