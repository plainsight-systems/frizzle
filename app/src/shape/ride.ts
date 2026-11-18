// Axis H (shape): the Ride's road: at the branch point, what each layer
// would say next if the model stopped there, read from the captures. Plain
// TypeScript, no React, no DOM.
//
// Contract
//   - roadOf(captures, next): one RideLayer a captured layer, in order:
//       guesses  its most likely next words, best first, as wordOf shows
//                them (the engine's guesses, runtime/engine.h)
//       top      the best guess's token identifier
//       lights   the layer's LIGHTS strongest neurons (shape/rank.ts), each
//                its index and value as the run used it
//       grid     the fold of the layer's neurons (shape/fold.ts), where a
//                neuron sits, as on the deck
//     Road.next is `next`, the token the reply went on to write, as wordOf
//     shows it, or the last layer's best guess where the reply ended.
//   - forkOf(a, b): the first layer whose best guess differs between two
//     roads, or null when none does. Roads of different lengths throw a
//     RangeError.
//   - wordOf(text): a token's text as a person reads it alone: leading white
//     space dropped; line breaks "↵"; other characters that draw nothing as
//     "⟨U+XXXX⟩"; marks with nothing to sit on after "◌"; a token of spaces
//     alone "▁"; "" as "(no text)". Never blank.
//   - onTheWay(word, upcoming): whether a guess is a word the reply goes on
//     to write: the next one, or any later one of at least four characters
//     (shorter ones, "the", match by chance).
//
// Design and cost
//   - The readout is a side reading, not the model's own output, except the
//     last layer's (capture/capture.h); nothing here computes a probability.
//   - A road is 26 sorts of 6,912 neurons, when the shown captures
//     change, never per frame.

import type { Captures } from "../protocol/messages.ts";
import { type Grid, fold } from "./fold.ts";
import { strongest } from "./rank.ts";

type Taken = Extract<Captures, { readonly kind: "taken" }>;

export const LIGHTS = 6;

export interface NextToken {
  readonly id: number;
  readonly text: string;
}

export interface Light {
  readonly neuron: number;
  readonly value: number;
}

export interface RideLayer {
  readonly guesses: readonly string[];
  readonly top: number;
  readonly lights: readonly Light[];
  readonly grid: Grid;
}

export interface Road {
  readonly next: string;
  readonly layers: readonly RideLayer[];
}

const UNSEEN = /^[\p{Cc}\p{Cf}\p{Cn}\p{Co}\p{Z}\u{FE00}-\u{FE0F}]$/u;
const MARKS = /^\p{M}+$/u;

export function wordOf(text: string): string {
  if (text === "") return "(no text)";
  const trimmed = text.replace(/^[ \t ]+/u, "");
  if (trimmed === "") return "▁";
  let out = "";
  for (const character of trimmed) {
    const code = character.codePointAt(0) ?? 0;
    if (character === "\n") out += "↵";
    else if (UNSEEN.test(character)) {
      out += `⟨U+${code.toString(16).toUpperCase().padStart(4, "0")}⟩`;
    } else out += character;
  }
  return MARKS.test(out) ? `◌${out}` : out;
}

export function roadOf(captures: Taken, next: NextToken | null): Road {
  const { words, layers, guesses } = captures;
  const lastGuess = guesses.at(-1)?.[0];
  const target = next ?? lastGuess ?? null;
  const road = layers.map((slots, l): RideLayer => {
    const own = guesses[l] ?? [];
    const values = words.subarray(
      slots.neurons.first,
      slots.neurons.first + slots.neurons.width,
    );
    return {
      guesses: own.map((g) => wordOf(g.text)),
      top: own[0]?.id ?? -1,
      lights: strongest(values, LIGHTS).map((neuron) => ({
        neuron,
        value: values[neuron] ?? 0,
      })),
      grid: fold(slots.neurons.width),
    };
  });
  return { next: target === null ? "" : wordOf(target.text), layers: road };
}

export function forkOf(a: Road, b: Road): number | null {
  if (a.layers.length !== b.layers.length) {
    throw new RangeError(
      `roads of ${a.layers.length} and ${b.layers.length} layers`,
    );
  }
  const at = a.layers.findIndex((l, i) => l.top !== b.layers[i]?.top);
  return at === -1 ? null : at;
}

export function onTheWay(word: string, upcoming: readonly string[]): boolean {
  return (
    word === upcoming[0] || ([...word].length >= 4 && upcoming.includes(word))
  );
}
