// Axis V: the shape suite (tests/README.md), the Ride's road.
//
// Contract locked (app/src/shape/ride.ts): each layer's guesses are the
// engine's, as a person reads them; the road's next word is the one given,
// or with none the last layer's best guess; a layer's lights are its
// strongest neurons by size; two roads fork at the first layer whose best guess differs; a
// token never reads as a blank; a guess is on the way when it is the next
// word, or a later one of at least four characters.
//
// Not locked: how the Ride draws any of it. Plain arrays, no browser.

import assert from "node:assert/strict";
import { test } from "node:test";

import type {
  Captures,
  LayerSlots,
} from "../../../app/src/protocol/messages.ts";
import {
  forkOf,
  onTheWay,
  roadOf,
  wordOf,
} from "../../../app/src/shape/ride.ts";

type Taken = Extract<Captures, { readonly kind: "taken" }>;

const slot = (first: number, width: number) => ({ first, width });

// Two layers of 8 neurons.
function captures(guesses: Taken["guesses"]): Taken {
  const words = new Float32Array(2 * 8);
  words.set([0, 5, -9, 1, 0, 0, 2, 0], 0);
  words.set([3, 0, 0, 0, -1, 0, 0, 0], 8);
  const empty = slot(0, 0);
  const layers = [0, 1].map((l): LayerSlots => ({
    neurons: slot(8 * l, 8),
    residual: empty,
    attentionAdded: empty,
    mlpAdded: empty,
    query: empty,
    weights: empty,
    stats: empty,
    readout: empty,
    keys: 0,
    queryHeads: 0,
    window: 1,
    full: false,
  }));
  return { kind: "taken", layers, words, position: 0, input: [], guesses };
}

const guess = (id: number, text: string) => ({ id, text });

test("a road holds each layer's guesses and its lights", () => {
  const road = roadOf(
    captures([[guess(10, " the"), guess(20, " Paris")], [guess(40, " Paris")]]),
    guess(40, " Paris"),
  );
  assert.equal(road.next, "Paris");
  assert.deepEqual(road.layers[0]?.guesses, ["the", "Paris"]);
  assert.equal(road.layers[0]?.top, 10);
  assert.deepEqual(road.layers[0]?.lights.slice(0, 3), [
    { neuron: 2, value: -9 },
    { neuron: 1, value: 5 },
    { neuron: 6, value: 2 },
  ]);
});

test("with no next word the road heads for the last layer's best guess", () => {
  const road = roadOf(captures([[guess(7, "a")], [guess(8, " end")]]), null);
  assert.equal(road.next, "end");
});

test("two roads fork at the first layer whose best guess differs", () => {
  const taken = (second: number) =>
    roadOf(captures([[guess(1, "a")], [guess(second, "b")]]), null);
  assert.equal(forkOf(taken(5), taken(6)), 1);
  assert.equal(forkOf(taken(5), taken(5)), null);
  assert.throws(() => forkOf(taken(5), { next: "", layers: [] }), RangeError);
});

test("a token never reads as a blank", () => {
  assert.equal(wordOf(" France"), "France");
  assert.equal(wordOf("   "), "▁");
  assert.equal(wordOf("\n\n"), "↵↵");
  assert.equal(wordOf(""), "(no text)");
  assert.equal(wordOf("\u{FE0E}"), "⟨U+FE0E⟩");
  assert.equal(wordOf("َ"), "◌َ");
});

test("a guess is on the way when the reply goes on to write it", () => {
  const upcoming = ["is", "Paris", "the", "city"];
  assert.equal(onTheWay("is", upcoming), true);
  assert.equal(onTheWay("Paris", upcoming), true);
  assert.equal(onTheWay("the", upcoming), false);
  assert.equal(onTheWay("London", upcoming), false);
});
