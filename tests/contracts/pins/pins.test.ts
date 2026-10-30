// Axis V: the pins suite (tests/README.md).
//
// Contract locked (app/src/session/pins.ts): the pins held for the next run
// are at most one edit a neuron, in the order first pinned; a range pinned
// at once is one change, and one chip; the value a pin sends is the edit
// kernel's:
//   - adding edits appends new neurons and replaces held ones in place,
//     keeping their place; adding what is already held returns the same
//     set, so nothing changes;
//   - removing takes only the named neurons of the named layer, the rest in
//     order; removing what is not held returns the same set;
//   - groups are one per layer and edit, -0 apart from 0, in the order each
//     group began, neurons in pin order;
//   - a set sends its value, a zero 0, a scale the f32 product.
//
// Not locked: the session around it (session.ts), which holds the set and
// notifies on change; the browser scenario pins through it end to end.
// Plain arrays, no browser.

import assert from "node:assert/strict";
import { test } from "node:test";

import type { Edit } from "../../../app/src/engine/edits.ts";
import {
  groupsOf,
  sentBy,
  withPins,
  withoutPins,
} from "../../../app/src/session/pins.ts";

const set = (layer: number, neuron: number, value: number): Edit => ({
  layer,
  neuron,
  kind: "set",
  value,
});
const zero = (layer: number, neuron: number): Edit => ({
  layer,
  neuron,
  kind: "zero",
  value: 0,
});

test("adding appends new neurons and replaces held ones in place", () => {
  const held = [set(2, 10, 1), set(2, 11, 1), set(5, 10, 1)];
  const next = withPins(held, [zero(2, 11), set(7, 0, 3), set(5, 10, 4)]);
  assert.deepEqual(next, [
    set(2, 10, 1),
    zero(2, 11),
    set(5, 10, 4),
    set(7, 0, 3),
  ]);
  assert.deepEqual(held, [set(2, 10, 1), set(2, 11, 1), set(5, 10, 1)]);
});

test("a range in one call: later edits of the same neuron win, one entry each", () => {
  const next = withPins([], [set(1, 3, 1), set(1, 4, 1), set(1, 3, 9)]);
  assert.deepEqual(next, [set(1, 3, 9), set(1, 4, 1)]);
});

test("adding what is held already is the same set", () => {
  const held = [set(2, 10, 1), zero(3, 4)];
  assert.equal(withPins(held, [set(2, 10, 1), zero(3, 4)]), held);
  assert.equal(withPins(held, []), held);
  assert.notEqual(withPins(held, [set(2, 10, -1)]), held);
  const negativeZero = withPins([set(0, 0, 0)], [set(0, 0, -0)]);
  assert.ok(Object.is(negativeZero[0]?.value, -0));
});

test("removing takes only the named neurons of the named layer", () => {
  const held = [set(2, 10, 1), set(3, 10, 1), set(2, 11, 1), set(2, 12, 1)];
  assert.deepEqual(withoutPins(held, 2, [10, 12, 99]), [
    set(3, 10, 1),
    set(2, 11, 1),
  ]);
  assert.equal(withoutPins(held, 4, [10]), held);
  assert.equal(withoutPins(held, 2, []), held);
});

test("groups: one per layer and edit, in the order each began", () => {
  const pins = [
    zero(6, 100),
    set(6, 7, 8),
    zero(6, 101),
    zero(2, 100),
    set(6, 8, 8),
    set(6, 9, -0),
    set(6, 10, 0),
    { layer: 6, neuron: 11, kind: "scale", value: 8 } as const,
  ];
  assert.deepEqual(groupsOf(pins), [
    { layer: 6, kind: "zero", value: 0, neurons: [100, 101] },
    { layer: 6, kind: "set", value: 8, neurons: [7, 8] },
    { layer: 2, kind: "zero", value: 0, neurons: [100] },
    { layer: 6, kind: "set", value: -0, neurons: [9] },
    { layer: 6, kind: "set", value: 0, neurons: [10] },
    { layer: 6, kind: "scale", value: 8, neurons: [11] },
  ]);
  assert.deepEqual(groupsOf([]), []);
});

test("a pin sends what the edit kernel computes", () => {
  assert.equal(sentBy(set(0, 0, 8), 0.25), 8);
  assert.equal(sentBy(zero(0, 0), 0.25), 0);
  const scale = { layer: 0, neuron: 0, kind: "scale", value: -4 } as const;
  assert.equal(sentBy(scale, 0.1), Math.fround(0.1 * -4));
  assert.notEqual(sentBy(scale, 0.1), 0.1 * -4);
  assert.ok(Number.isNaN(sentBy(scale, Number.NaN)));
});
