// Axis V: the shape suite (tests/README.md), one layer's pass.
//
// Contract locked (app/src/shape/pass.ts): a layer's stages at the last
// position, in the order the layer computes them, each at its real width
// from the captures, and each captured row where its values lie in the
// store:
//   - the eleven stages in order, with their names, units, rows and
//     whether captured; layer 0's residual named as the embedding;
//   - widths from the captures alone: the residual's, the neurons', the
//     head dimension as query width over query heads, K and V one head
//     dimension each;
//   - captured rows at their slots, Q's head h at h head dimensions in,
//     attention's head h at h × keys in; every row of a stage not captured
//     null;
//   - attention n = min(position + 1, keys) keys wide, not its slot's
//     keys: a local layer whose window the input outgrows reads its last
//     `keys` tokens, key 0 at position − keys + 1; a full layer reads back
//     to position 0;
//   - the layer's window and kind as the captures state them;
//   - a layer outside the captures, or not whole, and a query width that
//     is not a whole multiple of the heads, refused;
//   - the scales: the Map's column scale for the four columns it draws,
//     one scale across the layers for Q, and 1 for attention.
//
// Not locked: how the pass map draws a stage (PassMap.tsx). A fixture
// smaller than any model, so no width can come from Gemma 3 1B's.

import assert from "node:assert/strict";
import { test } from "node:test";

import type {
  Captures,
  InputToken,
  LayerSlots,
} from "../../../app/src/protocol/messages.ts";
import { passOf, passScales } from "../../../app/src/shape/pass.ts";
import { scaleAcross } from "../../../app/src/shape/strip.ts";

type Taken = Extract<Captures, { readonly kind: "taken" }>;

// Two layers: E = 4, F = 6, H = 2 query heads of d = 3; layer 0 local, a
// window of 2; layer 1 full, 16 keys (its trained context). The input is 5
// tokens, so layer 0's window is outgrown and layer 1's is not.
const E = 4;
const F = 6;
const H = 2;
const D = 3;

function layer(
  neurons: number,
  at: number,
  keys: number,
  window: number,
  full: boolean,
): LayerSlots {
  const residual = { first: at, width: E };
  const attentionAdded = { first: at + E, width: E };
  const mlpAdded = { first: at + 2 * E, width: E };
  const query = { first: at + 3 * E, width: H * D };
  const weights = { first: query.first + H * D, width: H * keys };
  const stats = { first: weights.first + H * keys, width: H * 2 };
  return {
    neurons: { first: neurons, width: F },
    residual,
    attentionAdded,
    mlpAdded,
    query,
    weights,
    stats,
    keys,
    queryHeads: H,
    window,
    full,
  };
}

const L0 = layer(0, 12, 2, 2, false); // region 2 from 12: 12 .. 37
const L1 = layer(6, 38, 16, 16, true); // 38 .. 91
const WORDS = 92;

const input: InputToken[] = Array.from({ length: 5 }, (_, i) => ({
  id: i,
  type: "normal",
  text: `t${i}`,
}));

// Every word distinct and signed, so scales differ between slots.
const captures: Taken = {
  kind: "taken",
  layers: [L0, L1],
  words: Float32Array.from(
    { length: WORDS },
    (_, i) => (i % 3 === 0 ? -1 : 1) * (i + 1) * 0.25,
  ),
  position: 4,
  input,
};

test("the stages in the order the layer computes them, captured or not", () => {
  const pass = passOf(captures, 1);
  assert.deepEqual(
    pass.stages.map((s) => [s.id, s.name, s.unit, s.rows.length, s.captured]),
    [
      ["residual", "Residual in", "dimension", 1, true],
      ["norm", "RMSNorm", "dimension", 1, false],
      ["query", "Q", "dimension", H, true],
      ["keyValue", "K, V", "dimension", 2, false],
      ["attention", "Attention", "key", H, true],
      ["headsValue", "Heads × V", "dimension", H, false],
      ["attentionAdded", "Attention added", "dimension", 1, true],
      ["mlpNorm", "RMSNorm", "dimension", 1, false],
      ["gateUp", "Gate, up", "neuron", 2, false],
      ["neurons", "MLP neurons", "neuron", 1, true],
      ["mlpAdded", "MLP added", "dimension", 1, true],
    ],
  );
  assert.equal(
    passOf(captures, 0).stages[0]?.name,
    "Residual in (the embedding)",
  );
});

test("every width comes from the captures", () => {
  const widths = passOf(captures, 1).stages.map((s) => [s.id, s.width]);
  assert.deepEqual(widths, [
    ["residual", E],
    ["norm", E],
    ["query", D],
    ["keyValue", D],
    ["attention", 5],
    ["headsValue", D],
    ["attentionAdded", E],
    ["mlpNorm", E],
    ["gateUp", F],
    ["neurons", F],
    ["mlpAdded", E],
  ]);
});

test("captured rows lie at their slots; rows not captured are null", () => {
  const rows = Object.fromEntries(
    passOf(captures, 1).stages.map((s) => [
      s.id,
      s.rows.map((r) => [r.label, r.first]),
    ]),
  );
  assert.deepEqual(rows, {
    residual: [["", L1.residual.first]],
    norm: [["", null]],
    query: [
      ["head 0", L1.query.first],
      ["head 1", L1.query.first + D],
    ],
    keyValue: [
      ["K", null],
      ["V", null],
    ],
    attention: [
      ["head 0", L1.weights.first],
      ["head 1", L1.weights.first + 16],
    ],
    headsValue: [
      ["head 0", null],
      ["head 1", null],
    ],
    attentionAdded: [["", L1.attentionAdded.first]],
    mlpNorm: [["", null]],
    gateUp: [
      ["gate", null],
      ["up", null],
    ],
    neurons: [["", L1.neurons.first]],
    mlpAdded: [["", L1.mlpAdded.first]],
  });
});

test("attention is as wide as the keys read: a full layer back to position 0", () => {
  const pass = passOf(captures, 1);
  const attention = pass.stages.find((s) => s.id === "attention");
  assert.equal(attention?.width, 5); // position + 1, not the slot's 16
  assert.equal(pass.firstKey, 0);
  assert.equal(pass.window, 16);
  assert.equal(pass.full, true);
});

test("a local layer the input outgrows reads its last `keys` tokens", () => {
  const pass = passOf(captures, 0);
  const attention = pass.stages.find((s) => s.id === "attention");
  assert.equal(attention?.width, 2);
  assert.deepEqual(
    attention?.rows.map((r) => r.first),
    [L0.weights.first, L0.weights.first + 2],
  );
  assert.equal(pass.firstKey, 3); // keys at positions 3 and 4
  assert.equal(pass.window, 2);
  assert.equal(pass.full, false);
});

test("a layer outside the captures, or not whole, or a ragged query, is refused", () => {
  for (const bad of [-1, 2, 0.5, Number.NaN]) {
    assert.throws(() => passOf(captures, bad), RangeError);
  }
  const ragged: Taken = {
    ...captures,
    layers: [{ ...L0, query: { first: L0.query.first, width: 5 } }, L1],
  };
  assert.throws(() => passOf(ragged, 0), RangeError);
});

test("the scales: the Map's for its columns, one across the layers for Q, 1 for attention", () => {
  const across = (pick: (l: LayerSlots) => { first: number; width: number }) =>
    scaleAcross(
      captures.layers.map((l) => {
        const slot = pick(l);
        return captures.words.subarray(slot.first, slot.first + slot.width);
      }),
    );
  assert.deepEqual(passScales(captures), {
    residual: across((l) => l.residual),
    attentionAdded: across((l) => l.attentionAdded),
    neurons: across((l) => l.neurons),
    mlpAdded: across((l) => l.mlpAdded),
    query: across((l) => l.query),
    attention: 1,
  });
});
