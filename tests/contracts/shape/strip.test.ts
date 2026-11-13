// Axis V: the shape suite (tests/README.md), the Map's strips.
//
// Contract locked (app/src/shape/strip.ts): a strip shows each group of a
// slot by its strongest value, a value the model made, chosen, never
// combined, coloured as the deck colours a neuron, on one scale for the
// whole column across the layers:
//   - each mark is the index of its group's largest magnitude, either sign;
//     infinities above every finite value; ties to the lowest index (v
//     before −v, 0 before −0); NaN never chosen unless the whole group is
//     NaN, then the group's first; a short last group; a group that is not
//     a whole number of at least 1 is refused;
//   - the column's scale is the 99th percentile of every slot's finite
//     magnitudes together, so it can differ from every layer's own; the
//     largest when that is 0; 0 with no finite non-zero value; infinities
//     and NaN never set it; for one slot it is paint's fullAt;
//   - a strip's pixel i is heatOf the value mark i selected, at the column's
//     scale, opaque; a mark outside the slot is refused.
//
// Not locked: GROUP's value, 12, which the view chooses; the colours
// between the ramp's ends. Plain arrays, no browser.

import assert from "node:assert/strict";
import { test } from "node:test";

import { fold } from "../../../app/src/shape/fold.ts";
import { type Palette, heatOf, paint } from "../../../app/src/shape/paint.ts";
import {
  paintStrip,
  scaleAcross,
  strongestOfGroups,
} from "../../../app/src/shape/strip.ts";

const PALETTE: Palette = {
  heat: [
    [10, 20, 30],
    [100, 60, 40],
    [250, 200, 110],
  ],
  notANumber: [200, 0, 200],
  empty: [1, 2, 3],
};

const f32 = (values: readonly number[]): Float32Array =>
  Float32Array.from(values);

test("each mark is its group's largest magnitude, either sign, ties to the first", () => {
  const values = f32([0.1, -0.5, 0.5, 2, -2, 1, 0, -0, 0, -3, 1, 3]);
  assert.deepEqual([...strongestOfGroups(values, 3)], [1, 3, 6, 9]);
});

test("infinities beat every finite value; NaN is never chosen unless the group is all NaN", () => {
  const values = f32([
    5,
    Number.NEGATIVE_INFINITY,
    Number.NaN,
    Number.NaN,
    Number.NaN,
    Number.NaN,
    Number.NaN,
    1,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    1e30,
  ]);
  assert.deepEqual([...strongestOfGroups(values, 3)], [1, 3, 7, 9]);
});

test("a short last group is read as it is", () => {
  const values = f32([1, 2, 3, 4, 5, 6, -7]);
  assert.deepEqual([...strongestOfGroups(values, 3)], [2, 5, 6]);
  assert.equal(strongestOfGroups(f32([]), 3).length, 0);
});

test("6,912 neurons in groups of 12 make 576 marks, 1,152 make 96, each in its group", () => {
  for (const [width, marks] of [
    [6_912, 576],
    [1_152, 96],
  ] as const) {
    const values = Float32Array.from({ length: width }, (_, i) =>
      Math.sin(i * 1.7),
    );
    const chosen = strongestOfGroups(values, 12);
    assert.equal(chosen.length, marks);
    chosen.forEach((index, g) => {
      assert.ok(index >= g * 12 && index < (g + 1) * 12, `mark ${g}`);
    });
  }
});

test("a group that is not a whole number of at least 1 is refused", () => {
  for (const group of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(
      () => strongestOfGroups(f32([1, 2]), group),
      RangeError,
      String(group),
    );
  }
});

test("one scale for the column: every layer's magnitudes together, not each layer's own", () => {
  // Together: 100 × 0.5, 98 × 2, 2 × 50; rank 198 of 200 is a 2. On their
  // own, the first slot's scale is 0.5 and the second's 50 (rank 99 of 100).
  const first = f32(Array.from({ length: 100 }, () => -0.5));
  const second = f32([...Array.from({ length: 98 }, () => 2), 50, -50]);
  assert.equal(scaleAcross([first, second]), 2);
  assert.equal(scaleAcross([first]), 0.5);
  assert.equal(scaleAcross([second]), 50);
});

test("for one slot the column's scale is paint's", () => {
  const values = Float32Array.from({ length: 6_912 }, (_, i) =>
    i % 97 === 0 ? 40 + i / 1000 : Math.cos(i) * 0.3,
  );
  assert.equal(
    scaleAcross([values]),
    paint(values, fold(values.length), PALETTE).fullAt,
  );
});

test("a sparse column is scaled by its largest value; none at all scales to 0", () => {
  const sparse = f32(Array.from({ length: 300 }, (_, i) => (i === 7 ? -6 : 0)));
  assert.equal(scaleAcross([sparse, f32([0, -0])]), 6);
  assert.equal(scaleAcross([f32([0, -0]), f32([])]), 0);
  assert.equal(scaleAcross([]), 0);
});

test("infinities and NaN never set the column's scale", () => {
  const plain = f32([1, 2, 3]);
  const odd = f32([
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NaN,
  ]);
  assert.equal(scaleAcross([plain, odd]), scaleAcross([plain]));
});

test("each pixel is the colour of the value its mark selected, at the column's scale", () => {
  const values = f32([0.1, -4, 2, Number.NaN, 0, Number.NEGATIVE_INFINITY]);
  const marks = Int32Array.from([1, 2, 3, 4, 5, 0]);
  const fullAt = 3;
  const pixels = paintStrip(values, marks, fullAt, PALETTE);
  assert.equal(pixels.length, marks.length * 4);
  marks.forEach((mark, i) => {
    const expected = heatOf(values[mark] ?? Number.NaN, fullAt, PALETTE);
    assert.deepEqual(
      [...pixels.subarray(i * 4, i * 4 + 4)],
      [...expected, 255],
      `pixel ${i}`,
    );
  });
});

test("a mark outside the slot, or a ramp of one colour, is refused", () => {
  const values = f32([1, 2]);
  assert.throws(
    () => paintStrip(values, Int32Array.from([2]), 1, PALETTE),
    RangeError,
  );
  assert.throws(
    () => paintStrip(values, Int32Array.from([-1]), 1, PALETTE),
    RangeError,
  );
  assert.throws(
    () =>
      paintStrip(values, Int32Array.from([0]), 1, {
        ...PALETTE,
        heat: [[0, 0, 0]],
      }),
    RangeError,
  );
});
