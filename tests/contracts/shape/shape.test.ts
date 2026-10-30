// Axis V: the shape suite (tests/README.md).
//
// Contract locked (app/src/shape/fold.ts, paint.ts, rank.ts, numbers.ts):
// what the deck shows at a cell is the neuron folded there, a heat map of
// its size, whichever its sign; the inspector ranks neurons by strength;
// sizes and values read as people read them:
//   - a slot folds 96 to a row in reading order, neuron i at column i % 96,
//     row floor(i / 96), and each cell back to its neuron; a short last row
//     ends in empty cells; what is not a grid or a neuron is refused; a box
//     of cells is its neurons, ascending, clipped to the grid, either
//     corner first, past the last neuron skipped;
//   - each neuron's pixel is at its cell; v and −v are the same colour; 0
//     and -0 are the ramp's first colour and the scale's magnitude its last;
//     a larger magnitude is never further back along the ramp than a
//     smaller one; infinities are full heat and NaN its own colour, and
//     none of them sets the scale; the scale is the 99th percentile of the
//     finite magnitudes, so a few outliers sit at full heat without
//     darkening the rest, or the largest when that percentile is 0; a slot
//     with no finite non-zero value has a scale of 0; values that are not
//     the slot's width, and a ramp of fewer than two colours, are refused;
//     heatOf is the colour paint gives; an overlay colours its cells as
//     heatOf at the slot's own scale and leaves every other byte;
//   - the strongest neurons are the largest magnitudes, either sign, ties
//     in neuron order, infinities first, NaN never;
//   - bytes read as whole decimal megabytes; a value with its sign and two
//     places; a scale's end with two significant digits.
//
// Not locked: the gamma, or any colour between the ramp's ends; the deck's
// own palette (theme/palette.ts) and its scene. Plain arrays, no browser.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  type Grid,
  cellOf,
  fold,
  neuronAt,
  neuronsIn,
} from "../../../app/src/shape/fold.ts";
import {
  megabytes,
  signed,
  significant,
} from "../../../app/src/shape/numbers.ts";
import {
  type Palette,
  type Rgb,
  heatOf,
  overlay,
  paint,
} from "../../../app/src/shape/paint.ts";
import { strongest } from "../../../app/src/shape/rank.ts";

// A ramp whose every channel rises, so "further along" is "further from
// the first colour" in each channel.
const PALETTE: Palette = {
  heat: [
    [10, 20, 30],
    [100, 60, 40],
    [250, 200, 110],
  ],
  notANumber: [200, 0, 200],
  empty: [1, 2, 3],
};
const COOL = PALETTE.heat[0] as Rgb;
const HOT = PALETTE.heat[PALETTE.heat.length - 1] as Rgb;

// The colour paint gave the cell at (column, row), and its alpha.
function pixelAt(
  pixels: Uint8Array,
  grid: Grid,
  column: number,
  row: number,
): { readonly rgb: Rgb; readonly alpha: number } {
  const at = (row * grid.columns + column) * 4;
  return {
    rgb: [pixels[at] ?? -1, pixels[at + 1] ?? -1, pixels[at + 2] ?? -1],
    alpha: pixels[at + 3] ?? -1,
  };
}

function colourOf(pixels: Uint8Array, grid: Grid, neuron: number): Rgb {
  const { column, row } = cellOf(grid, neuron);
  return pixelAt(pixels, grid, column, row).rgb;
}

// How far each channel sits from the ramp's first colour.
const distance = (rgb: Rgb): number[] =>
  rgb.map((c, k) => Math.abs(c - (COOL[k] ?? 0)));

test("6,912 neurons fold to 96 × 72 in reading order, and each cell back", () => {
  const grid = fold(6912);
  assert.deepEqual(grid, { width: 6912, columns: 96, rows: 72 });
  for (let i = 0; i < 6912; i++) {
    const cell = cellOf(grid, i);
    assert.deepEqual(cell, { column: i % 96, row: Math.floor(i / 96) });
    assert.equal(neuronAt(grid, cell.column, cell.row), i);
  }
  assert.equal(neuronAt(grid, 96, 0), null);
  assert.equal(neuronAt(grid, 0, 72), null);
  assert.equal(neuronAt(grid, -1, 0), null);
});

test("a short last row ends in empty cells", () => {
  const grid = fold(100);
  assert.deepEqual(grid, { width: 100, columns: 96, rows: 2 });
  assert.equal(neuronAt(grid, 3, 1), 99);
  assert.equal(neuronAt(grid, 4, 1), null);
  const { pixels } = paint(new Float32Array(100).fill(1), grid, PALETTE);
  assert.equal(pixels.length, 96 * 2 * 4);
  assert.deepEqual(pixelAt(pixels, grid, 3, 1).rgb, HOT);
  for (let column = 4; column < 96; column++) {
    assert.deepEqual(pixelAt(pixels, grid, column, 1).rgb, PALETTE.empty);
  }
});

test("a box of cells is its neurons, clipped, either corner first", () => {
  const grid = fold(100, 10); // ten rows of ten
  const box = [12, 13, 14, 22, 23, 24];
  assert.deepEqual(
    neuronsIn(grid, { column: 2, row: 1 }, { column: 4, row: 2 }),
    box,
  );
  assert.deepEqual(
    neuronsIn(grid, { column: 4, row: 2 }, { column: 2, row: 1 }),
    box,
  );
  assert.deepEqual(
    neuronsIn(grid, { column: 7, row: 7 }, { column: 7, row: 7 }),
    [77],
  );
  assert.deepEqual(
    neuronsIn(grid, { column: -5, row: -1 }, { column: 1, row: 0 }),
    [0, 1],
  );
  assert.deepEqual(
    neuronsIn(grid, { column: 8.6, row: 9.2 }, { column: 40, row: 30 }),
    [98, 99],
  );
  assert.deepEqual(
    neuronsIn(grid, { column: 11, row: 0 }, { column: 20, row: 3 }),
    [],
  );
  const short = fold(25, 10); // the last row holds 5
  assert.deepEqual(
    neuronsIn(short, { column: 3, row: 1 }, { column: 7, row: 2 }),
    [13, 14, 15, 16, 17, 23, 24],
  );
});

test("what is not a grid or a neuron is refused", () => {
  assert.throws(() => fold(0), RangeError);
  assert.throws(() => fold(1.5), RangeError);
  assert.throws(() => fold(10, 0), RangeError);
  const grid = fold(10);
  assert.throws(() => cellOf(grid, -1), RangeError);
  assert.throws(() => cellOf(grid, 10), RangeError);
  assert.throws(() => cellOf(grid, 0.5), RangeError);
  assert.throws(() => paint(new Float32Array(9), grid, PALETTE), RangeError);
  assert.throws(
    () => paint(new Float32Array(10), grid, { ...PALETTE, heat: [[0, 0, 0]] }),
    RangeError,
  );
});

test("each neuron's pixel is at its cell, opaque", () => {
  const grid = fold(300);
  const values = new Float32Array(300);
  values[0] = 8;
  values[97] = -8;
  values[150] = Number.NaN;
  values[299] = 8;
  const { pixels } = paint(values, grid, PALETTE);
  for (let i = 0; i < 300; i++) {
    const expected =
      i === 0 || i === 97 || i === 299
        ? HOT
        : i === 150
          ? PALETTE.notANumber
          : COOL;
    assert.deepEqual(colourOf(pixels, grid, i), expected, `neuron ${i}`);
  }
  for (let row = 0; row < grid.rows; row++) {
    for (let column = 0; column < grid.columns; column++) {
      assert.equal(pixelAt(pixels, grid, column, row).alpha, 255);
    }
  }
});

test("zero is the ramp's first colour; the scale's magnitude its last, either sign", () => {
  const grid = fold(6);
  const { pixels, fullAt } = paint(
    new Float32Array([0, -0, 4, -2, 2, -4]),
    grid,
    PALETTE,
  );
  assert.equal(fullAt, 4);
  assert.deepEqual(colourOf(pixels, grid, 0), COOL);
  assert.deepEqual(colourOf(pixels, grid, 1), COOL);
  assert.deepEqual(colourOf(pixels, grid, 2), HOT);
  assert.deepEqual(colourOf(pixels, grid, 5), HOT);
  assert.deepEqual(colourOf(pixels, grid, 3), colourOf(pixels, grid, 4));
});

test("a larger magnitude is never further back along the ramp, and the sign never matters", () => {
  const steps = 64;
  const values = new Float32Array(steps * 2);
  for (let k = 0; k < steps; k++) {
    values[k] = (k + 1) / steps;
    values[steps + k] = -(k + 1) / steps;
  }
  const grid = fold(values.length);
  const { pixels } = paint(values, grid, PALETTE);
  for (let k = 0; k < steps; k++) {
    assert.deepEqual(
      colourOf(pixels, grid, k),
      colourOf(pixels, grid, steps + k),
      `±${(k + 1) / steps}`,
    );
  }
  for (const start of [0, steps]) {
    for (let k = 1; k < steps; k++) {
      const smaller = distance(colourOf(pixels, grid, start + k - 1));
      const larger = distance(colourOf(pixels, grid, start + k));
      larger.forEach((d, channel) =>
        assert.ok(
          d >= (smaller[channel] ?? 0),
          `neuron ${start + k}, channel ${channel}`,
        ),
      );
    }
  }
  assert.deepEqual(colourOf(pixels, grid, steps - 1), HOT);
  assert.deepEqual(colourOf(pixels, grid, 2 * steps - 1), HOT);
});

test("infinities are full heat, NaN its own colour, and none sets the scale", () => {
  const grid = fold(5);
  const { pixels, fullAt } = paint(
    new Float32Array([1, Infinity, -Infinity, Number.NaN, -0.5]),
    grid,
    PALETTE,
  );
  assert.equal(fullAt, 1);
  assert.deepEqual(colourOf(pixels, grid, 0), HOT);
  assert.deepEqual(colourOf(pixels, grid, 1), HOT);
  assert.deepEqual(colourOf(pixels, grid, 2), HOT);
  assert.deepEqual(colourOf(pixels, grid, 3), PALETTE.notANumber);
  const half = colourOf(pixels, grid, 4);
  assert.notDeepEqual(half, COOL);
  assert.notDeepEqual(half, HOT);
});

test("a slot with no finite non-zero value has a scale of 0", () => {
  const zeros = fold(4);
  const allZero = paint(new Float32Array(4), zeros, PALETTE);
  assert.equal(allZero.fullAt, 0);
  for (let i = 0; i < 4; i++) {
    assert.deepEqual(colourOf(allZero.pixels, zeros, i), COOL);
  }
  const odd = fold(3);
  const noFinite = paint(
    new Float32Array([Number.NaN, Infinity, 0]),
    odd,
    PALETTE,
  );
  assert.equal(noFinite.fullAt, 0);
  assert.deepEqual(colourOf(noFinite.pixels, odd, 0), PALETTE.notANumber);
  assert.deepEqual(colourOf(noFinite.pixels, odd, 1), HOT);
  assert.deepEqual(colourOf(noFinite.pixels, odd, 2), COOL);
});

test("the scale is the 99th percentile: outliers at full heat without darkening the rest", () => {
  const values = new Float32Array(1000).fill(1);
  for (let i = 990; i < 1000; i++) values[i] = 100;
  const grid = fold(1000);
  const { pixels, fullAt } = paint(values, grid, PALETTE);
  assert.equal(fullAt, 1);
  assert.deepEqual(colourOf(pixels, grid, 0), HOT);
  assert.deepEqual(colourOf(pixels, grid, 999), HOT);
});

test("a sparse slot is scaled by its largest value", () => {
  const values = new Float32Array(1000);
  values[500] = -5;
  const grid = fold(1000);
  const { pixels, fullAt } = paint(values, grid, PALETTE);
  assert.equal(fullAt, 5);
  assert.deepEqual(colourOf(pixels, grid, 500), HOT);
  assert.deepEqual(colourOf(pixels, grid, 0), COOL);
});

test("heatOf is the colour paint gives each value", () => {
  const values = new Float32Array([0, 0.3, -0.7, 2, Infinity, Number.NaN, -9]);
  const grid = fold(values.length);
  const { pixels, fullAt } = paint(values, grid, PALETTE);
  values.forEach((v, i) =>
    assert.deepEqual(
      colourOf(pixels, grid, i),
      heatOf(v, fullAt, PALETTE),
      `${v}`,
    ),
  );
});

test("an overlay colours its cells at the slot's scale and leaves the rest", () => {
  const values = new Float32Array(200).fill(0.5);
  values[3] = 1;
  const grid = fold(200);
  const painted = paint(values, grid, PALETTE);
  const before = painted.pixels.slice();
  const pixels = overlay(
    painted,
    grid,
    [
      { neuron: 5, value: 8 },
      { neuron: 150, value: 0 },
      { neuron: 160, value: -0.5 },
    ],
    PALETTE,
  );
  assert.deepEqual(painted.pixels, before, "the painted card is not changed");
  assert.deepEqual(colourOf(pixels, grid, 5), HOT);
  assert.deepEqual(colourOf(pixels, grid, 150), COOL);
  assert.deepEqual(
    colourOf(pixels, grid, 160),
    heatOf(-0.5, painted.fullAt, PALETTE),
  );
  for (let i = 0; i < pixels.length; i++) {
    const neuron = Math.floor(i / 4);
    if (neuron === 5 || neuron === 150 || neuron === 160) continue;
    assert.equal(pixels[i], before[i], `byte ${i}`);
  }
  assert.throws(
    () => overlay(painted, grid, [{ neuron: 200, value: 1 }], PALETTE),
    RangeError,
  );
});

test("the strongest are the largest magnitudes, either sign, ties in order", () => {
  const values = new Float32Array([0.5, -3, Number.NaN, 3, Infinity, -0.1, 2]);
  assert.deepEqual(strongest(values, 4), [4, 1, 3, 6]);
  assert.deepEqual(strongest(values, 10), [4, 1, 3, 6, 0, 5]);
  assert.deepEqual(strongest(values, 0), []);
  assert.throws(() => strongest(values, -1), RangeError);
});

test("sizes and values read as people read them", () => {
  assert.equal(megabytes(721_918_496), "722 MB");
  assert.equal(megabytes(312_400_000), "312 MB");
  assert.equal(megabytes(0), "0 MB");
  assert.equal(megabytes(1200), "<1 MB");
  assert.equal(signed(8), "+8.00");
  assert.equal(signed(-0.0798), "\u22120.08");
  assert.equal(signed(-0.001), "0.00");
  assert.equal(signed(-0), "0.00");
  assert.equal(signed(-Infinity), "\u2212\u221e");
  assert.equal(signed(Number.NaN), "NaN");
  assert.equal(significant(97.39), "97");
  assert.equal(significant(0.5999138951301575), "0.6");
  assert.equal(significant(12.79), "13");
  assert.equal(significant(1.4), "1.4");
  assert.equal(significant(0), "0");
});
