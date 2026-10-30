// Axis H (shape): a layer's neurons as the pixels of its deck card, one
// pixel a neuron, coloured as a heat map of its size: how hard it pushes,
// whichever way. Plain TypeScript, no React, no DOM: typed arrays in, typed
// arrays out.
//
// Contract
//   - paint(values, grid, palette): `values` is one slot's words, the
//     neurons as the run used them (length grid.width; anything else throws
//     a RangeError). Returns {pixels, fullAt}:
//       pixels   RGBA bytes, grid.columns × grid.rows × 4, row 0 first (the
//                top row, as fold.ts reads), alpha 255 everywhere. The
//                pixel at cellOf(grid, i) is neuron i's colour; a cell past
//                the last neuron is palette.empty.
//       fullAt   the magnitude at which the heat is full: the 99th
//                percentile of the slot's finite magnitudes (nearest rank,
//                zeros counted), or, when that is 0, the largest finite
//                magnitude; 0 when no finite value is non-zero. The deck's
//                legend states it, rounded.
//   - A neuron's colour depends on its magnitude alone; v and −v are the
//     same colour:
//       0 or -0                 palette.heat[0], the coolest
//       finite                  along palette.heat, reaching its last
//                               colour at fullAt and staying there beyond;
//                               a larger magnitude is never at an earlier
//                               point of the ramp than a smaller one
//       +Infinity, -Infinity    palette.heat's last colour
//       NaN                     palette.notANumber
//     Infinities and NaNs never enter fullAt.
//   - palette.heat is at least two colours, evenly spaced from no push to
//     full; fewer throws a RangeError.
//   - heatOf(v, fullAt, palette): the colour paint gives a neuron of value v
//     in a slot whose scale is fullAt, by the rules above.
//   - overlay(painted, grid, cells, palette): a copy of painted.pixels with
//     each {neuron, value} of `cells` coloured as heatOf(value,
//     painted.fullAt) at the neuron's cell, and every other byte as it was.
//     The scale stays the slot's: an overlaid value never sets it. Throws
//     a RangeError for a neuron outside the grid (fold.ts, cellOf).
//
// Algorithm
//   1. fullAt: the finite magnitudes, sorted; the one at rank
//      ceil(0.99 × n) (counting from 1); if it is 0, the largest.
//   2. Each neuron i: t = min(1, |v| / fullAt) ^ GAMMA, in [0, 1] (0 when
//      fullAt is 0); the colour at t along palette.heat, each channel
//      interpolated between the two stops either side of t and rounded to
//      the nearest integer. Written at its cell's four bytes.
//   3. Cells past the last neuron: palette.empty.
//
// Design and cost
//   - Magnitude, not sign: a neuron adds value × its column of the down
//     projection into the residual, so either sign moves the residual as
//     far, in opposite directions along the same line. The sign is the
//     tip's and the inspector's to give, with the value.
//   - Each layer is scaled against itself: magnitudes grow through the
//     layers, and a scale shared by all would leave the early layers dark.
//   - The 99th percentile, not the largest: a layer's activations have a
//     few huge outliers (layer 0 of the browser scenario's prompt has one
//     near 97 among values mostly under 1), and scaling to the largest
//     leaves every other neuron dark. With the percentile, the strongest 1%
//     are at full heat and the rest spread below. fullAt is presentation,
//     not a finding: every value shown is still the model's own, and the
//     inspector gives each neuron's value.
//   - GAMMA 2 keeps the many small values at the background, so a layer
//     reads as a few hot neurons on a dark card: a neuron at half the scale
//     is a quarter of the way along the ramp, one at a fifth is barely lit.
//     It changes how far along the ramp a cell is, never the order.
//   - overlay is how the deck shows the values pins will send on a card
//     painted from captures (views/deck/cards.ts): the card keeps its own
//     scale, so a pin of 8 in a layer of values under 1 is full heat.
//   - A sort of 6,912 magnitudes and one pass a layer, 26 layers a run,
//     when the shown captures change; a new array a call (27,648 bytes a
//     layer), which the deck uploads as that card's texture.

import { type Grid, cellOf } from "./fold.ts";

export const GAMMA = 2;
export const PERCENTILE = 0.99;

export type Rgb = readonly [number, number, number]; // each 0..255

export interface Palette {
  readonly heat: readonly Rgb[]; // no push … full, evenly spaced
  readonly notANumber: Rgb;
  readonly empty: Rgb;
}

export interface Painted {
  readonly pixels: Uint8Array;
  readonly fullAt: number;
}

export function paint(
  values: Float32Array,
  grid: Grid,
  palette: Palette,
): Painted {
  if (palette.heat.length < 2) {
    throw new RangeError(
      `a heat ramp of ${palette.heat.length} colours; it needs two or more`,
    );
  }
  if (values.length !== grid.width) {
    throw new RangeError(
      `${values.length} values for a slot of ${grid.width} neurons`,
    );
  }
  const fullAt = scaleOf(values);
  const pixels = new Uint8Array(grid.columns * grid.rows * 4);
  // Neuron i is at row floor(i / columns), column i % columns (fold.ts), so
  // in reading order its pixel is simply the i-th.
  for (let i = 0; i < values.length; i++) {
    put(pixels, i, colour(values[i] ?? Number.NaN, fullAt, palette));
  }
  for (let i = values.length; i < grid.columns * grid.rows; i++) {
    put(pixels, i, palette.empty);
  }
  return { pixels, fullAt };
}

export function heatOf(v: number, fullAt: number, palette: Palette): Rgb {
  return colour(v, fullAt, palette);
}

export function overlay(
  painted: Painted,
  grid: Grid,
  cells: readonly { readonly neuron: number; readonly value: number }[],
  palette: Palette,
): Uint8Array {
  const pixels = painted.pixels.slice();
  for (const { neuron, value } of cells) {
    const { column, row } = cellOf(grid, neuron);
    put(
      pixels,
      row * grid.columns + column,
      colour(value, painted.fullAt, palette),
    );
  }
  return pixels;
}

// The header's Algorithm, step 1.
function scaleOf(values: Float32Array): number {
  const magnitudes = Float32Array.from(
    values.filter((v) => Number.isFinite(v)),
    Math.abs,
  ).sort();
  if (magnitudes.length === 0) return 0;
  const rank = Math.ceil(PERCENTILE * magnitudes.length) - 1;
  const at = magnitudes[rank] ?? 0;
  return at > 0 ? at : (magnitudes[magnitudes.length - 1] ?? 0);
}

// A neuron's colour (the header's Contract).
function colour(v: number, fullAt: number, palette: Palette): Rgb {
  if (Number.isNaN(v)) return palette.notANumber;
  if (!Number.isFinite(v)) return along(palette.heat, 1);
  // fullAt is 0 only when no finite value is non-zero, so v is 0 there.
  const t = fullAt === 0 ? 0 : Math.min(1, Math.abs(v) / fullAt) ** GAMMA;
  return along(palette.heat, t);
}

// The colour at t in [0, 1] along evenly spaced stops (two or more).
function along(stops: readonly Rgb[], t: number): Rgb {
  const span = stops.length - 1;
  const at = Math.min(span - 1, Math.floor(t * span));
  const f = t * span - at;
  const a = stops[at] as Rgb;
  const b = stops[at + 1] as Rgb;
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function put(pixels: Uint8Array, at: number, rgb: Rgb): void {
  const o = at * 4;
  pixels[o] = rgb[0];
  pixels[o + 1] = rgb[1];
  pixels[o + 2] = rgb[2];
  pixels[o + 3] = 255;
}
