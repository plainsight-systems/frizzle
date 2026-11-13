// Axis H (shape): a layer's slot as one strip of the Map, a mark a group of
// its values, each mark the group's strongest value, coloured as a heat map
// of its size on one scale for the whole column. Plain TypeScript, no
// React, no DOM: typed arrays in, typed arrays out.
//
// Contract
//   - strongestOfGroups(values, group): the slot read in groups of `group`
//     consecutive values, the last group shorter when the width is not a
//     multiple; for each group, in order, the index into `values` of its
//     strongest value. Returns an Int32Array of ceil(width / group) indices,
//     empty for an empty slot.
//       strongest   the largest magnitude, either sign; +∞ and −∞ above
//                   every finite value
//       ties        the lowest index: v before −v, 0 before −0, +∞ before
//                   −∞ when +∞ comes first
//       NaN         never chosen, unless every value of the group is NaN:
//                   then the group's first index, so the mark is NaN's own
//                   colour and says so
//     A group that is not a whole number of at least 1 throws a RangeError.
//     The mark is a value the model made, chosen, never combined: no sum,
//     mean or norm of the group.
//   - scaleAcross(slots): the magnitude at which the heat is full, for one
//     column of the Map, every layer's slot of that column together: the
//     99th percentile of all their finite magnitudes (nearest rank, zeros
//     counted), or, when that is 0, the largest finite magnitude; 0 when no
//     finite value is non-zero. Infinities and NaN never enter it. For one
//     slot it is paint's fullAt (paint.ts) for that slot.
//   - paintStrip(values, marks, fullAt, palette): RGBA bytes, marks.length
//     × 4, alpha 255 everywhere; pixel i is heatOf(values[marks[i]], fullAt,
//     palette) (paint.ts). A mark outside `values`, or a ramp of fewer than
//     two colours, throws a RangeError.
//
// Algorithm
//   1. strongestOfGroups: one pass over the slot; for each group keep the
//      first index whose magnitude beats the kept one (strictly), an
//      infinity beating any finite magnitude, a NaN beating nothing; a group
//      that keeps nothing keeps its first index.
//   2. scaleAcross: the finite magnitudes of every slot in one Float32Array,
//      sorted; the one at rank ceil(0.99 × n) (counting from 1); if it is 0,
//      the largest. paint.ts's rule, over the column rather than a layer.
//   3. paintStrip: for each mark, heatOf its value, written at its four
//      bytes.
//
// Design and cost
//   - Strongest of a group, not every value: a strip is a row of a few
//     hundred pixels wide, and the Map's question is which layer holds
//     something strong. A group of 12 gives 576 marks for 6,912 neurons and
//     96 for the embedding's 1,152 (views/deck/LayerMap.tsx). The strongest
//     is selected, so every mark is one value the run computed, which the
//     readout names by index; nothing is computed beside the model.
//   - Magnitude, not sign, as the deck (paint.ts): either sign pushes as
//     far, and a hot mark on the Map is a hot neuron on the deck. The sign
//     is the readout's and the inspector's.
//   - One scale for a column across every layer, unlike the deck's card a
//     layer: the Map compares layers, and a scale of their own would make
//     every layer look alike. Each column its own scale: the residual's
//     magnitudes are not the neurons'.
//   - The 99th percentile, not the largest, for paint.ts's reason: a few
//     huge outliers would leave every other mark dark. With it the
//     strongest 1% of the column are at full heat.
//   - Cost, counted for Gemma 3 1B, when the shown captures change, never a
//     frame: sorts of 179,712 neuron magnitudes and three of 29,952 (26 ×
//     1,152), one pass a slot, and 26 × (576 + 3 × 96) pixels, 89,856
//     bytes, in new arrays.

import { type Palette, heatOf, scaleOf } from "./paint.ts";

export const GROUP = 12;

export function strongestOfGroups(
  values: Float32Array,
  group: number,
): Int32Array {
  if (!Number.isInteger(group) || group < 1) {
    throw new RangeError(
      `a group of ${group}; it must be a whole number of at least 1`,
    );
  }
  const marks = new Int32Array(Math.ceil(values.length / group));
  for (let g = 0; g < marks.length; g++) {
    const first = g * group;
    const end = Math.min(first + group, values.length);
    // Below every magnitude, so the first non-NaN value is kept; NaN's
    // magnitude is NaN, which beats nothing (the header's Algorithm, step 1).
    let kept = -1;
    let keptSize = -1;
    for (let i = first; i < end; i++) {
      const size = Math.abs(values[i] ?? Number.NaN);
      if (size > keptSize) {
        kept = i;
        keptSize = size;
      }
    }
    marks[g] = kept === -1 ? first : kept;
  }
  return marks;
}

export function scaleAcross(slots: readonly Float32Array[]): number {
  // Every slot's words in one array; scaleOf keeps the finite ones.
  const together = new Float32Array(
    slots.reduce((sum, slot) => sum + slot.length, 0),
  );
  let at = 0;
  for (const slot of slots) {
    together.set(slot, at);
    at += slot.length;
  }
  return scaleOf(together);
}

export function paintStrip(
  values: Float32Array,
  marks: Int32Array,
  fullAt: number,
  palette: Palette,
): Uint8Array {
  if (palette.heat.length < 2) {
    throw new RangeError(
      `a heat ramp of ${palette.heat.length} colours; it needs two or more`,
    );
  }
  const pixels = new Uint8Array(marks.length * 4);
  marks.forEach((mark, i) => {
    if (mark < 0 || mark >= values.length) {
      throw new RangeError(
        `mark ${i} is ${mark}, outside a slot of ${values.length}`,
      );
    }
    const rgb = heatOf(values[mark] ?? Number.NaN, fullAt, palette);
    pixels.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
  });
  return pixels;
}
