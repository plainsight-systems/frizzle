// Axis H (theme): the deck's colours, as numbers for shape/paint.ts. The
// same night glass as theme.css; the deck view hands this to paint, and
// changing a colour here changes the deck and nothing else (change axes,
// boundary tests: "The deck's colours").
//
// Contract
//   - DECK_PALETTE: each colour an [r, g, b] of integers 0..255.
//       heat        the heat ramp, no push to full, evenly spaced: the
//                   card's dim cell, a step above the card itself; deep
//                   ember; ember; amber; white-hot. Each stop is brighter
//                   (Rec. 709 luma) than the one before, so a hotter cell
//                   never reads as a weaker one.
//       notANumber  magenta: a hue the ramp never passes through, so a NaN
//                   is never read as a size
//       empty       a cell past the last neuron: the card itself
//   - CARD and CARD_RULE: the card's face and edge, as CSS colours.
//   - RING_PIN and RING_SELECTED: the rings round a pinned and a selected
//     neuron, as CSS colours, in hues the ramp never passes through, so a
//     ring is never read as heat.
//
// Design
//   - The card and its dim cell are from the reviewed concept (docs/design/
//     concepts/deck.js and deck.html): zero #18233a, card #121b2c, its rule
//     #2b3a55. The heat ramp replaces the concept's amber-and-blue sign
//     colours (a neuron's sign is the tip's and the inspector's to show).
//   - Luma of the stops: 34, 50, 96, 173, 236.

import type { Palette } from "../shape/paint.ts";

export const DECK_PALETTE: Palette = {
  heat: [
    [24, 35, 58],
    [92, 38, 48],
    [196, 72, 34],
    [255, 160, 60],
    [255, 236, 186],
  ],
  notANumber: [214, 84, 196],
  empty: [18, 27, 44],
};

export const CARD = "#121b2c";
export const CARD_RULE = "#2b3a55";
export const RING_PIN = "#7fd6ff";
export const RING_SELECTED = "#f4f1ea";
