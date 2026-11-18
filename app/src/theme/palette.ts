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
//   - RIDE: the Ride's scene (views/ride/scene.ts), as CSS colours: the
//     banner's night, its road and panes, amber for a word the reply goes
//     on to write and for a positive neuron, cool blue for a negative one,
//     and the coach's cream, as in docs/assets/frizzle-readme.png.
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
export const RIDE = {
  skyTop: "#070c16",
  skyLow: "#16233a",
  star: "#efe6cf",
  moon: "#efe0b8",
  moonShade: "#0b1321",
  dusk: "242,160,70", // the glow at the road's end, as r,g,b
  ridgeFar: "#101a2c",
  ridgeNear: "#0b1422",
  groundTop: "#10192a",
  groundLow: "#060a12",
  road: "#202a3c",
  roadEdge: "#8a7443",
  roadLine: "#e2d3aa",
  glass: "135,165,200", // r,g,b
  glassFocus: "13,21,36", // r,g,b
  paneEdge: "239,230,207", // r,g,b
  word: "238,230,211", // r,g,b
  wordLit: "#ffd27e",
  glow: "#f2b54f",
  positive: "242,181,79", // r,g,b
  negative: "143,183,232", // r,g,b
  pin: "#ffcf73",
  dialLight: "#f0d591",
  dialDark: "#6e5220",
  dialInk: "#241a08",
  coachTop: "#f1e5c6",
  coachMiddle: "#dfcfa6",
  coachLow: "#a98f60",
  coachLine: "#5f4b2a",
  coachRack: "#8a6f43",
  coachGlass: "#17233a",
  coachLit: "rgba(255,207,115,.33)",
  coachFrame: "#d9c79c",
  coachStripe: "#7a2d1f",
  coachTyre: "#10131a",
  coachLamp: "#ff5a3a",
  headlamp: "255,214,140", // r,g,b
} as const;
