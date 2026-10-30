// Axis H (view): one reply's captures as the deck's cards, folded and
// painted once each time the captures shown change.
//
// Contract
//   - cardsOf(captures): null unless the captures are taken; otherwise one
//     Card per slot, in layer order, holding the slot's values (a view of
//     the neurons, not a copy), its grid (shape/fold.ts) and its pixels and
//     scale (shape/paint.ts, theme/palette.ts).
//   - facesOf(cards, pins, which): the pixels each card shows. For the
//     original reply, each pinned neuron's cell takes the heat of the value
//     its pin will send (session/pins.ts, sentBy, against the card's own
//     value), at the card's own scale (shape/paint.ts, overlay): what the
//     next run is asked to use, before it runs. For the edited reply, the
//     card's own pixels, unchanged: those are what its run used.
//   - valueAt(reply, ref): the neuron as that reply's run used it, or the
//     words for why there is none, as the page writes them everywhere:
//     "No run yet" | "No pins" | "Waiting" | "None: the reply failed" |
//     "Not captured: the run was cancelled" | "Not read back: {reason}" |
//     "Not captured: no layer {l}, neuron {n}".

import type { NeuronRef } from "../../App.tsx";
import type { Captures } from "../../protocol/messages.ts";
import type { ReplyState, Run } from "../../session/session.ts";
import { type Grid, fold } from "../../shape/fold.ts";
import type { Edit } from "../../engine/edits.ts";
import { sentBy } from "../../session/pins.ts";
import { type Painted, overlay, paint } from "../../shape/paint.ts";
import { DECK_PALETTE } from "../../theme/palette.ts";

export interface Card {
  readonly values: Float32Array;
  readonly grid: Grid;
  readonly painted: Painted;
}

export type Which = "original" | "edited";

export function replyOf(run: Run | null, which: Which): ReplyState | null {
  if (run === null) return null;
  return which === "original" ? run.original : run.edited;
}

export function capturesOf(reply: ReplyState | null): Captures | null {
  return reply !== null && reply.kind === "done" ? reply.answer.captures : null;
}

export function cardsOf(captures: Captures | null): readonly Card[] | null {
  if (captures === null || captures.kind !== "taken") return null;
  return captures.slots.map((slot) => {
    const grid = fold(slot.width);
    const values = captures.neurons.subarray(
      slot.first,
      slot.first + slot.width,
    );
    return { values, grid, painted: paint(values, grid, DECK_PALETTE) };
  });
}

export function facesOf(
  cards: readonly Card[],
  pins: readonly Edit[],
  which: Which,
): readonly Uint8Array[] {
  return cards.map((card, layer) => {
    const mine = pins.filter(
      (p) => p.layer === layer && p.neuron < card.grid.width,
    );
    if (which === "edited" || mine.length === 0) return card.painted.pixels;
    return overlay(
      card.painted,
      card.grid,
      mine.map((p) => ({
        neuron: p.neuron,
        value: sentBy(p, card.values[p.neuron] ?? Number.NaN),
      })),
      DECK_PALETTE,
    );
  });
}

export type Valued =
  | { readonly kind: "value"; readonly value: number }
  | { readonly kind: "none"; readonly text: string };

// Why a reply has no neurons to show, in the page's words; null when it has.
export function absence(
  run: Run | null,
  reply: ReplyState | null,
): string | null {
  if (run === null) return "No run yet";
  if (reply === null) return "No pins";
  if (reply.kind === "failed") return "None: the reply failed";
  if (reply.kind !== "done") return "Waiting";
  const captures = reply.answer.captures;
  switch (captures.kind) {
    case "not-taken":
      return "Not captured: the run was cancelled";
    case "unread":
      return `Not read back: ${captures.reason}`;
    case "taken":
      return null;
  }
}

export function valueAt(
  run: Run | null,
  reply: ReplyState | null,
  ref: NeuronRef,
): Valued {
  const why = absence(run, reply);
  if (why !== null) return { kind: "none", text: why };
  const captures = capturesOf(reply);
  if (captures === null || captures.kind !== "taken") {
    return { kind: "none", text: "Waiting" };
  }
  const slot = captures.slots[ref.layer];
  if (slot === undefined || ref.neuron >= slot.width) {
    return {
      kind: "none",
      text: `Not captured: no layer ${ref.layer}, neuron ${ref.neuron}`,
    };
  }
  return {
    kind: "value",
    value: captures.neurons[slot.first + ref.neuron] ?? Number.NaN,
  };
}
