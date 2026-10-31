// Axis H (view): the neurons each side shows at the branch point, as the
// deck's cards, folded and painted once each time the captures shown
// change.
//
// Contract
//   - neuronsOf(run, which): the neurons at the run's branch point, every
//     layer's, for "reply", the reply to steer from, as its tokens were
//     made (an inspection's at the point when there is one, else its own
//     captures, taken at replyAt: session.ts), or "steered", the steered reply, as its
//     pins changed them (its captures, taken at its own `from`); or the
//     words for why there are none, as the page writes them everywhere:
//     "No run yet" | "Not steered" | "The steered reply branches after
//     another word" | "Waiting" | "None: the reply failed" | "None:
//     {failure}" (an inspection) | "Not captured: the run was cancelled" |
//     "Not read back: {reason}".
//   - cardsOf(neurons): null without captures; otherwise one Card per
//     slot, in layer order, holding the slot's values (a view of the
//     neurons, not a copy), its grid (shape/fold.ts) and its pixels and
//     scale (shape/paint.ts, theme/palette.ts).
//   - facesOf(cards, pins, which): the pixels each card shows. For the
//     reply, each pinned neuron's cell takes the heat of the value its pin
//     will send (session/pins.ts, sentBy, against the card's own value), at
//     the card's own scale (shape/paint.ts, overlay): what the next run is
//     asked to use, before it runs. For the steered reply, the card's own
//     pixels, unchanged: those are what its run used.
//   - valueAt(neurons, ref): the neuron as that side used it, or the words
//     for why there is none: those above, or "Not captured: no layer {l},
//     neuron {n}".

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

export type Which = "reply" | "steered";

type Taken = Extract<Captures, { readonly kind: "taken" }>;

export type Neurons =
  | { readonly kind: "taken"; readonly captures: Taken }
  | { readonly kind: "none"; readonly why: string };

const none = (why: string): Neurons => ({ kind: "none", why });

function fromCaptures(captures: Captures): Neurons {
  switch (captures.kind) {
    case "not-taken":
      return none("Not captured: the run was cancelled");
    case "unread":
      return none(`Not read back: ${captures.reason}`);
    case "taken":
      return { kind: "taken", captures };
  }
}

// A reply not done yet, or failed.
function unfinished(
  reply: Exclude<ReplyState, { readonly kind: "done" }>,
): Neurons {
  return none(reply.kind === "failed" ? "None: the reply failed" : "Waiting");
}

export function neuronsOf(run: Run | null, which: Which): Neurons {
  if (run === null) return none("No run yet");
  if (which === "reply") {
    const { reply, inspection } = run;
    if (reply.kind !== "done") return unfinished(reply);
    if (inspection === null || inspection.at !== run.point) {
      return run.point === run.replyAt
        ? fromCaptures(reply.captures)
        : none("Waiting");
    }
    switch (inspection.state.kind) {
      case "running":
        return none("Waiting");
      case "failed":
        return none(`None: ${inspection.state.failure.message}`);
      case "done":
        return fromCaptures(inspection.state.captures);
    }
  }
  const { branch } = run;
  if (branch === null) return none("Not steered");
  if (branch.from !== run.point) {
    return none("The steered reply branches after another word");
  }
  if (branch.reply.kind !== "done") return unfinished(branch.reply);
  return fromCaptures(branch.reply.captures);
}

export function cardsOf(neurons: Neurons): readonly Card[] | null {
  if (neurons.kind !== "taken") return null;
  const { captures } = neurons;
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
    if (which === "steered" || mine.length === 0) return card.painted.pixels;
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

export function valueAt(neurons: Neurons, ref: NeuronRef): Valued {
  if (neurons.kind === "none") return { kind: "none", text: neurons.why };
  const { captures } = neurons;
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
