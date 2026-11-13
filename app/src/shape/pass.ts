// Axis H (shape): one layer's pass, the stages it computes at the last
// position in order, each at its real width, and where each captured one
// lies in the store, so the pass map draws the computation's shape and
// colours only what the run captured. Plain TypeScript, no React, no DOM.
//
// Contract
//   - passOf(captures, layer): the layer's pass from taken captures
//     (protocol/messages.ts): its window, whether it is full attention, the
//     position of its first attention key, and its stages in the order the
//     layer computes them. With E the residual's width, F the neurons', H
//     the query heads, d = query width / H, the head dimension, and n =
//     min(position + 1, keys), the keys the last position read
//     (capture/weights.h):
//       id              name                rows × width, unit    captured
//       residual        "Residual in"       1 × E, dimension      residual
//       norm            "RMSNorm"           1 × E, dimension      no
//       query           "Q"                 H × d, dimension      query
//       keyValue        "K, V"              2 × d, dimension      no
//       attention       "Attention"         H × n, key            weights
//       headsValue      "Heads × V"         H × d, dimension      no
//       attentionAdded  "Attention added"   1 × E, dimension      attentionAdded
//       mlpNorm         "RMSNorm"           1 × E, dimension      no
//       gateUp          "Gate, up"          2 × F, neuron         no
//       neurons         "MLP neurons"       1 × F, neuron         neurons
//       mlpAdded        "MLP added"         1 × E, dimension      mlpAdded
//     Layer 0's residual is named "Residual in (the embedding)": it is the
//     embedding, scaled, before any layer.
//     Row labels: "" for a stage of one row; "head h" for Q, attention and
//     heads × V, h from 0; "K", "V"; "gate", "up".
//     A captured row's `first` is the index in the store's words of its
//     first value, the row being words[first .. first + width − 1]:
//     the slot's first for a one-row stage; query.first + h × d for Q's
//     head h; weights.first + h × keys for attention's head h, of which
//     only the first n words are the row (the rest of a head's keys are
//     not live, and are never part of it). An uncaptured row's `first` is
//     null, and so is every row of a stage not captured.
//     Key i of the attention stage is the input's token at firstKey + i,
//     firstKey = position − n + 1.
//   - Every width comes from the captures, none from Gemma 3 1B's numbers.
//     K and V are drawn one key head wide (d): the captures do not say how
//     many key heads a model has; Gemma 3 1B has one, so for the pinned
//     model that is K and V whole.
//   - A layer that is not a whole number below the captures' layer count,
//     or a query width that is not a whole multiple of the query heads,
//     throws a RangeError.
//   - passScales(captures): the magnitude at which each captured stage's
//     heat is full, one for each across every layer, so stepping layers
//     compares them: residual, attentionAdded, neurons and mlpAdded each
//     the Map's column scale (shape/strip.ts, scaleAcross over that slot
//     of every layer), so a mark hot on one is hot on the other; query
//     scaleAcross over every layer's query slot; attention 1, the weights
//     being probabilities.
//
// Algorithm
//   1. passOf: check the layer; read its slots; d = query.width / H (whole,
//      or refused); n = min(position + 1, keys); the stages in the table's
//      order, each row's first from its slot, or null.
//   2. passScales: one scaleAcross a captured stage other than attention,
//      over views of the words (subarray, never copies).
//
// Design and cost
//   - The layer alone, not the embedding before layer 0 nor the final
//     norm and logits after layer 25: the pass map shows one layer, and the
//     vocabulary's width is not on the page; the next-token prediction was
//     removed (research/plan.md). Layer 0's first stage says it is the
//     embedding.
//   - No stage the page would compute: "residual + attention" (X plus what
//     attention added) is not a slot, so it is not a stage; the stages that
//     are drawn without values (norms, K and V, heads × V, gate and up) are
//     outlined as not captured, at their widths, for the layer's shape.
//     "Residual out" is the next layer's residual in, where ‹ › leads.
//   - The attention stage n keys wide, not its slot's: a global layer's
//     slot holds 32,768 keys a head, of which a short prompt's last position
//     reads a few dozen.
//   - Rows by index into the words, not copied arrays: the view takes
//     subarrays, and passOf runs each time the front layer changes.
//   - Cost: passOf is a few dozen objects; passScales sorts the four Map
//     columns' magnitudes (179,712 and three of 29,952 for Gemma 3 1B) and
//     the query slots' (26 × 1,024 = 26,624), once each time the captures
//     change.

import type { Captures } from "../protocol/messages.ts";
import { scaleAcross } from "./strip.ts";

type Taken = Extract<Captures, { readonly kind: "taken" }>;

export type StageId =
  | "residual"
  | "norm"
  | "query"
  | "keyValue"
  | "attention"
  | "headsValue"
  | "attentionAdded"
  | "mlpNorm"
  | "gateUp"
  | "neurons"
  | "mlpAdded";

// The stages that hold values the run captured.
export type CapturedId =
  | "residual"
  | "query"
  | "attention"
  | "attentionAdded"
  | "neurons"
  | "mlpAdded";

export type Unit = "dimension" | "neuron" | "key";

export interface Row {
  readonly label: string;
  readonly first: number | null; // index in the store's words, or not captured
}

export interface Stage {
  readonly id: StageId;
  readonly name: string;
  readonly unit: Unit;
  readonly width: number; // values a row
  readonly rows: readonly Row[];
  readonly captured: boolean;
}

export interface Pass {
  readonly layer: number;
  readonly window: number; // the layer's attention window, in tokens
  readonly full: boolean; // full attention
  readonly firstKey: number; // the input position of attention's key 0
  readonly stages: readonly Stage[];
}

export type PassScales = Readonly<Record<CapturedId, number>>;

export function passOf(captures: Taken, layer: number): Pass {
  const slots = captures.layers[layer];
  if (!Number.isInteger(layer) || slots === undefined) {
    throw new RangeError(
      `layer ${layer}; the captures hold layers 0 to ${captures.layers.length - 1}`,
    );
  }
  const heads = slots.queryHeads;
  const d = slots.query.width / heads;
  if (!Number.isInteger(d)) {
    throw new RangeError(
      `layer ${layer}'s query is ${slots.query.width} wide, not a whole multiple of its ${heads} heads`,
    );
  }
  const E = slots.residual.width;
  const F = slots.neurons.width;
  const n = Math.min(captures.position + 1, slots.keys);
  const one = (first: number | null): readonly Row[] => [{ label: "", first }];
  const perHead = (at: (h: number) => number | null): readonly Row[] =>
    Array.from({ length: heads }, (_, h) => ({
      label: `head ${h}`,
      first: at(h),
    }));
  const pair = (a: string, b: string): readonly Row[] => [
    { label: a, first: null },
    { label: b, first: null },
  ];
  const stage = (
    id: StageId,
    name: string,
    unit: Unit,
    width: number,
    rows: readonly Row[],
  ): Stage => ({
    id,
    name,
    unit,
    width,
    rows,
    captured: rows.every((r) => r.first !== null),
  });
  return {
    layer,
    window: slots.window,
    full: slots.full,
    firstKey: captures.position - n + 1,
    stages: [
      stage(
        "residual",
        layer === 0 ? "Residual in (the embedding)" : "Residual in",
        "dimension",
        E,
        one(slots.residual.first),
      ),
      stage("norm", "RMSNorm", "dimension", E, one(null)),
      stage(
        "query",
        "Q",
        "dimension",
        d,
        perHead((h) => slots.query.first + h * d),
      ),
      stage("keyValue", "K, V", "dimension", d, pair("K", "V")),
      stage(
        "attention",
        "Attention",
        "key",
        n,
        perHead((h) => slots.weights.first + h * slots.keys),
      ),
      stage(
        "headsValue",
        "Heads × V",
        "dimension",
        d,
        perHead(() => null),
      ),
      stage(
        "attentionAdded",
        "Attention added",
        "dimension",
        E,
        one(slots.attentionAdded.first),
      ),
      stage("mlpNorm", "RMSNorm", "dimension", E, one(null)),
      stage("gateUp", "Gate, up", "neuron", F, pair("gate", "up")),
      stage("neurons", "MLP neurons", "neuron", F, one(slots.neurons.first)),
      stage("mlpAdded", "MLP added", "dimension", E, one(slots.mlpAdded.first)),
    ],
  };
}

export function passScales(captures: Taken): PassScales {
  const across = (
    slot: "residual" | "attentionAdded" | "neurons" | "mlpAdded" | "query",
  ): number =>
    scaleAcross(
      captures.layers.map((l) =>
        captures.words.subarray(l[slot].first, l[slot].first + l[slot].width),
      ),
    );
  return {
    residual: across("residual"),
    attentionAdded: across("attentionAdded"),
    neurons: across("neurons"),
    mlpAdded: across("mlpAdded"),
    query: across("query"),
    attention: 1,
  };
}
