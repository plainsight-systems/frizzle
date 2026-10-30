// Axis H (session): the pins held for the next run, as a set: adding,
// removing, grouping them, and the value a pin sends. Plain TypeScript, no
// React; session.ts holds the set, the views read it.
//
// Contract
//   - A pin set is at most one edit a neuron (a layer and a neuron), in the
//     order first pinned.
//   - withPins(pins, edits): each edit in turn added at the end, or put in
//     place of the edit on the same layer and neuron, which keeps its place
//     in the order. Returns `pins` itself when every edit is already held
//     as it is, so a pin that changes nothing changes no state.
//   - withoutPins(pins, layer, neurons): the set without those neurons'
//     edits in that layer, the rest in order; `pins` itself when none of
//     them was held.
//   - groupsOf(pins): the set as the chips show it: one group per layer and
//     edit (kind and value, -0 apart from 0), in the order each group's
//     first pin was made, each holding its neurons in pin order. A range
//     pinned at once is one group, however its cells lie on the grid.
//   - sentBy(edit, used): the value the model takes for that neuron when
//     the edit is applied to `used`, the value it would otherwise use: a
//     set sends its value, a zero 0, a scale the f32 product used × factor,
//     as the edit kernel computes it (src/core/edit/edit.wgsl).
//
// Design
//   - Whether a set is valid for the model, its size (the engine takes at
//     most 4,096 edits a run, edit.h) and its layers and neurons, is the
//     engine's to say, refused by name at Run (engine/edits.ts); nothing
//     here limits it, so there is one judge.
//   - sentBy serves the deck's preview of what a pin will send. For a
//     scale, `used` is the deck's captured value; the run may use another
//     when an earlier layer is pinned too, which the deck says.

import type { Edit } from "../engine/edits.ts";

export interface PinGroup {
  readonly layer: number;
  readonly kind: Edit["kind"];
  readonly value: number;
  readonly neurons: readonly number[];
}

export function withPins(
  pins: readonly Edit[],
  edits: readonly Edit[],
): readonly Edit[] {
  const next = [...pins];
  const at = new Map(next.map((p, i) => [key(p.layer, p.neuron), i]));
  let changed = false;
  for (const edit of edits) {
    const k = key(edit.layer, edit.neuron);
    const i = at.get(k);
    if (i === undefined) {
      at.set(k, next.length);
      next.push(edit);
      changed = true;
    } else if (!same(next[i] as Edit, edit)) {
      next[i] = edit;
      changed = true;
    }
  }
  return changed ? next : pins;
}

export function withoutPins(
  pins: readonly Edit[],
  layer: number,
  neurons: readonly number[],
): readonly Edit[] {
  const gone = new Set(neurons);
  const kept = pins.filter((p) => p.layer !== layer || !gone.has(p.neuron));
  return kept.length === pins.length ? pins : kept;
}

export function groupsOf(pins: readonly Edit[]): readonly PinGroup[] {
  const groups: { readonly head: Edit; readonly neurons: number[] }[] = [];
  for (const pin of pins) {
    const group = groups.find(
      (g) =>
        g.head.layer === pin.layer &&
        g.head.kind === pin.kind &&
        Object.is(g.head.value, pin.value),
    );
    if (group === undefined) groups.push({ head: pin, neurons: [pin.neuron] });
    else group.neurons.push(pin.neuron);
  }
  return groups.map(({ head, neurons }) => ({
    layer: head.layer,
    kind: head.kind,
    value: head.value,
    neurons,
  }));
}

export function sentBy(edit: Edit, used: number): number {
  switch (edit.kind) {
    case "set":
      return edit.value;
    case "zero":
      return 0;
    case "scale":
      return Math.fround(used * edit.value);
  }
}

const key = (layer: number, neuron: number) => `${layer}:${neuron}`;

const same = (a: Edit, b: Edit) =>
  a.kind === b.kind && Object.is(a.value, b.value);
