// Axis H (view): the pins held for the next run, as chips: on the prompt
// box and in the workbench's bar, wherever Run is.
//
// Contract
//   - PinChips({session, pins, onPick}): nothing when no pin is held;
//     otherwise a list "Pins held for the next run", one chip a group of
//     pins (session/pins.ts, groupsOf: a layer and one edit), in the order
//     the groups began:
//       a group of one neuron: button "Show layer {l} neuron {n}" reading
//         "L{l} #{n}" and its edit ("= 8", "× −4", "zero"); button "Unpin
//         layer {l} neuron {n}"
//       a group of k neurons: button "Show layer {l}: {k} neurons {edit}"
//         reading "L{l} · {k} neurons" and its edit; button "Unpin layer
//         {l}: {k} neurons {edit}"
//     "Show" calls onPick with the group's neurons, as a selection;
//     "Unpin" removes the group's pins (session.unpinAll).
//
// Design
//   - Grouped, as a range is pinned at once: a box of 300 neurons is one
//     chip, not 300.

import type { Selection } from "../../App.tsx";
import type { Edit } from "../../engine/edits.ts";
import { groupsOf } from "../../session/pins.ts";
import type { Session } from "../../session/session.ts";

const MINUS = "−";
const COUNT = new Intl.NumberFormat("en-US");

export function PinChips(props: {
  readonly session: Session;
  readonly pins: readonly Edit[];
  readonly onPick: (chosen: Selection) => void;
}) {
  const { session, pins, onPick } = props;
  if (pins.length === 0) return null;
  return (
    <ul className="chips" aria-label="Pins held for the next run">
      {groupsOf(pins).map((group) => {
        const { layer, neurons } = group;
        const edit = editText(group);
        const single = neurons.length === 1 ? neurons[0] : undefined;
        const what =
          single !== undefined
            ? `layer ${layer} neuron ${single}`
            : `layer ${layer}: ${COUNT.format(neurons.length)} neurons ${edit}`;
        return (
          <li
            key={`${layer}:${group.kind}:${Object.is(group.value, -0) ? "-0" : group.value}`}
            className="chip"
          >
            <button
              type="button"
              className="chip-name"
              aria-label={`Show ${what}`}
              onClick={() =>
                onPick({ layer, neurons: [...neurons].sort((a, b) => a - b) })
              }
            >
              {single !== undefined
                ? `L${layer} #${single}`
                : `L${layer} · ${COUNT.format(neurons.length)} neurons`}{" "}
              {edit}
            </button>
            <button
              type="button"
              className="chip-remove"
              aria-label={`Unpin ${what}`}
              onClick={() => session.unpinAll(layer, neurons)}
            >
              ×
            </button>
          </li>
        );
      })}
    </ul>
  );
}

// A pin's edit, short: "= 8", "× −4", "zero".
export function editText(edit: Pick<Edit, "kind" | "value">): string {
  const value = (v: number) =>
    v < 0 || Object.is(v, -0) ? `${MINUS}${String(-v)}` : String(v);
  switch (edit.kind) {
    case "set":
      return `= ${value(edit.value)}`;
    case "scale":
      return `× ${value(edit.value)}`;
    case "zero":
      return "zero";
  }
}
