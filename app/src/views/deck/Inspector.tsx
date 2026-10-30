// Axis H (view): the inspector, the workbench's right column. Which
// reply's neurons the deck shows; the strongest neurons in the front
// layer; the selection: one neuron, with its value in both replies, or
// many, with how many are pinned; Set, Scale × and Zero for what is
// selected; and going to a neuron by number.
//
// Contract
//   - Its names are the page's interface, which the browser scenario uses:
//       radio "Original reply" (chosen at first), radio "Edited reply"
//         (disabled while the run shown has no pins), in a group "Neurons
//         from"
//       a list "Strongest in layer {l}": eight buttons, the front layer's
//         largest magnitudes (shape/rank.ts), each "#{n}", a bar of its
//         size in one colour, as the deck's heat (either sign), and the
//         value with its sign and two places; pressing one selects it
//         alone (pressed while it is in the selection)
//       with one neuron selected: heading "Neuron #{n} · layer {l}"; status
//         "Neuron original value" and status "Neuron edited value", each
//         the value with its sign and three places, the exact f32 in a
//         <data value> (String, "-0" for negative zero), or the page's
//         words for why there is none (cards.ts); status "Neuron pin":
//         "Not pinned" | "Pinned: set to {v}" | "Pinned: scaled by {v}" |
//         "Pinned: zeroed"; textbox "Pin value"; buttons "Set", "Scale ×",
//         "Zero", and "Unpin" when pinned
//       with k > 1 neurons selected: heading "{k} neurons · layer {l}";
//         status "Selection pins": "None pinned" | "{p} of {k} pinned";
//         textbox "Pin value"; buttons "Set", "Scale ×", "Zero", and
//         "Unpin" when any is pinned
//       textbox "Go to layer", textbox "Go to neuron", button "Go": selects
//         that neuron alone (its layer comes to the front)
//     A field that does not parse (fields.ts) shows its problem beside it,
//     and the button it would feed does nothing.
//   - Set and Scale × pin every selected neuron with the value typed, as
//     one change; Zero pins them to zero; Unpin removes their pins
//     (session.pinAll, session.unpinAll). The pins are held for the next
//     run (the prompt shows them). How many a run takes is the engine's to
//     say (4,096, edit.h): past it, Run fails by name (session/pins.ts).
//   - Without a selection: "Click a neuron on the deck, drag across
//     several, or go to one by number."
//
// Design
//   - The values are read from each reply's own captures, whichever reply
//     the deck shows, so a pin's effect reads directly: original, then
//     edited.
//   - Rounded for reading; exact in <data value> for anyone who needs it.

import { useId, useState } from "react";

import { type NeuronRef, type Selection, one } from "../../App.tsx";
import type { Edit } from "../../engine/edits.ts";
import type { Session, SessionState } from "../../session/session.ts";
import { signed } from "../../shape/numbers.ts";
import { strongest } from "../../shape/rank.ts";
import { parseValue, parseWord } from "../prompt/fields.ts";
import { type Card, type Which, valueAt } from "./cards.ts";

const LISTED = 8;
const COUNT = new Intl.NumberFormat("en-US");

export function Inspector(props: {
  readonly session: Session;
  readonly state: SessionState;
  readonly which: Which;
  readonly onWhich: (which: Which) => void;
  readonly cards: readonly Card[] | null;
  readonly layer: number;
  readonly selection: Selection | null;
  readonly onSelect: (chosen: Selection) => void;
}) {
  const { session, state, which, onWhich, cards, layer, selection, onSelect } =
    props;
  const { run, pins } = state;
  const card = cards?.[layer];
  return (
    <aside className="column inspector" aria-label="Inspector">
      <fieldset className="segmented">
        <legend>Neurons from</legend>
        {(["original", "edited"] as const).map((w) => (
          <label key={w} className={which === w ? "on" : undefined}>
            <input
              type="radio"
              name="neurons-from"
              checked={which === w}
              disabled={w === "edited" && (run === null || run.edited === null)}
              onChange={() => onWhich(w)}
            />
            {w === "original" ? "Original reply" : "Edited reply"}
          </label>
        ))}
      </fieldset>
      {card !== undefined && (
        <Strongest
          card={card}
          layer={layer}
          pins={pins}
          selection={selection}
          onSelect={onSelect}
        />
      )}
      {selection === null ? (
        <p className="hint">
          Click a neuron on the deck, drag across several, or go to one by
          number.
        </p>
      ) : selection.neurons.length === 1 ? (
        <NeuronCard
          key={`${selection.layer}:${selection.neurons[0]}`}
          session={session}
          state={state}
          neuron={{ layer: selection.layer, neuron: selection.neurons[0] ?? 0 }}
        />
      ) : (
        <SelectionCard
          key={`${selection.layer}:${selection.neurons.join(",")}`}
          session={session}
          pins={pins}
          selection={selection}
        />
      )}
      <GoTo onSelect={(ref) => onSelect(one(ref))} />
    </aside>
  );
}

function Strongest(props: {
  readonly card: Card;
  readonly layer: number;
  readonly pins: readonly Edit[];
  readonly selection: Selection | null;
  readonly onSelect: (chosen: Selection) => void;
}) {
  const { card, layer, pins, selection, onSelect } = props;
  const ranked = strongest(card.values, LISTED);
  const top = Math.abs(card.values[ranked[0] ?? 0] ?? 0);
  return (
    <section aria-label={`Strongest in layer ${layer}`}>
      <h3>Strongest in layer {layer}</h3>
      <ul className="ranked">
        {ranked.map((neuron) => {
          const v = card.values[neuron] ?? Number.NaN;
          const share = Number.isFinite(v) && top > 0 ? Math.abs(v) / top : 1;
          const isPinned = pins.some(
            (p) => p.layer === layer && p.neuron === neuron,
          );
          const isSelected =
            selection?.layer === layer && selection.neurons.includes(neuron);
          return (
            <li key={neuron}>
              <button
                type="button"
                aria-pressed={isSelected}
                onClick={() => onSelect({ layer, neurons: [neuron] })}
              >
                <span>
                  #{neuron}
                  {isPinned && <span className="pin-dot"> ●</span>}
                </span>
                <span className="bar" aria-hidden>
                  <i style={{ width: `${Math.round(share * 100)}%` }} />
                </span>
                <span>{signed(v)}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function NeuronCard(props: {
  readonly session: Session;
  readonly state: SessionState;
  readonly neuron: NeuronRef;
}) {
  const { session, state, neuron } = props;
  const { run, pins } = state;
  const pin = pins.find(
    (p) => p.layer === neuron.layer && p.neuron === neuron.neuron,
  );
  const [text, setText] = useState(
    pin !== undefined && pin.kind !== "zero" ? String(pin.value) : "1",
  );
  const problemId = useId();
  const value = parseValue(text, "Pin value");
  const pinAs = (kind: Edit["kind"]) => {
    if (kind === "zero") {
      session.pin({ ...neuron, kind, value: 0 });
    } else if (value.ok) {
      session.pin({ ...neuron, kind, value: value.value });
    }
  };
  return (
    <section className="neuron-card" aria-label="Neuron">
      <h3>
        Neuron #{neuron.neuron}{" "}
        <span className="muted">· layer {neuron.layer}</span>
      </h3>
      <dl>
        <dt>Original</dt>
        <dd>
          <Value
            label="Neuron original value"
            valued={valueAt(run, run?.original ?? null, neuron)}
          />
        </dd>
        <dt className="edited">Edited</dt>
        <dd className="edited">
          <Value
            label="Neuron edited value"
            valued={valueAt(run, run?.edited ?? null, neuron)}
          />
        </dd>
      </dl>
      <p>
        <output aria-label="Neuron pin" className={pin ? "edited" : "muted"}>
          {pinText(pin)}
        </output>
      </p>
      <div className="pin-controls">
        <input
          type="text"
          inputMode="decimal"
          size={7}
          aria-label="Pin value"
          aria-invalid={!value.ok}
          aria-describedby={value.ok ? undefined : problemId}
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button type="button" onClick={() => pinAs("set")}>
          Set
        </button>
        <button type="button" onClick={() => pinAs("scale")}>
          Scale ×
        </button>
        <button type="button" onClick={() => pinAs("zero")}>
          Zero
        </button>
        {pin !== undefined && (
          <button
            type="button"
            onClick={() => session.unpin(neuron.layer, neuron.neuron)}
          >
            Unpin
          </button>
        )}
      </div>
      {!value.ok && (
        <p id={problemId} role="note">
          {value.problem.reason}
        </p>
      )}
    </section>
  );
}

function SelectionCard(props: {
  readonly session: Session;
  readonly pins: readonly Edit[];
  readonly selection: Selection;
}) {
  const { session, pins, selection } = props;
  const { layer, neurons } = selection;
  const chosen = new Set(neurons);
  const pinned = pins.filter(
    (p) => p.layer === layer && chosen.has(p.neuron),
  ).length;
  const [text, setText] = useState("1");
  const problemId = useId();
  const value = parseValue(text, "Pin value");
  const pinAs = (kind: Edit["kind"]) => {
    const v = kind === "zero" ? 0 : value.ok ? value.value : null;
    if (v === null) return;
    session.pinAll(
      neurons.map((neuron) => ({ layer, neuron, kind, value: v })),
    );
  };
  return (
    <section className="neuron-card" aria-label="Selection">
      <h3>
        {COUNT.format(neurons.length)} neurons{" "}
        <span className="muted">· layer {layer}</span>
      </h3>
      <p>
        <output
          aria-label="Selection pins"
          className={pinned > 0 ? "edited" : "muted"}
        >
          {pinned === 0
            ? "None pinned"
            : `${COUNT.format(pinned)} of ${COUNT.format(neurons.length)} pinned`}
        </output>
      </p>
      <div className="pin-controls">
        <input
          type="text"
          inputMode="decimal"
          size={7}
          aria-label="Pin value"
          aria-invalid={!value.ok}
          aria-describedby={value.ok ? undefined : problemId}
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button type="button" onClick={() => pinAs("set")}>
          Set
        </button>
        <button type="button" onClick={() => pinAs("scale")}>
          Scale ×
        </button>
        <button type="button" onClick={() => pinAs("zero")}>
          Zero
        </button>
        {pinned > 0 && (
          <button
            type="button"
            onClick={() => session.unpinAll(layer, neurons)}
          >
            Unpin
          </button>
        )}
      </div>
      {!value.ok && (
        <p id={problemId} role="note">
          {value.problem.reason}
        </p>
      )}
    </section>
  );
}

function Value(props: {
  readonly label: string;
  readonly valued: ReturnType<typeof valueAt>;
}) {
  const { label, valued } = props;
  return (
    <output aria-label={label}>
      {valued.kind === "value" ? (
        <data value={exact(valued.value)}>{signed(valued.value, 3)}</data>
      ) : (
        valued.text
      )}
    </output>
  );
}

function GoTo({ onSelect }: { readonly onSelect: (n: NeuronRef) => void }) {
  const [layerText, setLayerText] = useState("");
  const [neuronText, setNeuronText] = useState("");
  const problemId = useId();
  const layer = parseWord(layerText, "Go to layer");
  const neuron = parseWord(neuronText, "Go to neuron");
  const touched = layerText !== "" || neuronText !== "";
  const problem =
    !touched || (layer.ok && neuron.ok)
      ? null
      : !layer.ok
        ? layer.problem
        : !neuron.ok
          ? neuron.problem
          : null;
  return (
    <form
      className="go-to"
      onSubmit={(event) => {
        event.preventDefault();
        if (layer.ok && neuron.ok) {
          onSelect({ layer: layer.value, neuron: neuron.value });
        }
      }}
    >
      <h3>Go to a neuron</h3>
      <label>
        Layer{" "}
        <input
          type="text"
          inputMode="numeric"
          aria-label="Go to layer"
          value={layerText}
          aria-describedby={problem ? problemId : undefined}
          onChange={(event) => setLayerText(event.target.value)}
        />
      </label>
      <label>
        Neuron{" "}
        <input
          type="text"
          inputMode="numeric"
          aria-label="Go to neuron"
          value={neuronText}
          aria-describedby={problem ? problemId : undefined}
          onChange={(event) => setNeuronText(event.target.value)}
        />
      </label>
      <button type="submit">Go</button>
      {problem !== null && (
        <p id={problemId} role="note">
          {problem.field}: {problem.reason}
        </p>
      )}
    </form>
  );
}

function pinText(pin: Edit | undefined): string {
  if (pin === undefined) return "Not pinned";
  switch (pin.kind) {
    case "set":
      return `Pinned: set to ${exact(pin.value)}`;
    case "scale":
      return `Pinned: scaled by ${exact(pin.value)}`;
    case "zero":
      return "Pinned: zeroed";
  }
}

// A value exactly as the model used it: an f32 widened to a double, as
// String writes it, with a negative zero as "-0", which String drops.
function exact(value: number): string {
  return Object.is(value, -0) ? "-0" : String(value);
}
