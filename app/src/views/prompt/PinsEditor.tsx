// Axis H (view): the pins, by number: one row per pin, a neuron of one
// layer and what to do to it.
//
// Contract
//   - Shows `rows` as typed (fields.ts's PinFields), with each field's
//     problem beside it (`problems`, matched by field label), and reports
//     every change through onChange(rows), the whole list each time.
//   - Pin n, counting from 1 in the order shown: textbox "Pin n layer",
//     textbox "Pin n neuron", combobox "Pin n kind" with options "set",
//     "scale" and "zero" (each option's value is its label, the EditKind),
//     textbox "Pin n value" (disabled for "zero"), button "Remove pin n".
//     Button "Add pin" appends a row: empty layer and neuron, kind "set",
//     empty value.
//   - Removing a pin renumbers those after it; nothing is kept per number.
//
// Design
//   - Text fields, not number inputs: the browser's number input coerces
//     and hides what was typed, and fields.ts reads the text itself.
//   - No limit on the number of rows: how many edits a run may carry is
//     the engine's to say, by name.

import { useId } from "react";

import type { EditKind } from "../../engine/edits.ts";
import type { PinFields, Problem } from "./fields.ts";

const KINDS: readonly EditKind[] = ["set", "scale", "zero"];

const EMPTY: PinFields = { layer: "", neuron: "", kind: "set", value: "" };

export function PinsEditor(props: {
  readonly rows: readonly PinFields[];
  readonly problems: readonly Problem[];
  readonly onChange: (rows: readonly PinFields[]) => void;
}) {
  const { rows, problems, onChange } = props;
  const id = useId();
  const change = (index: number, patch: Partial<PinFields>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));

  // A field, labelled, with its problem beside it when it has one.
  const field = (
    index: number,
    label: string,
    key: "layer" | "neuron" | "value",
    disabled = false,
  ) => {
    const problem = problems.find((p) => p.field === label);
    const problemId = `${id}-${index}-${key}`;
    return (
      <span>
        <input
          type="text"
          inputMode={key === "value" ? "decimal" : "numeric"}
          aria-label={label}
          aria-describedby={problem === undefined ? undefined : problemId}
          aria-invalid={problem !== undefined}
          size={8}
          value={rows[index]?.[key] ?? ""}
          disabled={disabled}
          onChange={(event) => change(index, { [key]: event.target.value })}
        />
        {problem !== undefined && (
          <span id={problemId} role="note">
            {" "}
            {problem.reason}
          </span>
        )}
      </span>
    );
  };

  return (
    <fieldset>
      <legend>Pins</legend>
      {rows.map((row, index) => {
        const n = index + 1;
        return (
          <div key={index}>
            Layer {field(index, `Pin ${n} layer`, "layer")} neuron{" "}
            {field(index, `Pin ${n} neuron`, "neuron")}{" "}
            <select
              aria-label={`Pin ${n} kind`}
              value={row.kind}
              onChange={(event) =>
                change(index, { kind: event.target.value as EditKind })
              }
            >
              {KINDS.map((kind) => (
                <option key={kind} value={kind}>
                  {kind}
                </option>
              ))}
            </select>{" "}
            {field(index, `Pin ${n} value`, "value", row.kind === "zero")}{" "}
            <button
              type="button"
              onClick={() => onChange(rows.filter((_, i) => i !== index))}
            >
              Remove pin {n}
            </button>
          </div>
        );
      })}
      <button type="button" onClick={() => onChange([...rows, EMPTY])}>
        Add pin
      </button>
    </fieldset>
  );
}
