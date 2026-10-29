// Axis H (view): the plain page. A message, a seed and pins by number;
// the original reply and the edited reply side by side, from one seed.
//
// Contract
//   - Shows the session's state (session/session.ts) and nothing derived
//     from it: what the GPU, the worker and the model report, in their
//     words.
//   - The page is read and driven by role and accessible name; these
//     names are its interface, which the browser scenario uses
//     (tests/integration/browser/), so they change only with it:
//       status "WebGPU"          "Waiting for the GPU" | "Ready: {adapter}"
//                                | "Failed: {error}"
//       button "Load model"      enabled once WebGPU is ready, while the
//                                model is not loading or loaded
//       status "Model"           "Not loaded" | "Downloading: {done} of
//                                {total} bytes" | "Loading: {done} of
//                                {total} bytes" | "Ready: downloaded {n}
//                                bytes, {c} tokens of context" (n in a
//                                <data value> holding the integer) |
//                                "Ready: kept from an earlier visit, {c}
//                                tokens of context" | "Failed at {stage}:
//                                {message}"
//       textbox "Message", textbox "Seed" (starting at "1")
//       button "Add pin"; for pin n, counting from 1: textbox "Pin n
//         layer", textbox "Pin n neuron", combobox "Pin n kind" (options
//         "set", "scale", "zero"), textbox "Pin n value", button
//         "Remove pin n" (PinsEditor.tsx)
//       button "Run"             enabled once the model is ready and every
//                                field parses (fields.ts)
//       status "Original reply", status "Edited reply"   the reply's text
//       status "Original reply ending", status "Edited reply ending"
//                                "Running" | "Done: {end}, {tokens} tokens"
//                                | "Failed at {stage}: {message}"; the
//                                edited also "Waiting for the original" |
//                                "No pins"
//     A field that does not parse shows its problem beside it, named by
//     aria-describedby, and Run stays disabled.
//   - Run sends the message, the parsed pins and the seed as typed when
//     pressed; editing the fields afterwards changes nothing shown until
//     the next Run. Pressing Run during a run starts a new one, which
//     supersedes it (session.ts).
//
// Design
//   - Numbers are written with Intl.NumberFormat("en-US"), named, never the
//     browser's locale, so the page reads the same everywhere.
//   - No theme: the night-glass theme arrives with the workbench (slice 4).
//     Two columns side by side on a wide screen, one above the other on a
//     narrow one.
//   - The fields' text is the view's own state; the session holds only
//     what has been sent.

import { type ReactNode, useId, useState, useSyncExternalStore } from "react";

import type {
  DeviceState,
  ModelState,
  ReplyState,
  Run,
  Session,
} from "../../session/session.ts";
import { PinsEditor } from "./PinsEditor.tsx";
import {
  type PinFields,
  type Problem,
  parsePins,
  parseSeed,
} from "./fields.ts";

const NUMBER = new Intl.NumberFormat("en-US");
const n = (value: number): string => NUMBER.format(value);

export function PromptView({ session }: { readonly session: Session }) {
  const state = useSyncExternalStore(session.subscribe, session.snapshot);
  const [message, setMessage] = useState("");
  const [seedText, setSeedText] = useState("1");
  const [rows, setRows] = useState<readonly PinFields[]>([]);
  const seedProblemId = useId();

  const seed = parseSeed(seedText);
  const pins = parsePins(rows);
  const seedProblem = seed.ok ? undefined : seed.problems[0];
  const pinProblems: readonly Problem[] = pins.ok ? [] : pins.problems;
  const { device, model, run } = state;
  const canLoad =
    device.kind === "ready" &&
    (model.kind === "idle" || model.kind === "failed");
  const canRun = model.kind === "ready" && seed.ok && pins.ok;

  return (
    <main>
      <h1>Frizzle</h1>
      <p>
        WebGPU: <output aria-label="WebGPU">{deviceText(device)}</output>
      </p>
      <p>
        <button
          type="button"
          disabled={!canLoad}
          onClick={() => session.load()}
        >
          Load model
        </button>{" "}
        <output aria-label="Model">{modelText(model)}</output>
      </p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (seed.ok && pins.ok) session.run(message, pins.value, seed.value);
        }}
      >
        <p>
          <label>
            Message
            <br />
            <textarea
              rows={4}
              cols={60}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
            />
          </label>
        </p>
        <p>
          <label>
            Seed{" "}
            <input
              type="text"
              inputMode="numeric"
              size={12}
              value={seedText}
              aria-invalid={seedProblem !== undefined}
              aria-describedby={
                seedProblem === undefined ? undefined : seedProblemId
              }
              onChange={(event) => setSeedText(event.target.value)}
            />
          </label>
          {seedProblem !== undefined && (
            <span id={seedProblemId} role="note">
              {" "}
              {seedProblem.reason}
            </span>
          )}
        </p>
        <PinsEditor rows={rows} problems={pinProblems} onChange={setRows} />
        <p>
          <button type="submit" disabled={!canRun}>
            Run
          </button>
        </p>
      </form>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(20rem, 1fr))",
          gap: "1rem",
        }}
      >
        <Reply label="Original reply" run={run} reply={run?.original ?? null} />
        <Reply label="Edited reply" run={run} reply={run?.edited ?? null} />
      </div>
    </main>
  );
}

function Reply(props: {
  readonly label: string;
  readonly run: Run | null;
  readonly reply: ReplyState | null;
}) {
  const { label, run, reply } = props;
  const text = reply === null || reply.kind === "waiting" ? "" : reply.text;
  return (
    <section aria-label={label}>
      <h2>{label}</h2>
      <p>
        <output aria-label={`${label} ending`}>{endingText(run, reply)}</output>
      </p>
      <output aria-label={label} style={{ whiteSpace: "pre-wrap" }}>
        {text}
      </output>
    </section>
  );
}

function deviceText(device: DeviceState): string {
  switch (device.kind) {
    case "waiting":
      return "Waiting for the GPU";
    case "ready":
      return `Ready: ${device.adapter}`;
    case "failed":
      return `Failed: ${device.error}`;
  }
}

function modelText(model: ModelState): ReactNode {
  switch (model.kind) {
    case "idle":
      return "Not loaded";
    case "starting":
      return "Starting";
    case "downloading":
      return `Downloading: ${n(model.done)} of ${n(model.total)} bytes`;
    case "loading":
      return `Loading: ${n(model.done)} of ${n(model.total)} bytes`;
    case "ready":
      return model.downloaded === null ? (
        `Ready: kept from an earlier visit, ${n(model.contextTokens)} tokens of context`
      ) : (
        <>
          Ready: downloaded{" "}
          <data value={model.downloaded}>{n(model.downloaded)}</data> bytes,{" "}
          {n(model.contextTokens)} tokens of context
        </>
      );
    case "failed":
      return `Failed at ${model.failure.stage}: ${model.failure.message}`;
  }
}

// The run's ending for one reply: nothing before the first run.
function endingText(run: Run | null, reply: ReplyState | null): string {
  if (run === null) return "";
  if (reply === null) return "No pins";
  switch (reply.kind) {
    case "waiting":
      return "Waiting for the original";
    case "running":
      return "Running";
    case "done":
      return `Done: ${reply.answer.end}, ${n(reply.answer.tokens)} tokens`;
    case "failed":
      return `Failed at ${reply.failure.stage}: ${reply.failure.message}`;
  }
}
