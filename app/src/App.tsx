// Axis H (frame): the app's frame. A header with the GPU, the model and
// Load model; under it the conversation, full width; over it, when opened,
// the neuron workbench, which Run closes.
//
// Contract
//   - App({session}): renders, on night glass (theme/), for that session:
//       header   "Frizzle"; status "WebGPU" and status "Model" (their words
//                below), the model's with a progress bar while it loads.
//                Load model itself is on the prompt (views/prompt/), where
//                the person starts.
//       page     the conversation (views/prompt/): the prompt, then the
//                original and edited replies side by side.
//       layer    the workbench (views/deck/), a dialog over the whole
//                window, opened from the prompt's "Open neurons" or a
//                reply's "Neurons" button, showing that reply's neurons.
//                While it is open the page behind it is inert. Close or Esc
//                closes it; its Run runs and closes it, so the replies show
//                as they stream.
//   - status "WebGPU": "Waiting for the GPU" | "Ready: {adapter}" |
//     "Failed: {error}".
//   - status "Model": "Not loaded" | "Starting" | "Downloading: {done} of
//     {total} MB" | "Loading: {done} of {total} MB", each with a progress
//     bar | "Ready: downloaded {n} MB, {c} tokens of context", n in a <data
//     value> holding the exact bytes | "Ready: kept from an earlier visit,
//     {c} tokens of context" | "Failed at {stage}: {message}". Megabytes
//     are decimal and whole (shape/numbers.ts).
//   - Held here, so they last while the workbench is closed and reopened:
//     the message and seed as typed, the reply whose neurons are shown,
//     the front layer, and the selection: one or more neurons of one layer
//     (Selection), which is brought to the front when selected. The pins
//     are the session's.
//   - The session, and the worker behind it, are made once by main.tsx,
//     not here: React may mount a component twice (StrictMode), and a page
//     must not start two workers.
//   - Nothing loads until asked: the model is 722 MB, so the download
//     begins at the person's Load model, never when the page opens.
//
// Design
//   - Reading replies and working on neurons each get the whole window:
//     the two are done in turns (read, go in, pin, run, read again), and
//     side by side each was too narrow. The workbench is drawn only while
//     open, so the GPU does not draw the deck behind the replies.

import { useState, useSyncExternalStore } from "react";

import type { Edit } from "./engine/edits.ts";
import type { DeviceState, ModelState, Session } from "./session/session.ts";
import { megabytes } from "./shape/numbers.ts";
import { Workbench } from "./views/deck/Workbench.tsx";
import type { Which } from "./views/deck/cards.ts";
import { parseSeed } from "./views/prompt/fields.ts";
import { PromptView } from "./views/prompt/PromptView.tsx";

export type NeuronRef = Pick<Edit, "layer" | "neuron">;

// One or more neurons of one layer: `neurons` ascending, none twice, at
// least one.
export interface Selection {
  readonly layer: number;
  readonly neurons: readonly number[];
}

export const one = ({ layer, neuron }: NeuronRef): Selection => ({
  layer,
  neurons: [neuron],
});

const COUNT = new Intl.NumberFormat("en-US");

export function App({ session }: { readonly session: Session }) {
  const state = useSyncExternalStore(session.subscribe, session.snapshot);
  const [message, setMessage] = useState("");
  const [seedText, setSeedText] = useState("1");
  const [open, setOpen] = useState(false);
  const [which, setWhich] = useState<Which>("original");
  const [front, setFront] = useState(0);
  const [selection, setSelection] = useState<Selection | null>(null);
  const { device, model } = state;
  const canLoad =
    device.kind === "ready" &&
    (model.kind === "idle" || model.kind === "failed");
  const seed = parseSeed(seedText);
  const canRun = model.kind === "ready" && seed.ok;
  const run = () => {
    if (!canRun || !seed.ok) return;
    session.run(message, seed.value);
    setOpen(false);
  };
  const select = (chosen: Selection) => {
    setSelection(chosen);
    setFront(chosen.layer);
  };
  // Open the workbench, on a reply's neurons and a selection when given.
  const openOn = (reply?: Which, chosen?: Selection) => {
    if (reply !== undefined) setWhich(reply);
    if (chosen !== undefined) select(chosen);
    setOpen(true);
  };

  return (
    <div className="frame">
      <div className="page" inert={open}>
        <header className="masthead">
          <b className="wordmark">Frizzle</b>
          <span className="pill">
            WebGPU <output aria-label="WebGPU">{deviceText(device)}</output>
          </span>
          <span className="pill">
            Model <output aria-label="Model">{modelText(model)}</output>
            {(model.kind === "downloading" || model.kind === "loading") && (
              <progress
                aria-hidden
                value={model.done}
                max={Math.max(1, model.total)}
              />
            )}
          </span>
        </header>
        <PromptView
          session={session}
          state={state}
          message={message}
          onMessage={setMessage}
          seedText={seedText}
          onSeedText={setSeedText}
          canLoad={canLoad}
          onLoad={() => session.load()}
          canRun={canRun}
          onRun={run}
          onOpen={openOn}
        />
      </div>
      {open && (
        <Workbench
          session={session}
          state={state}
          message={message}
          canRun={canRun}
          onRun={run}
          onClose={() => setOpen(false)}
          which={which}
          onWhich={setWhich}
          front={front}
          onFront={setFront}
          selection={selection}
          onSelect={select}
        />
      )}
    </div>
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

function modelText(model: ModelState) {
  switch (model.kind) {
    case "idle":
      return "Not loaded";
    case "starting":
      return "Starting";
    case "downloading":
    case "loading":
      return `${model.kind === "downloading" ? "Downloading" : "Loading"}: ${megabytes(model.done).replace(" MB", "")} of ${megabytes(model.total)}`;
    case "ready":
      return model.downloaded === null ? (
        `Ready: kept from an earlier visit, ${COUNT.format(model.contextTokens)} tokens of context`
      ) : (
        <>
          Ready: downloaded{" "}
          <data value={model.downloaded}>{megabytes(model.downloaded)}</data>,{" "}
          {COUNT.format(model.contextTokens)} tokens of context
        </>
      );
    case "failed":
      return `Failed at ${model.failure.stage}: ${model.failure.message}`;
  }
}
