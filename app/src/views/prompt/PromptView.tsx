// Axis H (view): the conversation, the full width of the page. The prompt,
// with Run on it and the pins it will send; under it the original reply
// and the edited reply side by side, from one seed.
//
// Contract
//   - Shows the session's state (session/session.ts) and nothing derived
//     from it beyond rounding for reading. The message and seed are the
//     frame's (App.tsx), so the workbench's Run sends them too.
//   - The page is read and driven by role and accessible name; these names
//     are its interface, which the browser scenario uses
//     (tests/integration/browser/), so they change only with it:
//       textbox "Message"; textbox "Seed" (starting at "1"), its problem
//         beside it when it does not parse (fields.ts), named by
//         aria-describedby
//       the pins held, as chips (PinChips.tsx); a chip's "Show …" opens
//         the workbench on its neurons
//       button "Open neurons": opens the workbench (onOpen)
//       until the model is ready, button "Load model" in Run's place,
//         enabled when canLoad (onLoad), and beside it the model's size
//         (config/model.json) before it starts, then the load's progress in
//         megabytes with a bar, or its failure; the person's first step
//         sits where they type
//       once the model is ready, button "Run": enabled once the seed
//         parses; Ctrl+Enter or ⌘+Enter in the message runs too
//       status "Original reply", status "Edited reply": the reply's text,
//         each in a pane of at most about half the window's height that
//         scrolls on its own, focusable so the keys scroll it too
//       status "Original reply ending", status "Edited reply ending":
//         "Running" | "Done: {end}, {tokens} tokens" | "Failed at {stage}:
//         {message}"; the edited also "Waiting for the original" | "No
//         pins"
//       button "Neurons in the original reply", button "Neurons in the
//         edited reply" (once the run has an edited reply): open the
//         workbench on that reply's neurons
//   - The two panes scroll together: scrolling one, by any means, moves
//     the other to the same distance from its top (as far as it goes), so
//     the replies stay lined up line for line where they agree. While a
//     reply streams, its pane follows the end of the text if it was at the
//     end; scrolled up, it stays where it was put. A new run starts both at
//     the top.
//   - Run sends the message, the pins held and the seed as they are when
//     pressed; changing them afterwards changes nothing shown until the
//     next Run. Pressing Run during a run starts a new one, which
//     supersedes it (session.ts).
//
// Design
//   - After Dolittle's composer: the prompt box carries its own action
//     row, so Run is beside what it sends, and the pins sit on the box as
//     chips because they go out with the run. Pins are made in the
//     workbench, never typed here.
//   - The replies side by side, the comparison the demo is for; one above
//     the other where the window is narrow. Capped, so a long reply (one
//     can run to the end of the context) scrolls inside its pane rather
//     than growing the page.
//   - Same distance, not the same fraction: the replies share their start
//     from one seed until the edit changes them, so equal offsets keep the
//     shared lines side by side; a fraction would drift as the lengths
//     differ.

import { useId, useLayoutEffect, useRef } from "react";

import type { Selection } from "../../App.tsx";
import type {
  ReplyState,
  Run,
  Session,
  SessionState,
} from "../../session/session.ts";
import type { Which } from "../deck/cards.ts";
import pin from "../../../../config/model.json" with { type: "json" };
import { megabytes } from "../../shape/numbers.ts";
import { PinChips } from "./PinChips.tsx";
import { parseSeed } from "./fields.ts";

const COUNT = new Intl.NumberFormat("en-US");

export function PromptView(props: {
  readonly session: Session;
  readonly state: SessionState;
  readonly message: string;
  readonly onMessage: (message: string) => void;
  readonly seedText: string;
  readonly onSeedText: (text: string) => void;
  readonly canLoad: boolean;
  readonly onLoad: () => void;
  readonly canRun: boolean;
  readonly onRun: () => void;
  readonly onOpen: (reply?: Which, chosen?: Selection) => void;
}) {
  const { session, state, message, onMessage, seedText, onSeedText } = props;
  const { canLoad, onLoad, canRun, onRun, onOpen } = props;
  const { model } = state;
  const seedProblemId = useId();
  const seed = parseSeed(seedText);
  const { pins, run } = state;

  // The two reply panes, scrolled together (the header's Contract). A
  // scroll this code sets on one pane fires that pane's scroll event too;
  // `echo` marks it, so it is not sent back.
  const panes = useRef<Record<Which, HTMLElement | null>>({
    original: null,
    edited: null,
  });
  const echo = useRef<HTMLElement | null>(null);
  const follow = (from: Which) => {
    const self = panes.current[from];
    const other = panes.current[from === "original" ? "edited" : "original"];
    if (self === null) return;
    if (echo.current === self) {
      echo.current = null;
      return;
    }
    if (other === null || other.scrollTop === self.scrollTop) return;
    const before = other.scrollTop;
    echo.current = other;
    other.scrollTop = self.scrollTop;
    // A pane that cannot move (it is shorter) fires no event to clear it.
    if (other.scrollTop === before) echo.current = null;
  };

  return (
    <main className="conversation" aria-label="Conversation">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          onRun();
        }}
      >
        <textarea
          aria-label="Message"
          rows={3}
          placeholder="Ask the model something"
          value={message}
          onChange={(event) => onMessage(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              onRun();
            }
          }}
        />
        <PinChips
          session={session}
          pins={pins}
          onPick={(chosen) => onOpen(undefined, chosen)}
        />
        <div className="actions">
          <label className="seed">
            Seed{" "}
            <input
              type="text"
              inputMode="numeric"
              size={6}
              value={seedText}
              aria-invalid={!seed.ok}
              aria-describedby={seed.ok ? undefined : seedProblemId}
              onChange={(event) => onSeedText(event.target.value)}
            />
          </label>
          <button type="button" onClick={() => onOpen()}>
            Open neurons
          </button>
          <span className="hint">
            {pins.length === 0
              ? "No pins yet"
              : `${pins.length} ${pins.length === 1 ? "pin" : "pins"} held`}
          </span>
          {model.kind === "ready" ? (
            <button type="submit" className="primary" disabled={!canRun}>
              Run
            </button>
          ) : (
            <>
              <span className="load-progress">{loadText(model)}</span>
              <button
                type="button"
                className="primary"
                disabled={!canLoad}
                onClick={onLoad}
              >
                Load model
              </button>
            </>
          )}
        </div>
        {!seed.ok && (
          <p id={seedProblemId} role="note">
            {seed.problem.field}: {seed.problem.reason}
          </p>
        )}
      </form>
      <div className="replies">
        <Reply
          pane={(el) => {
            panes.current.original = el;
          }}
          onScroll={() => follow("original")}
          label="Original reply"
          title="Original"
          which="original"
          run={run}
          reply={run?.original ?? null}
          onOpen={onOpen}
        />
        <Reply
          label="Edited reply"
          title={
            run !== null && run.edits.length > 0
              ? `Edited · ${run.edits.length} ${run.edits.length === 1 ? "pin" : "pins"}`
              : "Edited"
          }
          which="edited"
          pane={(el) => {
            panes.current.edited = el;
          }}
          onScroll={() => follow("edited")}
          run={run}
          reply={run?.edited ?? null}
          onOpen={onOpen}
        />
      </div>
    </main>
  );
}

function Reply(props: {
  readonly label: string;
  readonly title: string;
  readonly which: Which;
  readonly pane: (element: HTMLElement | null) => void;
  readonly onScroll: () => void;
  readonly run: Run | null;
  readonly reply: ReplyState | null;
  readonly onOpen: (reply?: Which, chosen?: Selection) => void;
}) {
  const { label, title, which, pane, onScroll, run, reply, onOpen } = props;
  const text = reply === null || reply.kind === "waiting" ? "" : reply.text;
  const element = useRef<HTMLElement | null>(null);
  // Whether the pane was at the end of its text before this render.
  const atEnd = useRef(true);
  useLayoutEffect(() => {
    const el = element.current;
    if (el === null) return;
    if (text === "") {
      atEnd.current = true;
      el.scrollTop = 0;
    } else if (atEnd.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [text]);
  return (
    <section
      aria-label={label}
      className={which === "edited" ? "reply reply-edited" : "reply"}
    >
      <h3>
        <span>{title}</span>
        <output aria-label={`${label} ending`}>{endingText(run, reply)}</output>
      </h3>
      <output
        aria-label={label}
        className="reply-text"
        tabIndex={0}
        ref={(el) => {
          element.current = el;
          pane(el);
        }}
        onScroll={(event) => {
          const el = event.currentTarget;
          atEnd.current = el.scrollHeight - el.scrollTop - el.clientHeight < 4;
          onScroll();
        }}
      >
        {text}
      </output>
      {reply !== null && (
        <button
          type="button"
          className="neurons"
          aria-label={`Neurons in the ${which} reply`}
          onClick={() => onOpen(which)}
        >
          Neurons ↗
        </button>
      )}
    </section>
  );
}

// Beside Load model: what loading costs, how far it is, or why it failed.
function loadText(model: SessionState["model"]) {
  switch (model.kind) {
    case "idle":
      return `${megabytes(pin.sizeBytes)}, kept after the first visit`;
    case "starting":
      return "Starting";
    case "downloading":
    case "loading":
      return (
        <>
          {model.kind === "downloading" ? "Downloading" : "Loading"}{" "}
          {megabytes(model.done).replace(" MB", "")} of {megabytes(model.total)}
          <progress
            aria-hidden
            value={model.done}
            max={Math.max(1, model.total)}
          />
        </>
      );
    case "failed":
      return `Failed at ${model.failure.stage}: ${model.failure.message}`;
    case "ready":
      return "";
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
      return `Done: ${reply.answer.end}, ${COUNT.format(reply.answer.tokens)} tokens`;
    case "failed":
      return `Failed at ${reply.failure.stage}: ${reply.failure.message}`;
  }
}
