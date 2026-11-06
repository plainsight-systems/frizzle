// Axis H (view): the conversation, the full width of the page. The prompt,
// with Run on it and the pins it will send; under it the reply to steer
// from and the steered reply side by side, from one seed: pick a word in
// the reply, look at its neurons, pin, Run, and Accept what reads better.
//
// Contract
//   - Shows the session's state (session/session.ts) and nothing derived
//     from it beyond rounding for reading and the reply's words
//     (session/branch.ts, wordsOf). The message and seed are the frame's
//     (App.tsx), so the workbench's Run sends them too.
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
//       status "Reply", status "Steered reply": the text, each in a pane
//         of at most about half the window's height that scrolls on its
//         own, focusable so the keys scroll it too; their text content is
//         exactly the reply's text, and for the steered reply the reply's
//         kept text before its own
//       status "Reply ending", status "Steered reply ending": "Running:
//         {tokens} tokens" (so far, so a reply of white space shows it
//         runs) |
//         "Done: {end}, {tokens} tokens" (the reply's tokens; the steered
//         reply's own, past its branch point) | "Failed at {stage}:
//         {message}"; the steered also "Waiting" | "Not steered"
//       in the reply, once it is done: each word that holds more than white
//         space a button named by its text (its leading white space outside
//         it), pressed when the branch point is its end; pressing one sets
//         the branch point after it (session.branchAt). Button "From the
//         start", pressed at point 0, sets it before the first word. While
//         a reply or a branch runs they are disabled. With the reply's pane
//         focused, ← and → move the point to the word before or after.
//       button "Stop": beside Run while the reply or the steered reply
//         runs or waits; ends the run (session.stop), so a reply that
//         would run to the end of the context is never a dead end
//       button "Accept": enabled once the steered reply is done, stopped
//         included; makes it the reply (session.accept)
//       button "Neurons in the reply", button "Neurons in the steered
//         reply" (once the run has a steered reply): open the workbench on
//         that side's neurons at the branch point
//   - Once a word is chosen, the branch point shows as a mark in the reply
//     where it falls (after a word, or inside one when an accepted steered
//     reply went on without a space), the text past it faded: what a
//     branch from it replaces. At the start nothing is faded, so the reply
//     reads whole. In the steered reply, the kept text is faded and its own
//     is not; its title names its pins and where it branched, the word as
//     written (not in the heading's capitals).
//   - The two panes scroll together: scrolling one, by any means, moves
//     the other to the same distance from its top (as far as it goes), so
//     the replies stay lined up line for line where they agree. While a
//     reply streams, its pane follows the end of the text if it was at the
//     end; scrolled up, it stays where it was put. A new run starts both at
//     the top.
//   - Run sends the message, the pins held and the seed as they are when
//     pressed; changing them afterwards changes nothing shown until the
//     next Run. With the reply done and the same message and seed, Run
//     steers from the branch point; otherwise it starts a new reply
//     (session.ts). Pressing Run during a run starts a new one, which
//     supersedes it.
//
// Design
//   - A composer: the prompt box carries its own action row, so Run is
//     beside what it sends, and the pins sit on the box as chips because
//     they go out with the run. Pins are made in the workbench, never
//     typed here.
//   - Words, not tokens, as the points to steer from: a person steers
//     meaning, and a word is whole tokens (branch.ts). The words are
//     buttons only once the reply is done: while it streams it is text, so
//     a long reply is not redrawn as hundreds of buttons on every token.
//     Only the pressed word takes the tab; the arrows move the point, so
//     the keys reach every word without a tab stop for each.
//   - The replies side by side, the comparison the demo is for; one above
//     the other where the window is narrow. Capped, so a long reply (one
//     can run to the end of the context) scrolls inside its pane rather
//     than growing the page.
//   - Same distance, not the same fraction: the replies share their text
//     up to the branch point, so equal offsets keep the shared lines side
//     by side; a fraction would drift as the lengths differ.

import {
  type ReactNode,
  type UIEvent,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
} from "react";

import type { Selection } from "../../App.tsx";
import {
  type ReplyToken,
  type Word,
  beforePoint,
  wordsOf,
} from "../../session/branch.ts";
import {
  type ReplyState,
  type Run,
  type Session,
  type SessionState,
  running,
  textOf,
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
    reply: null,
    steered: null,
  });
  const echo = useRef<HTMLElement | null>(null);
  const follow = (from: Which) => {
    const self = panes.current[from];
    const other = panes.current[from === "reply" ? "steered" : "reply"];
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
            <>
              {running(run) && (
                <button type="button" onClick={() => session.stop()}>
                  Stop
                </button>
              )}
              <button type="submit" className="primary" disabled={!canRun}>
                Run
              </button>
            </>
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
        <ReplyColumn
          session={session}
          run={run}
          pane={(el) => {
            panes.current.reply = el;
          }}
          onScroll={() => follow("reply")}
          onOpen={onOpen}
        />
        <SteeredColumn
          session={session}
          run={run}
          pane={(el) => {
            panes.current.steered = el;
          }}
          onScroll={() => follow("steered")}
          onOpen={onOpen}
        />
      </div>
    </main>
  );
}

// A reply's scrolling pane: it follows the end of a streaming text while
// it was at the end, and goes back to the top when the text empties.
function usePane(
  text: string,
  pane: (element: HTMLElement | null) => void,
  onScroll: () => void,
) {
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
  return {
    ref: (el: HTMLElement | null) => {
      element.current = el;
      pane(el);
    },
    onScroll: (event: UIEvent<HTMLElement>) => {
      const el = event.currentTarget;
      atEnd.current = el.scrollHeight - el.scrollTop - el.clientHeight < 4;
      onScroll();
    },
  };
}

function ReplyColumn(props: {
  readonly session: Session;
  readonly run: Run | null;
  readonly pane: (element: HTMLElement | null) => void;
  readonly onScroll: () => void;
  readonly onOpen: (reply?: Which, chosen?: Selection) => void;
}) {
  const { session, run, pane, onScroll, onOpen } = props;
  const reply = run?.reply ?? null;
  const text = reply === null || reply.kind === "waiting" ? "" : reply.text;
  const scroll = usePane(text, pane, onScroll);
  const done = reply !== null && reply.kind === "done" ? reply : null;
  const words = useMemo(
    () => (done === null ? [] : wordsOf(done.tokens)),
    [done],
  );
  const busy = running(run);
  const point = run?.point ?? 0;
  // The branch points the arrows step through: the start, then each
  // offered word's end.
  const stops = useMemo(
    () => [0, ...words.filter(offered).map((w) => w.end)],
    [words],
  );
  const move = (by: number) => {
    if (done === null || busy) return;
    const here = stops.findLastIndex((stop) => stop <= point);
    const next = stops[Math.max(0, Math.min(stops.length - 1, here + by))];
    if (next !== undefined && next !== point) session.branchAt(next);
  };
  return (
    <section aria-label="Reply" className="reply">
      <h3>
        <span>Reply</span>
        <output aria-label="Reply ending">{endingText(run, reply)}</output>
      </h3>
      <output
        aria-label="Reply"
        className="reply-text"
        tabIndex={0}
        ref={scroll.ref}
        onScroll={scroll.onScroll}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault();
            move(event.key === "ArrowLeft" ? -1 : 1);
          }
        }}
      >
        {done === null
          ? text
          : words.map((word) => (
              <WordView
                key={word.start}
                word={word}
                tokens={done.tokens}
                point={point}
                disabled={busy}
                onPick={() => session.branchAt(word.end)}
              />
            ))}
      </output>
      {done !== null && (
        <p className="branch-row">
          <button
            type="button"
            aria-pressed={point === 0}
            disabled={busy}
            onClick={() => session.branchAt(0)}
          >
            From the start
          </button>
          <span className="hint">
            Click a word to read its neurons and steer after it · ← → move
          </span>
        </p>
      )}
      {reply !== null && (
        <button
          type="button"
          className="neurons"
          aria-label="Neurons in the reply"
          onClick={() => onOpen("reply")}
        >
          Neurons ↗
        </button>
      )}
    </section>
  );
}

// A word that holds more than white space is a branch point.
function offered(word: Word): boolean {
  return /\S/u.test(word.text);
}

// One word of the done reply: its leading white space, then a button for
// the rest when offered; the point's mark where it falls in or after it,
// and what follows the point faded.
function WordView(props: {
  readonly word: Word;
  readonly tokens: readonly ReplyToken[];
  readonly point: number;
  readonly disabled: boolean;
  readonly onPick: () => void;
}) {
  const { word, tokens, point, disabled, onPick } = props;
  const lead = /^\s*/u.exec(word.text)?.[0] ?? "";
  const after = point > 0 && word.start >= point;
  if (!offered(word)) {
    return <span className={after ? "past" : undefined}>{word.text}</span>;
  }
  const rest = word.text.slice(lead.length);
  // Where the point falls inside the word, in its rest's characters.
  const inside = word.start < point && point < word.end;
  const cut = inside
    ? Math.max(0, textOf(tokens.slice(word.start, point)).length - lead.length)
    : rest.length;
  return (
    <>
      <span className={after ? "past" : undefined}>{lead}</span>
      <button
        type="button"
        className={after ? "word past" : "word"}
        aria-pressed={point === word.end}
        tabIndex={point === word.end ? 0 : -1}
        disabled={disabled}
        onClick={onPick}
      >
        {rest.slice(0, cut)}
        {inside && (
          <>
            <span className="point" aria-hidden />
            <span className="past">{rest.slice(cut)}</span>
          </>
        )}
      </button>
      {point === word.end && <span className="point" aria-hidden />}
    </>
  );
}

function SteeredColumn(props: {
  readonly session: Session;
  readonly run: Run | null;
  readonly pane: (element: HTMLElement | null) => void;
  readonly onScroll: () => void;
  readonly onOpen: (reply?: Which, chosen?: Selection) => void;
}) {
  const { session, run, pane, onScroll, onOpen } = props;
  const branch = run?.branch ?? null;
  const reply = branch?.reply ?? null;
  const own = reply === null || reply.kind === "waiting" ? "" : reply.text;
  const base = run?.reply.kind === "done" ? run.reply.tokens : null;
  const kept =
    branch === null || base === null ? "" : textOf(base.slice(0, branch.from));
  const scroll = usePane(kept + own, pane, onScroll);
  return (
    <section aria-label="Steered reply" className="reply reply-edited">
      <h3>
        <span>{steeredTitle(branch, base)}</span>
        <output aria-label="Steered reply ending">
          {run === null
            ? ""
            : branch === null
              ? "Not steered"
              : endingText(run, reply)}
        </output>
      </h3>
      <output
        aria-label="Steered reply"
        className="reply-text"
        tabIndex={0}
        ref={scroll.ref}
        onScroll={scroll.onScroll}
      >
        {kept !== "" && <span className="past">{kept}</span>}
        {own}
      </output>
      {branch !== null && (
        <p className="branch-row">
          <button
            type="button"
            className="primary"
            disabled={reply?.kind !== "done"}
            onClick={() => session.accept()}
          >
            Accept
          </button>
          <span className="hint">
            Make this the reply, to steer from again; the pins stay
          </span>
        </p>
      )}
      {branch !== null && (
        <button
          type="button"
          className="neurons"
          aria-label="Neurons in the steered reply"
          onClick={() => onOpen("steered")}
        >
          Neurons ↗
        </button>
      )}
    </section>
  );
}

// "Steered", and once there is a steered reply its pins and where it
// branched.
function steeredTitle(
  branch: Run["branch"],
  base: readonly ReplyToken[] | null,
): ReactNode {
  if (branch === null) return "Steered";
  const pins = `${branch.pins.length} ${branch.pins.length === 1 ? "pin" : "pins"}`;
  if (branch.from === 0 || base === null) {
    return `Steered · ${pins} · from the start`;
  }
  return (
    <>
      Steered · {pins} · after{" "}
      <q className="as-written">{beforePoint(base, branch.from).trim()}</q>
    </>
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
  if (run === null || reply === null) return "";
  switch (reply.kind) {
    case "waiting":
      return "Waiting";
    case "running":
      return `Running: ${COUNT.format(reply.tokens)} tokens`;
    case "done":
      return `Done: ${reply.end}, ${COUNT.format(reply.tokens.length)} tokens`;
    case "failed":
      return `Failed at ${reply.failure.stage}: ${reply.failure.message}`;
  }
}
