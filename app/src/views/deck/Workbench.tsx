// Axis H (view): the MLP workbench, a layer over the whole window: a bar
// with the prompt, the branch point, the pins and Run; under it the layer
// deck and the inspector beside it, over one side's neurons at the branch
// point, one front layer and one selected neuron (docs/design/concepts/deck.html, the reviewed concept).
//
// Contract
//   - Workbench({session, state, message, canRun, onRun, onClose, which,
//     onWhich, front, onFront, selection, onSelect}): a dialog "Neuron
//     workbench", modal. When it opens, the focus goes to the deck's cards
//     when there are any, so the wheel and the keys work on them without a
//     click first (a click there would pick a neuron); otherwise to the
//     dialog.
//   - The bar: the prompt it works on (the run's message, or the message
//     typed when there is no run yet) › "MLP neurons" and where: "at the
//     reply's start" (point 0) or "after “{word}”", the word the point
//     falls in up to the point (session/branch.ts, beforePoint), trimmed;
//     the pins held, as
//     chips (PinChips.tsx: showing one selects its neurons here); button "Run"
//     (onRun, which also closes the workbench); button "Close", and Esc,
//     which close it without running (onClose). No Stop here: Stop is
//     beside the prompt's Run (PromptView.tsx), reached once this closes.
//   - The deck (Deck.tsx) and the inspector (Inspector.tsx), on the side
//     `which` (cards.ts, neuronsOf), or the reply's while the run has no
//     steered reply (none yet, or one just accepted), whose radio is then
//     disabled, the front layer `front`, the neurons in `selection`; all
//     three are the frame's (App.tsx), so they last between visits.
//     Selecting brings the selection's layer to the front (onSelect does
//     both).
//   - Pinning is the session's (session.pin, session.unpin): what is pinned
//     is held for the next run.
//
// Design
//   - The cards are folded and painted here, once each time the shown
//     captures change (cards.ts), and handed to both columns; the faces
//     the deck shows, with what the pins will send (cards.ts, facesOf),
//     each time the pins change too.
//   - Mounted only while open: the deck's WebGL context and textures exist
//     only then.

import { useEffect, useMemo, useRef } from "react";

import type { Selection } from "../../App.tsx";
import { beforePoint } from "../../session/branch.ts";
import type { Run, Session, SessionState } from "../../session/session.ts";
import { PinChips } from "../prompt/PinChips.tsx";
import { Deck } from "./Deck.tsx";
import { Inspector } from "./Inspector.tsx";
import { type Which, cardsOf, facesOf, neuronsOf } from "./cards.ts";

export function Workbench(props: {
  readonly session: Session;
  readonly state: SessionState;
  readonly message: string;
  readonly canRun: boolean;
  readonly onRun: () => void;
  readonly onClose: () => void;
  readonly which: Which;
  readonly onWhich: (which: Which) => void;
  readonly front: number;
  readonly onFront: (layer: number) => void;
  readonly selection: Selection | null;
  readonly onSelect: (chosen: Selection) => void;
}) {
  const { session, state, message, canRun, onRun, onClose } = props;
  const { onWhich, front, onFront, selection, onSelect } = props;
  // The steered side exists only with a steered reply.
  const which: Which =
    props.which === "steered" && (state.run?.branch ?? null) === null
      ? "reply"
      : props.which;
  const dialog = useRef<HTMLDivElement>(null);
  const neurons = neuronsOf(state.run, which);
  const captures = neurons.kind === "taken" ? neurons.captures : null;
  const cards = useMemo(
    () => (captures === null ? null : cardsOf({ kind: "taken", captures })),
    [captures],
  );
  const faces = useMemo(
    () => (cards === null ? null : facesOf(cards, state.pins, which)),
    [cards, state.pins, which],
  );
  const count = cards?.length ?? 0;
  const layer = count === 0 ? front : Math.max(0, Math.min(front, count - 1));
  const prompt = (state.run?.message ?? message).trim();

  // The deck takes the focus first (its own effect runs before this one);
  // the dialog only when nothing inside it has.
  useEffect(() => {
    const element = dialog.current;
    if (element !== null && !element.contains(document.activeElement)) {
      element.focus();
    }
  }, []);

  return (
    <div
      ref={dialog}
      className="workbench"
      role="dialog"
      aria-modal
      aria-label="Neuron workbench"
      tabIndex={-1}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          onClose();
        }
      }}
    >
      <header className="workbench-bar">
        <button type="button" aria-label="Close" onClick={onClose}>
          ← Close
        </button>
        <p className="prompt-summary">
          <span>{prompt === "" ? "No prompt yet" : `“${prompt}”`}</span>
          <span className="muted"> › MLP neurons {pointText(state.run)}</span>
        </p>
        <PinChips session={session} pins={state.pins} onPick={onSelect} />
        <button
          type="button"
          className="primary"
          disabled={!canRun}
          onClick={onRun}
        >
          Run
        </button>
      </header>
      <div className="workbench-body">
        <Deck
          neurons={neurons}
          cards={cards}
          faces={faces}
          previewing={which === "reply"}
          layer={layer}
          onLayer={onFront}
          pins={state.pins}
          selection={selection}
          onSelect={onSelect}
          autoFocus
        />
        <Inspector
          session={session}
          state={state}
          which={which}
          onWhich={onWhich}
          cards={cards}
          layer={layer}
          selection={selection}
          onSelect={onSelect}
        />
      </div>
    </div>
  );
}

// Where the neurons are read: the reply's start, or after the text of the
// word the point falls in, up to the point (an accepted reply's point may
// fall inside a word, where its steered reply went on without a space).
function pointText(run: Run | null): string {
  if (run === null || run.reply.kind !== "done" || run.point === 0) {
    return "at the reply's start";
  }
  return `after “${beforePoint(run.reply.tokens, run.point).trim()}”`;
}
