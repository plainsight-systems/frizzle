// Axis H (view): the MLP workbench, a layer over the whole window: a bar
// with the prompt, the pins and Run; under it the layer deck and the
// inspector beside it, over one reply's neurons, one front layer and one
// selected neuron (docs/design/concepts/deck.html, the reviewed concept).
//
// Contract
//   - Workbench({session, state, message, canRun, onRun, onClose, which,
//     onWhich, front, onFront, selection, onSelect}): a dialog "Neuron
//     workbench", modal. When it opens, the focus goes to the deck's cards
//     when there are any, so the wheel and the keys work on them without a
//     click first (a click there would pick a neuron); otherwise to the
//     dialog.
//   - The bar: the prompt it works on (the run's message, or the message
//     typed when there is no run yet) › "MLP neurons"; the pins held, as
//     chips (PinChips.tsx: showing one selects its neurons here); button "Run"
//     (onRun, which also closes the workbench); button "Close", and Esc,
//     which close it without running (onClose).
//   - The deck (Deck.tsx) and the inspector (Inspector.tsx), on the reply
//     `which`, the front layer `front`, the neurons in `selection`; all
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
import type { Session, SessionState } from "../../session/session.ts";
import { PinChips } from "../prompt/PinChips.tsx";
import { Deck } from "./Deck.tsx";
import { Inspector } from "./Inspector.tsx";
import { type Which, capturesOf, cardsOf, facesOf, replyOf } from "./cards.ts";

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
  const { which, onWhich, front, onFront, selection, onSelect } = props;
  const dialog = useRef<HTMLDivElement>(null);
  const reply = replyOf(state.run, which);
  const captures = capturesOf(reply);
  const cards = useMemo(() => cardsOf(captures), [captures]);
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
          <span className="muted"> › MLP neurons</span>
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
          run={state.run}
          reply={reply}
          cards={cards}
          faces={faces}
          previewing={which === "original"}
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
