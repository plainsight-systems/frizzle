// Axis H (view): the Ride, the workbench's main view: a drive through the
// model's layers at the branch point. Each layer is a pane showing the word
// the model would most likely say if it stopped there, and a sign reads it
// out, layer by layer. Pin neurons and ride, and the road forks where the
// answer changes.
//
// Contract
//   - Ride({session, state, layer, onLayer, onSelect, onSteer}): over the
//     run's branch point (views/deck/cards.ts, neuronsOf), the reply's side
//     and, when a steered reply branches here, the steered side.
//   - Without the reply's captures: its words for why over the scene, which
//     stays mounted, with the focus, while a new point is read.
//   - The reply, under the scene like a film's subtitle: the words written up
//     to the point, as they read, wrapped, then a blank for the next one; on
//     arrival at the last layer, the rest of the road's reply, all of it in
//     view up to 30% of the window's height, following the point past
//     that. A token is pressed to ride from just after it (session.branchAt); in a steered
//     reply, that first makes it the reply (session.accept), the pins
//     staying. "Back to the start" rides from the reply's beginning. All
//     disabled while a run goes.
//   - The scene (scene.ts): ← → or ↑ ↓ step a layer (the frame's front
//     layer, onLayer, so the inspector follows), the wheel or a vertical
//     drag move along the road, Space or Drive drives to the end, and
//     Again starts over. A clicked light or dial selects that neuron
//     (onSelect), for the inspector to pin. A neuron selected anywhere
//     brings the ride to its layer.
//   - The sign, status "Ride reading": "Layer {l}", its top guess, the word
//     the model would most likely say if it stopped there, and "top guess";
//     gold when the reply goes on to write that word. Under the scene, the
//     layer's two runners-up.
//   - The main button drives, unless pins are held that the ride here has
//     not used: then it is "Ride with {n} pinned" (onSteer), which runs and
//     keeps the Ride open, as Space does; when the steered reply arrives,
//     the ride starts two layers before the fork and drives on. With a
//     steered reply: a button to ride the other road. When the two roads
//     never differ: "Your pins did not change any layer's best guess".
//   - The route under it: a station a layer, labelled where the guess
//     changes; two lines from the fork. Pressing a station goes there.
//   - Reduced motion: the view moves a whole layer at a time, at the same
//     pace when driving, and the fork appears at once.
//
// Design
//   - Word by word from the engine's own readout (capture/capture.h): no
//     probability is computed or shown, and only the last layer's guess is
//     something the model computes for itself. The note under the route
//     says so.
//   - The animation state is a ref read by one requestAnimationFrame loop;
//     React renders only when the shown layer or arrival changes.

import { Fragment, useEffect, useMemo, useRef, useState } from "react";

import type { Selection } from "../../App.tsx";
import type { Edit } from "../../engine/edits.ts";
import type { ReplyToken } from "../../session/branch.ts";
import type { Session, SessionState } from "../../session/session.ts";
import { running } from "../../session/session.ts";
import {
  type NextToken,
  type Road,
  forkOf,
  onTheWay,
  roadOf,
  wordOf,
} from "../../shape/ride.ts";
import { neuronsOf } from "../deck/cards.ts";
import { type Target, drawScene } from "./scene.ts";

const COUNT = new Intl.NumberFormat("en-US");
const PACE = 1.35; // layers a second
const PACE_NEAR_FORK = 0.75;

const nextOf = (token: ReplyToken | undefined): NextToken | null =>
  token === undefined ? null : { id: token.token, text: token.text };

export function Ride(props: {
  readonly session: Session;
  readonly state: SessionState;
  readonly layer: number;
  readonly onLayer: (layer: number) => void;
  readonly onSelect: (chosen: Selection) => void;
  readonly onSteer: () => void;
}) {
  const { session, state, layer, onLayer, onSelect, onSteer } = props;
  const { run, pins } = state;
  const replySide = neuronsOf(run, "reply");
  const steeredSide = neuronsOf(run, "steered");
  const done = run?.reply.kind === "done" ? run.reply : null;
  const point = run?.point ?? 0;
  const tokens = useMemo(() => done?.tokens ?? [], [done]);
  const branchReply = run?.branch?.reply;
  const detourTokens = useMemo(
    () => (branchReply?.kind === "done" ? branchReply.tokens : []),
    [branchReply],
  );
  const replyCaptures = replySide.kind === "taken" ? replySide.captures : null;
  const steeredCaptures =
    steeredSide.kind === "taken" ? steeredSide.captures : null;
  const nextToken = tokens[point];
  const replyRoad = useMemo(
    () =>
      replyCaptures === null ? null : roadOf(replyCaptures, nextOf(nextToken)),
    [replyCaptures, nextToken],
  );
  const detourRoad = useMemo(
    () =>
      steeredCaptures === null
        ? null
        : roadOf(steeredCaptures, nextOf(detourTokens[0])),
    [steeredCaptures, detourTokens],
  );
  const [onOriginal, setOnOriginal] = useState(false);
  const riding = detourRoad !== null && !onOriginal;
  const main = riding ? detourRoad : replyRoad;
  const ghost = riding ? replyRoad : detourRoad;
  const forkAt = useMemo(
    () =>
      replyRoad !== null && detourRoad !== null
        ? forkOf(replyRoad, detourRoad)
        : null,
    [replyRoad, detourRoad],
  );
  const tail = useMemo(
    () => (riding ? detourTokens : tokens.slice(point)),
    [riding, detourTokens, tokens, point],
  );
  const upcoming = useMemo(() => tail.map((t) => wordOf(t.text)), [tail]);
  const ghostUpcoming = useMemo(
    () =>
      (riding ? tokens.slice(point) : detourTokens).map((t) => wordOf(t.text)),
    [riding, tokens, point, detourTokens],
  );
  // The road on show: the last one while the next point is read (a few
  // hundredths of a second), so the page does not jump.
  const held = useRef<Road | null>(null);
  if (main !== null) held.current = main;
  const shownRoad = main ?? held.current;
  const last = (shownRoad?.layers.length ?? 1) - 1;
  const busy = running(run);
  const pinsByLayer = useMemo(() => {
    const out = new Map<number, number[]>();
    for (const p of pins)
      out.set(p.layer, [...(out.get(p.layer) ?? []), p.neuron]);
    return out;
  }, [pins]);

  const canvas = useRef<HTMLCanvasElement>(null);
  const scene = useRef<HTMLDivElement>(null);
  const sentence = useRef<HTMLParagraphElement>(null);
  const live = useRef({
    t: layer,
    target: layer,
    fork: 0,
    driving: false,
    clock: 0,
    lean: 0,
    cameraX: 0,
    shown: layer,
    sent: new Set<number>(), // layers told to the frame, not yet come back
    targets: [] as readonly Target[],
  });
  const [shown, setShown] = useState(layer);
  const [arrived, setArrived] = useState(false);
  const [driving, setDriving] = useState(false);
  const still = useMemo(
    () => matchMedia("(prefers-reduced-motion: reduce)").matches,
    [],
  );

  const drive = (on: boolean) => {
    live.current.driving = on;
    setDriving(on);
  };
  const go = (to: number) => {
    drive(false);
    live.current.target = Math.max(0, Math.min(last, to));
  };

  // What the animation loop reads, current every render.
  const latest = useRef({
    main,
    ghost,
    forkAt,
    upcoming,
    ghostUpcoming,
    pinsByLayer,
    last,
    onLayer,
    arrived,
  });
  latest.current = {
    main,
    ghost,
    forkAt,
    upcoming,
    ghostUpcoming,
    pinsByLayer,
    last,
    onLayer,
    arrived,
  };

  // A new branch point starts the ride over; a steered reply arriving here
  // drives through its fork.
  const place = useRef({ point, detour: detourRoad });
  useEffect(() => {
    const before = place.current;
    place.current = { point, detour: detourRoad };
    const l = live.current;
    if (before.point !== point) {
      l.t = 0;
      l.target = 0;
      l.fork = 0;
      l.driving = false;
      setDriving(false);
      setOnOriginal(false);
    } else if (detourRoad !== null && before.detour !== detourRoad) {
      setOnOriginal(false);
      if (forkAt !== null) {
        l.t = Math.max(0, forkAt - 2);
        l.target = l.t;
        l.driving = true;
        setDriving(true);
      }
    }
  }, [point, detourRoad, forkAt]);

  // A layer the ride did not report itself: a neuron selected elsewhere.
  useEffect(() => {
    const l = live.current;
    if (l.sent.has(layer)) {
      if (layer === l.shown) l.sent.clear();
      return;
    }
    if (layer === l.shown) return;
    l.driving = false;
    setDriving(false);
    l.target = layer;
  }, [layer]);

  useEffect(() => {
    scene.current?.focus({ preventScroll: true });
  }, []);

  // A reply too long to show whole keeps the point in view.
  useEffect(() => {
    const box = sentence.current;
    const mark = box?.querySelector<HTMLElement>("[data-here]");
    if (
      box === null ||
      box === undefined ||
      mark === null ||
      mark === undefined
    ) {
      return;
    }
    box.scrollTop = Math.max(0, mark.offsetTop - box.offsetTop - 30);
  }, [point, arrived, tail]);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (element === null || element === undefined || !context) return;
    let frame = 0;
    let previous = performance.now();
    let width = 0;
    let height = 0;
    const size = new ResizeObserver(() => {
      const box = element.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio || 1, 2);
      width = box.width;
      height = box.height;
      element.width = Math.round(width * dpr);
      element.height = Math.round(height * dpr);
    });
    size.observe(element);
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      const road = latest.current;
      if (road.main === null || width === 0) return;
      const l = live.current;
      const dt = Math.min(0.05, (now - previous) / 1000);
      previous = now;
      l.clock += dt;
      const fork = road.ghost !== null ? road.forkAt : null;
      if (l.driving) {
        const near = fork !== null && Math.abs(l.t - fork) < 2.2;
        l.target = Math.min(
          road.last,
          l.target + (near ? PACE_NEAR_FORK : PACE) * dt,
        );
        if (l.target >= road.last && l.t > road.last - 0.02) {
          l.driving = false;
          setDriving(false);
        }
      }
      const gap = l.target - l.t;
      l.t = still ? Math.round(l.target) : l.t + gap * (1 - Math.exp(-dt * 6));
      const wanted = fork !== null ? 1 : 0;
      l.fork = still
        ? wanted
        : l.fork + (wanted - l.fork) * (1 - Math.exp(-dt * 3.2));
      const dpr = element.width / width;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const drawn = drawScene(context, {
        width,
        height,
        t: l.t,
        fork: l.fork,
        clock: l.clock,
        still,
        moving: Math.min(1, Math.abs(gap) * 2 + (l.driving ? 1 : 0)),
        lean: l.lean,
        arrived: road.arrived,
        main: road.main,
        ghost: fork !== null ? road.ghost : null,
        forkAt: fork ?? road.last + 1,
        upcoming: road.upcoming,
        ghostUpcoming: road.ghostUpcoming,
        pins: road.pinsByLayer,
      });
      l.lean +=
        ((drawn.cameraX - l.cameraX) / Math.max(dt, 0.001)) * 0.012 * 0.12 -
        l.lean * 0.12;
      l.cameraX = drawn.cameraX;
      l.targets = drawn.targets;
      const at = Math.max(0, Math.min(road.last, Math.round(l.t)));
      if (at !== l.shown) {
        l.shown = at;
        l.sent.add(at);
        setShown(at);
        road.onLayer(at);
      }
      setArrived(l.t >= road.last - 0.1);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      size.disconnect();
    };
  }, [still]);

  const dragging = useRef<{ y: number; target: number } | null>(null);

  const row = shownRoad?.layers[Math.min(shown, last)];
  const stops: readonly ReplyStop[] = [
    ...tokens
      .slice(0, point)
      .map((t, j): ReplyStop => ({ text: t.text, kind: "past", to: j + 1 })),
    ...(arrived
      ? tail.map((t, j): ReplyStop => ({
          text: t.text,
          kind: j === 0 ? "now" : "later",
          to: point + j + 1,
        }))
      : []),
  ];
  const best = row?.guesses[0] ?? "";
  const pick = (index: number, kept: boolean) => {
    if (busy) return;
    if (!kept && riding) session.accept();
    session.branchAt(index);
  };
  const branch = run?.branch ?? null;
  const steering = branch !== null && branchReply?.kind !== "done";
  const canSteer = pins.length > 0 && !samePins(pins, branch?.pins ?? []);

  return (
    <section className="column ride-column" aria-label="Ride">
      <div
        ref={scene}
        className="ride-scene"
        tabIndex={0}
        aria-label="The road through the layers; arrow keys step, Space drives"
        onKeyDown={(event) => {
          const by =
            event.key === "ArrowRight" || event.key === "ArrowUp"
              ? 1
              : event.key === "ArrowLeft" || event.key === "ArrowDown"
                ? -1
                : 0;
          if (by !== 0) {
            event.preventDefault();
            event.stopPropagation();
            go(Math.round(live.current.target) + by);
          } else if (event.key === " ") {
            event.preventDefault();
            mainAction();
          }
        }}
        onWheel={(event) => {
          drive(false);
          live.current.target = Math.max(
            0,
            Math.min(last, live.current.target + event.deltaY / 340),
          );
        }}
      >
        <canvas
          ref={canvas}
          aria-hidden
          onPointerDown={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            const px = event.clientX - box.left;
            const py = event.clientY - box.top;
            const hit = live.current.targets.find(
              (t) => Math.hypot(px - t.x, py - t.y) < t.r,
            );
            if (hit !== undefined) {
              onSelect({ layer: hit.layer, neurons: [hit.neuron] });
              return;
            }
            dragging.current = {
              y: event.clientY,
              target: live.current.target,
            };
            event.currentTarget.setPointerCapture(event.pointerId);
            drive(false);
          }}
          onPointerMove={(event) => {
            const from = dragging.current;
            if (from === null) return;
            live.current.target = Math.max(
              0,
              Math.min(last, from.target + (from.y - event.clientY) / 46),
            );
          }}
          onPointerUp={() => {
            dragging.current = null;
          }}
        />
        {shownRoad === null ||
        (main === null &&
          replySide.kind === "none" &&
          replySide.why !== "Waiting") ? (
          <p className="ride-why">
            <output aria-label="Ride">
              {replySide.kind === "none" ? replySide.why : "Waiting"}
            </output>
          </p>
        ) : (
          <div
            className={
              onTheWay(best, upcoming) ? "ride-sign there" : "ride-sign"
            }
          >
            <output aria-label="Ride reading">
              <small>Layer {shown}</small>
              <b>{best}</b>
              <i>top guess</i>
            </output>
          </div>
        )}
      </div>
      {shownRoad !== null && (
        <>
          <div className="ride-read">
            <div>
              <p
                ref={sentence}
                className="ride-sentence"
                aria-label="The reply. Press a word to ride from it."
              >
                {wordsOf(stops).map((word) => (
                  <Fragment key={word.stops[0]?.to}>
                    {word.lead}
                    <span className="ride-word">
                      {word.stops.map((stop) => (
                        <button
                          key={stop.to}
                          type="button"
                          className={stop.kind}
                          data-here={stop.kind === "now" ? "" : undefined}
                          disabled={busy}
                          onClick={() => pick(stop.to, stop.kind === "past")}
                        >
                          {stop.text}
                        </button>
                      ))}
                    </span>
                  </Fragment>
                ))}
                {!arrived && (
                  <span
                    className="ride-blank"
                    data-here=""
                    aria-label="the next word"
                  />
                )}
              </p>
              <p className="ride-more">
                Also likely: {row?.guesses.slice(1).join(", ")}
                {point > 0 && (
                  <button
                    type="button"
                    className="ride-back"
                    disabled={busy}
                    onClick={() => pick(0, true)}
                  >
                    Back to the start
                  </button>
                )}
              </p>
            </div>
            <div className="ride-controls">
              <button
                type="button"
                aria-label="Previous layer"
                disabled={shown === 0}
                onClick={() => go(shown - 1)}
              >
                ‹
              </button>
              <button
                type="button"
                className="primary"
                disabled={steering || (canSteer && busy)}
                onClick={mainAction}
              >
                {steering
                  ? "Steering…"
                  : canSteer
                    ? `Ride with ${COUNT.format(pins.length)} pinned`
                    : driving
                      ? "Pause"
                      : shown >= last
                        ? "Again"
                        : "Drive"}
              </button>
              <button
                type="button"
                aria-label="Next layer"
                disabled={shown === last}
                onClick={() => go(shown + 1)}
              >
                ›
              </button>
              {detourRoad !== null && (
                <button
                  type="button"
                  onClick={() => setOnOriginal(!onOriginal)}
                >
                  {riding ? "Ride the original" : "Ride the detour"}
                </button>
              )}
            </div>
          </div>
          {detourRoad !== null && forkAt === null && (
            <p className="ride-more ride-note">
              Your pins did not change any layer’s best guess.
            </p>
          )}
          <Route
            main={shownRoad}
            ghost={forkAt === null ? null : ghost}
            forkAt={forkAt}
            upcoming={upcoming}
            at={shown}
            onGo={go}
          />
          <p className="ride-more ride-note">
            Each top guess is the word the model would most likely say if it
            stopped at that layer. Only layer {last}’s is computed by the model
            itself; the others are a side reading. Click a light to pick that
            neuron.
          </p>
        </>
      )}
    </section>
  );

  function mainAction() {
    if (steering || (canSteer && busy)) return;
    if (canSteer) {
      drive(false);
      onSteer();
      return;
    }
    const l = live.current;
    if (!l.driving && l.target >= last) {
      l.t = 0;
      l.target = 0;
      setArrived(false);
    }
    drive(!l.driving);
  }
}

// Whether two pin sets are the same pins, in any order.
function samePins(a: readonly Edit[], b: readonly Edit[]): boolean {
  const key = (pins: readonly Edit[]) =>
    pins
      .map((p) => `${p.layer}:${p.neuron}:${p.kind}:${p.value}`)
      .sort()
      .join();
  return key(a) === key(b);
}

// A token of the reply, and the point just after it.
interface ReplyStop {
  readonly text: string;
  readonly kind: "past" | "now" | "later";
  readonly to: number;
}

// The reply as it reads: tokens with no space between them kept together as
// one word, so a line never breaks inside it. White space alone is not a
// place to ride from, and line breaks read as a space, keeping it compact.
function wordsOf(stops: readonly ReplyStop[]) {
  const words: { lead: string; stops: ReplyStop[] }[] = [];
  let pending = "";
  for (const stop of stops) {
    const shown = stop.text.replace(/\n+/gu, " ");
    const lead = pending + (/^\s*/u.exec(shown)?.[0] ?? "");
    const rest = shown.trimStart();
    if (rest === "") {
      pending = lead;
      continue;
    }
    pending = "";
    const item = { ...stop, text: rest };
    const current = words.at(-1);
    if (current === undefined || lead !== "") {
      words.push({ lead, stops: [item] });
    } else current.stops.push(item);
  }
  return words;
}

// The route: a station a layer, labelled where the best guess changes,
// latest first so the answer keeps its label when labels would collide.
function Route(props: {
  readonly main: Road;
  readonly ghost: Road | null;
  readonly forkAt: number | null;
  readonly upcoming: readonly string[];
  readonly at: number;
  readonly onGo: (layer: number) => void;
}) {
  const { main, ghost, forkAt, upcoming, at, onGo } = props;
  const count = main.layers.length;
  const x = (i: number) => 24 + (i * 952) / Math.max(1, count - 1);
  const parted = (i: number) =>
    forkAt === null ? 0 : Math.min(1, Math.max(0, (i - forkAt + 0.7) / 1.5));
  const y = (i: number, side: number) => 46 + side * 18 * parted(i);
  const line = (side: number) => {
    let d = "";
    for (let i = 0; i <= count - 1; i += 0.5) {
      d += `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(i, side).toFixed(1)}`;
    }
    return d;
  };
  const labels = (road: Road, side: number, from: number, above: boolean) => {
    const changes: { i: number; word: string }[] = [];
    let previous: string | null = null;
    road.layers.forEach((layer, i) => {
      const word = layer.guesses[0] ?? "";
      if (i >= from && word !== previous) changes.push({ i, word });
      previous = word;
    });
    const placed: number[] = [];
    return changes.reverse().flatMap(({ i, word }) => {
      if (placed.some((p) => Math.abs(p - x(i)) < 54)) return [];
      placed.push(x(i));
      return [
        <text
          key={`${side}-${i}`}
          x={x(i)}
          y={y(i, side) + (above ? -11 : 20)}
          textAnchor="middle"
          className={side <= 0 && onTheWay(word, upcoming) ? "lit" : undefined}
        >
          {[...word].length > 9 ? `${[...word].slice(0, 8).join("")}…` : word}
        </text>,
      ];
    });
  };
  const side = ghost !== null ? -1 : 0;
  return (
    <div className="ride-route">
      <svg
        viewBox="0 0 1000 92"
        role="img"
        aria-label="The route through the layers"
      >
        {ghost !== null && <path className="other" d={line(1)} />}
        <path d={line(side)} />
        {labels(main, side, 0, true)}
        {ghost !== null && forkAt !== null && labels(ghost, 1, forkAt, false)}
        {main.layers.map((_, i) => (
          <g key={i} onClick={() => onGo(i)}>
            <circle
              cx={x(i)}
              cy={y(i, side)}
              r={i === at ? 6 : 2.8}
              className={i === at ? "at" : i < at ? "past" : undefined}
            />
            <rect x={x(i) - 18} y={0} width={36} height={92} />
          </g>
        ))}
      </svg>
    </div>
  );
}
