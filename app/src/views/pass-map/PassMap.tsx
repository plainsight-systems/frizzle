// Axis H (view): the pass map, the workbench's third view: one layer at
// the branch point, isometric, each stage it computes drawn as the last
// position's vector at its real width, so the computation's shape shows
// (the residual's width, Q's heads, attention over the keys read, the MLP
// widening and coming back), and what the run captured in colour
// (docs/design/concepts/matrices.html, which draws every token's row;
// Frizzle draws the last position's: research/plan.md, Views).
//
// Contract
//   - PassMap({neurons, layer, onLayer, onOpenNeuron}): over the captures
//     of the side the workbench shows (views/deck/cards.ts, neuronsOf), as
//     the run computed them: no pin's value is overlaid, as on the Map.
//   - Status "Pass": "{count} layers" with captures; without them the
//     page's words for why (cards.ts), and no scene; "Not drawn: this
//     browser gives no WebGL 2" when it has none, and no scene.
//   - The head: button "‹" (aria-label "Previous layer"), "Layer {l}",
//     button "›" (aria-label "Next layer"), each disabled at its end (0,
//     the last layer); then the layer's attention, from the captures
//     (shape/pass.ts, window and full), never reckoned: "Global attention:
//     back to the first token" when full, else "Local attention: the last
//     {window} tokens", the window grouped in en-US. With the stage
//     framed, a button "Whole layer".
//   - The scene's wrapper takes the focus (tabIndex 0) and, while it has
//     it, ← and → step the front layer (onLayer), clamped to the layers;
//     on the dialog they would move the view switch's radios.
//   - The scene: the stages of shape/pass.ts, passOf(captures, layer), in
//     order along the flow, left to right; each stage its rows side by side
//     along the flow and its width along the depth, flat on the ground,
//     seen from a fixed angle by an orthographic camera. A captured stage
//     is filled, each row a strip of marks; a stage not captured is
//     outlined only, never filled. Under each stage its label: the name in
//     bold, then "{rows} × {width}" ("{width}" for one row), widths grouped
//     in en-US, and "keys" after attention's width, "not captured" after a
//     stage not captured. Attention's label adds "of {position + 1}
//     tokens" when n < position + 1. The labels stand 8 CSS pixels under
//     the stage's near edge, on three lines in turn (stage s (s mod 3) ×
//     30 pixels further down), so neighbours do not overlap.
//   - The marks: as the Map's (shape/strip.ts): each mark the strongest of
//     a group of g values, g the smallest that gives every mark at least
//     one physical pixel along the stage's width on screen (groupFor), so a
//     framing never drops a value; the deck's heat ramp
//     (theme/palette.ts, DECK_PALETTE) at the stage's scale
//     (shape/pass.ts, passScales): the Map's column scale for residual in,
//     attention added, MLP neurons and MLP added, one scale across the
//     layers for Q, and 1 for attention's weights.
//   - Under the scene, the legend: "0", the deck's heat ramp, then "full
//     at", and each captured stage's name and scale, in the pass's order,
//     separated by "·", each kept on one line, the scale to two
//     significant digits (shape/numbers.ts, significant); then one
//     line: "size, either sign · one scale a stage across the layers, the
//     strongest 1% at full; attention's weights full at 1 · each mark the
//     strongest of its group, sized so every mark has a pixel · outlined:
//     not captured".
//   - Attention's keys are named by their tokens (shape/tokens.ts): beside
//     the stage, each span's label (labelOf) at its keys, a control token's
//     name in the muted colour, where the keys' spacing up the screen is at
//     least a line of the label's text (14.3 CSS pixels: 11 pixels at a
//     line height of 1.3, theme.css); where it is not,
//     no labels are drawn, and the reading names the token. A span cut by
//     the window's start is labelled at the keys it has in the window.
//   - Framing: "whole layer" at first and after each step, every stage in
//     view; clicking a stage other than MLP neurons frames that stage, as
//     large as the scene allows; "Whole layer" goes back. No free zoom or
//     pan: two framings, so the groups follow the framing.
//   - Pointing at a mark shows, in status "Pass reading":
//       a dimension or neuron  "Layer {l} · {stage}{ · row} · {kind} {i}
//                              of {width} · {value}", kind "dimension" or
//                              "neuron", the value signed with two places
//                              (shape/numbers.ts, signed) and the exact
//                              f32 in a <data value>; "strongest of
//                              {a}–{b}" when g > 1
//       an attention weight    "Layer {l} · Attention · head {h} · {w} on
//                              {token} · position {p}", w to two places and
//                              exact in a <data value>, the token as
//                              readingOf says it, p its input position;
//                              "strongest of positions {a}–{b}" when
//                              g > 1, input positions as p is
//     Pointing at a stage not captured reads "Layer {l} · {stage} · not
//     captured"; at nothing, "Point at a mark to read it". A reading
//     belongs to the marks it was read from: when they change (a step, a
//     new run, a framing, a resize), it goes back to "Point at a mark to
//     read it" until the pointer moves.
//   - Clicking a mark of MLP neurons opens that neuron on the deck
//     (onOpenNeuron(layer, neuron), the neuron the reading names, from
//     the last pointer move: a click's position is rounded to a whole CSS
//     pixel, and a mark can be half of one): the
//     workbench selects it and switches to the deck at its layer, the deck
//     taking the focus (Workbench.tsx). Clicking any other stage frames
//     it. A press that moves more than 4 CSS pixels is not a click.
//
// Algorithm
//   1. When the shown captures change: passScales(captures); spansOf(the
//      input) (shape/tokens.ts).
//   2. When the captures or the front layer change: passOf(captures,
//      layer); the scene's geometry from it: a stage's rows ROW deep along
//      the flow each, its width along the depth at COL a dimension or
//      neuron and KEY = 15 COL a key, GAP between stages; value 0 of a row
//      at the stage's near edge (z = 0), its last at the far one. Framing
//      reset to the whole layer.
//   3. When the geometry, the framing or the canvas's size change: the
//      camera fitted to the framed box (the whole layer's widened by GAP
//      each side for its end labels), LABELS CSS pixels kept free under
//      it for the stage labels, and placed beyond the box along its
//      direction so no stage falls behind it; for each captured stage, the
//      physical pixels its width spans on screen, from its two ends
//      projected, along the screen's major axis (max(|dx|, |dy|), as
//      rasterization samples a line), CSS pixels × devicePixelRatio,
//      rounded down; g = groupFor(width, pixels); each row's marks,
//      strongestOfGroups over the row's view of the words
//      (words.subarray(first, first + width)); the stage's texture, a
//      texel a row across (u) and a texel a mark along (v), from
//      paintStrip at the stage's scale, nearest filtering both ways.
//   4. Each render (a pointer move that changes the mark is one): every
//      stage label, and attention's key labels for the spans that reach
//      its window, placed at their anchors projected to the screen; the
//      key labels drawn only when one key spans at least a line up the
//      screen (|dy|).
//   5. Pointer over a captured stage: from the hit's texture coordinates,
//      row = floor(u × rows), mark = floor(v × marks); the value's index
//      from the row's marks; the group's indices g × mark to g × mark + g
//      − 1, cut at the width; for attention, the key's position firstKey +
//      index and its span (spanAt).
//
// Design and cost
//   - A view of the workbench beside the inspector, not a page of its own:
//     the front layer, the side shown and the selection are the frame's
//     (App.tsx), so stepping layers here and switching to the deck keeps
//     the place, and clicking a neuron is the deck's selection.
//   - react-three-fiber's orthographic camera, as the deck's Canvas
//     (frameloop "demand", dpr [1, 2]); no drei, so labels are HTML placed
//     by projection, as the concept places them.
//   - Camera at the concept's fixed angle, along (0.18, 1, 1) from the
//     target: the flow runs left to right across the screen and the depth
//     up it, so key labels sit one under another beside the attention
//     stage, and "fits" is a question of line height.
//   - Two framings and no free zoom: the concept's orbit controls pan and
//     zoom freely, and every zoom would change each stage's pixels and so
//     its groups. Framing a stage gives attention's keys room to be named;
//     a long window (512 keys) still has no room, and the reading names
//     each key.
//   - A key KEY = 15 COL long: a key is not a dimension, so attention has
//     a unit of its own, stated by its label. Fifteen, seen in the browser:
//     framing attention fits its longer side to the scene, so the keys
//     must be longer than its four rows are deep (4 × ROW = 1.2) for a
//     short prompt's keys to get a line each; at 8, 29 keys were 0.93 and
//     got none. A full local window (512 keys, as long as 7,680
//     dimensions) is then a little longer than the MLP's 6,912.
//   - Not captured is drawn and said, never filled (the no-facades rule):
//     the layer's shape includes stages the run computed and Frizzle did
//     not keep.
//   - A change of devicePixelRatio with no change of size (a window moved
//     to another screen) keeps the groups until the next resize, as the
//     Map's do where the browser gives no devicePixelContentBoxSize.
//   - Cost, counted for Gemma 3 1B: each framing or size change, the strip
//     pass over 1,152 × 3 + 4 × 256 + 6,912 + 4n values and six texture
//     uploads of at most a few thousand texels each; each step of layer,
//     that and passOf; each render, eleven stage labels and at most one
//     label a span placed, the spans being at most n, a pass over the
//     input's spans to find them; a pointer move re-renders only when it
//     changes the mark under it. One WebGL context, the deck's released:
//     the workbench mounts one view at a time.

import { type ThreeEvent, Canvas, useThree } from "@react-three/fiber";
import {
  type ReactNode,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Box3,
  BufferGeometry,
  DataTexture,
  Float32BufferAttribute,
  NearestFilter,
  OrthographicCamera,
  RGBAFormat,
  SRGBColorSpace,
  Vector3,
} from "three";

import type { Captures } from "../../protocol/messages.ts";
import { signed, significant } from "../../shape/numbers.ts";
import {
  type CapturedId,
  type Pass,
  type PassScales,
  type Stage,
  type StageId,
  passOf,
  passScales,
} from "../../shape/pass.ts";
import { groupFor, paintStrip, strongestOfGroups } from "../../shape/strip.ts";
import {
  type Span,
  labelOf,
  readingOf,
  spanAt,
  spansOf,
} from "../../shape/tokens.ts";
import { DECK_PALETTE, PASS_EDGE, PASS_OUTLINE } from "../../theme/palette.ts";
import type { Neurons } from "../deck/cards.ts";

type Taken = Extract<Captures, { readonly kind: "taken" }>;

// The scene's units (the header's Algorithm, step 2).
const ROW = 0.3; // a row, along the flow
const COL = 0.004; // a dimension or neuron, along the depth
const KEY = 15 * COL; // a key, along the depth
const GAP = 2.4; // between stages, along the flow
// The camera's fixed direction from its target (the concept's).
const ISO = new Vector3(0.18, 1, 1).normalize();
// Room around the framed box, as a share of the scene.
const PADDING = 1.15;
// The keys' projected spacing, in CSS pixels, below which no key is named.
const LINE = 14.3; // the label's line: 11px at a line height of 1.3
// A press that moves more than this, in CSS pixels, is not a click.
const CLICK = 4;
// Free under the framed box for the stage labels: 8 pixels, then two lines
// of label on the third of three staggered lines, 30 pixels apart (the
// header's Algorithm, step 3).
const LABELS = 100;

const COUNT = new Intl.NumberFormat("en-US");

// Where a stage lies in the scene: its rows from x0 along the flow, its
// width from z = 0 to z = −length.
interface Placed {
  readonly stage: Stage;
  readonly x0: number;
  readonly x1: number;
  readonly length: number;
}

// A captured stage's marks: each row's indices into the row, and the
// texture, a texel a row across and a texel a mark along.
interface Strip {
  readonly group: number;
  readonly marks: readonly Int32Array[];
  readonly count: number;
  readonly texture: DataTexture;
}

interface Size {
  readonly width: number;
  readonly height: number;
  readonly dpr: number;
}

// The mark under the pointer, or a stage not captured.
interface Pointed {
  readonly stage: number;
  readonly row: number;
  readonly mark: number;
}

export function PassMap(props: {
  readonly neurons: Neurons;
  readonly layer: number;
  readonly onLayer: (layer: number) => void;
  readonly onOpenNeuron: (layer: number, neuron: number) => void;
}) {
  const { neurons, layer, onLayer, onOpenNeuron } = props;
  const captures = neurons.kind === "taken" ? neurons.captures : null;
  const count = captures?.layers.length ?? 0;
  const [webgl] = useState(hasWebGL);

  return (
    <section className="column pass-column" aria-label="Pass">
      <h2>
        Pass{" "}
        <output aria-label="Pass" className="muted">
          {neurons.kind === "none"
            ? neurons.why
            : webgl
              ? `${count} layers`
              : "Not drawn: this browser gives no WebGL 2"}
        </output>
      </h2>
      {captures !== null && webgl && count > 0 && (
        <Layer
          captures={captures}
          layer={Math.max(0, Math.min(layer, count - 1))}
          onLayer={onLayer}
          onOpenNeuron={onOpenNeuron}
        />
      )}
    </section>
  );
}

function Layer(props: {
  readonly captures: Taken;
  readonly layer: number;
  readonly onLayer: (layer: number) => void;
  readonly onOpenNeuron: (layer: number, neuron: number) => void;
}) {
  const { captures, layer, onLayer, onOpenNeuron } = props;
  const last = captures.layers.length - 1;
  // Step 1: once each time the captures change.
  const scales = useMemo(() => passScales(captures), [captures]);
  const spans = useMemo(() => spansOf(captures.input), [captures]);
  // Step 2: each time the captures or the front layer change.
  const pass = useMemo(() => passOf(captures, layer), [captures, layer]);
  const placed = useMemo(() => place(pass), [pass]);
  // The framing and the pointer belong to one layer of one run; a step or
  // a new run puts them back.
  const [framing, setFraming] = useState<{
    readonly pass: Pass;
    readonly stage: number;
  } | null>(null);
  const framed = framing?.pass === pass ? framing.stage : null;
  const [pointing, setPointing] = useState<{
    readonly strips: readonly (Strip | null)[];
    readonly at: Pointed;
  } | null>(null);

  const wrapper = useRef<HTMLDivElement>(null);
  const size = useSize(wrapper);
  // Step 3: the camera, then the strips, each time the framing or the size
  // change.
  const view = useMemo(
    () => (size === null ? null : fit(boxOf(placed, framed), size)),
    [placed, framed, size],
  );
  const strips = useMemo(
    () =>
      view === null || size === null
        ? null
        : drawStrips(placed, captures.words, scales, view, size),
    [placed, captures, scales, view, size],
  );
  useEffect(
    () => () => strips?.forEach((strip) => strip?.texture.dispose()),
    [strips],
  );
  // A reading belongs to the marks it was read from (the header).
  const pointed =
    strips !== null && pointing?.strips === strips ? pointing.at : null;

  const step = (by: number) => onLayer(Math.max(0, Math.min(last, layer + by)));

  return (
    <>
      <div className="pass-head">
        <button
          type="button"
          aria-label="Previous layer"
          disabled={layer === 0}
          onClick={() => step(-1)}
        >
          ‹
        </button>
        <span className="pass-layer">Layer {layer}</span>
        <button
          type="button"
          aria-label="Next layer"
          disabled={layer === last}
          onClick={() => step(1)}
        >
          ›
        </button>
        <span className="muted">
          {pass.full
            ? "Global attention: back to the first token"
            : `Local attention: the last ${COUNT.format(pass.window)} tokens`}
        </span>
        {framed !== null && (
          <button type="button" onClick={() => setFraming(null)}>
            Whole layer
          </button>
        )}
      </div>
      <p className="pass-reading">
        <output aria-label="Pass reading">
          {reading(pass, placed, strips, captures, spans, pointed)}
        </output>
      </p>
      <div
        ref={wrapper}
        className="pass-scene"
        tabIndex={0}
        aria-label={`Layer ${layer}'s pass; ← and → step the layer`}
        onKeyDown={(event) => {
          const by =
            event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
          if (by === 0) return;
          event.preventDefault();
          step(by);
        }}
      >
        {view !== null && strips !== null && size !== null && (
          <>
            <Canvas
              flat
              frameloop="demand"
              dpr={[1, 2]}
              camera={view}
              aria-hidden
            >
              <Scene
                view={view}
                placed={placed}
                strips={strips}
                onPoint={(at) => {
                  if (same(at, pointed)) return;
                  setPointing(at === null ? null : { strips, at });
                }}
                onClick={(hit) => {
                  // The mark the reading names: a click's position is
                  // rounded to a whole CSS pixel, a move's is not, and a
                  // mark can be half a pixel, so the click's own hit may
                  // be the next mark.
                  const at = pointed?.stage === hit.stage ? pointed : hit;
                  const stage = placed[at.stage]?.stage;
                  const strip = strips[at.stage];
                  const row = strip?.marks[at.row];
                  const index = row?.[at.mark];
                  if (stage?.id === "neurons" && index !== undefined) {
                    onOpenNeuron(layer, index);
                  } else {
                    setFraming({ pass, stage: at.stage });
                  }
                }}
              />
            </Canvas>
            <Labels
              pass={pass}
              placed={placed}
              view={view}
              size={size}
              position={captures.position}
              spans={spans}
            />
          </>
        )}
      </div>
      <Legend pass={pass} scales={scales} />
    </>
  );
}

// The legend under the scene (the header): the ramp, each captured
// stage's scale, and how to read a mark.
function Legend(props: {
  readonly pass: Pass;
  readonly scales: PassScales;
}): ReactNode {
  const { pass, scales } = props;
  const rgb = (c: readonly number[]) => `rgb(${c.join(",")})`;
  const shown = pass.stages.flatMap(({ id, name, captured }) =>
    isCaptured(id) && captured ? [{ id, name }] : [],
  );
  return (
    <>
      <div className="legend">
        <span>0</span>
        <span
          className="ramp"
          style={{
            background: `linear-gradient(90deg, ${DECK_PALETTE.heat.map(rgb).join(", ")})`,
          }}
        />
        <span className="muted">full at</span>
        {shown.map((stage, i) => (
          <span key={stage.id} className="pass-scale">
            {i > 0 && "· "}
            {stage.name} <b>{significant(scales[stage.id])}</b>
          </span>
        ))}
      </div>
      <p className="legend muted">
        size, either sign · one scale a stage across the layers, the strongest
        1% at full; attention's weights full at 1 · each mark the strongest of
        its group, sized so every mark has a pixel · outlined: not captured
      </p>
    </>
  );
}

// The scene: the camera from `view`, a plane a stage (filled where
// captured, a hit target where not) and the rows' outlines.
function Scene(props: {
  readonly view: OrthographicCamera;
  readonly placed: readonly Placed[];
  readonly strips: readonly (Strip | null)[];
  readonly onPoint: (at: Pointed | null) => void;
  readonly onClick: (at: Pointed) => void;
}) {
  const { view, placed, strips, onPoint, onClick } = props;
  const set = useThree((s) => s.set);
  const invalidate = useThree((s) => s.invalidate);
  useLayoutEffect(() => {
    set({ camera: view });
    invalidate();
  }, [view, set, invalidate]);
  useEffect(() => invalidate(), [strips, invalidate]);
  const outlines = useMemo(() => outlinesOf(placed), [placed]);
  useEffect(
    () => () => {
      outlines.captured.dispose();
      outlines.not.dispose();
    },
    [outlines],
  );

  // The mark under a hit: the texture's u across the rows, v along the
  // width (step 5).
  const at = (s: number, event: ThreeEvent<PointerEvent | MouseEvent>) => {
    const p = placed[s];
    const uv = event.uv;
    if (p === undefined || uv === undefined) return null;
    const rows = p.stage.rows.length;
    const marks = strips[s]?.count ?? 1;
    return {
      stage: s,
      row: Math.min(rows - 1, Math.max(0, Math.floor(uv.x * rows))),
      mark: Math.min(marks - 1, Math.max(0, Math.floor(uv.y * marks))),
    };
  };

  return (
    <>
      {placed.map((p, s) => {
        const strip = strips[s] ?? null;
        const depth = p.x1 - p.x0;
        return (
          <mesh
            key={p.stage.id}
            rotation-x={-Math.PI / 2}
            position={[p.x0 + depth / 2, 0, -p.length / 2]}
            onPointerMove={(event) => {
              event.stopPropagation();
              onPoint(at(s, event));
            }}
            onPointerOut={() => onPoint(null)}
            onClick={(event) => {
              event.stopPropagation();
              if (event.delta > CLICK) return;
              const where = at(s, event);
              if (where !== null) onClick(where);
            }}
          >
            <planeGeometry args={[depth, p.length]} />
            {strip === null ? (
              <meshBasicMaterial transparent opacity={0} depthWrite={false} />
            ) : (
              <meshBasicMaterial map={strip.texture} toneMapped={false} />
            )}
          </mesh>
        );
      })}
      <lineSegments geometry={outlines.captured}>
        <lineBasicMaterial color={PASS_EDGE} toneMapped={false} />
      </lineSegments>
      <lineSegments geometry={outlines.not}>
        <lineBasicMaterial color={PASS_OUTLINE} toneMapped={false} />
      </lineSegments>
    </>
  );
}

// The stage labels under each stage, and attention's tokens beside it
// where the keys have room (step 4), placed by projection.
function Labels(props: {
  readonly pass: Pass;
  readonly placed: readonly Placed[];
  readonly view: OrthographicCamera;
  readonly size: Size;
  readonly position: number;
  readonly spans: readonly Span[];
}) {
  const { pass, placed, view, size, position, spans } = props;
  const screen = (x: number, z: number) => {
    const v = new Vector3(x, 0, z).project(view);
    return {
      left: ((v.x + 1) / 2) * size.width,
      top: ((1 - v.y) / 2) * size.height,
    };
  };
  const attention = placed.find((p) => p.stage.id === "attention");
  const keys: ReactNode[] = [];
  if (attention !== undefined) {
    const a = screen(attention.x1, 0);
    const b = screen(attention.x1, -KEY);
    if (Math.abs(b.top - a.top) >= LINE) {
      const from = pass.firstKey;
      const to = from + attention.stage.width; // past the last key
      // The spans that reach the window, from the one holding its first key.
      const start = spans.indexOf(spanAt(spans, from));
      for (const span of spans.slice(start)) {
        if (span.first >= to) break;
        const first = Math.max(span.first, from);
        const end = Math.min(span.first + span.count, to);
        const middle = (first + end) / 2 - from;
        const at = screen(attention.x1 + 0.08, -middle * KEY);
        keys.push(
          <div
            key={span.first}
            className={
              span.shown.kind === "control" ? "pass-key control" : "pass-key"
            }
            style={{ left: at.left, top: at.top }}
          >
            {labelOf(span)}
          </div>,
        );
      }
    }
  }
  return (
    <div className="pass-labels" aria-hidden>
      {placed.map((p, s) => {
        // Under the near edge by pixels, not scene units, so a framed
        // stage's label stays in the room kept for it (LABELS).
        const at = screen((p.x0 + p.x1) / 2, 0);
        return (
          <div
            key={p.stage.id}
            className={p.stage.captured ? "pass-label" : "pass-label not"}
            style={{ left: at.left, top: at.top + 8 + (s % 3) * 30 }}
          >
            <b>{p.stage.name}</b>
            <br />
            {shapeOf(p.stage, position)}
          </div>
        );
      })}
      {keys}
    </div>
  );
}

// "{rows} × {width}", "keys" and "of {n} tokens" for attention, "not
// captured" where so.
function shapeOf(stage: Stage, position: number): string {
  const rows = stage.rows.length;
  let text =
    rows === 1
      ? COUNT.format(stage.width)
      : `${rows} × ${COUNT.format(stage.width)}`;
  if (stage.id === "attention") {
    text += " keys";
    if (stage.width < position + 1) {
      text += ` of ${COUNT.format(position + 1)} tokens`;
    }
  }
  return stage.captured ? text : `${text} · not captured`;
}

// Step 2: each stage's place along the flow and its length along the
// depth.
function place(pass: Pass): readonly Placed[] {
  let x = 0;
  return pass.stages.map((stage) => {
    const x0 = x;
    const x1 = x0 + stage.rows.length * ROW;
    x = x1 + GAP;
    const unit = stage.unit === "key" ? KEY : COL;
    return { stage, x0, x1, length: stage.width * unit };
  });
}

// The framed box: every stage, or the one framed.
function boxOf(placed: readonly Placed[], framed: number | null): Box3 {
  const box = new Box3();
  const chosen =
    framed === null ? placed : placed.filter((_, s) => s === framed);
  for (const p of chosen) {
    box.expandByPoint(new Vector3(p.x0, 0, 0));
    box.expandByPoint(new Vector3(p.x1, 0, -p.length));
  }
  // Room for the whole layer's end labels, wider than their stages; the
  // room under the box is kept in pixels (fit, LABELS).
  if (framed === null) {
    box.min.x -= GAP;
    box.max.x += GAP;
  }
  return box;
}

// Step 3: an orthographic camera along ISO, beyond the box, its frustum
// the scene's CSS pixels, zoomed so the box fills the scene above LABELS
// pixels with room, and aimed so the box sits in that part. Marked
// manual, so react-three-fiber leaves its frustum as set here.
function fit(box: Box3, size: Size): OrthographicCamera {
  const { width, height } = size;
  const camera = new OrthographicCamera(
    -width / 2,
    width / 2,
    height / 2,
    -height / 2,
    0.1,
    1000,
  );
  (camera as OrthographicCamera & { manual: boolean }).manual = true;
  // Beyond every point of the box along ISO, so none falls behind the
  // camera, however long attention is.
  const reach = box.getSize(new Vector3()).length() + 10;
  camera.far = 2 * reach;
  const target = box.getCenter(new Vector3());
  camera.position.copy(target).addScaledVector(ISO, reach);
  camera.lookAt(target);
  camera.updateMatrixWorld();
  // The box's extent across and up the screen, in scene units.
  const across = new Box3();
  for (const x of [box.min.x, box.max.x]) {
    for (const z of [box.min.z, box.max.z]) {
      across.expandByPoint(
        new Vector3(x, 0, z).applyMatrix4(camera.matrixWorldInverse),
      );
    }
  }
  const spanX = Math.max(across.max.x - across.min.x, 1e-6);
  const spanY = Math.max(across.max.y - across.min.y, 1e-6);
  const above = Math.max(1, height - LABELS);
  camera.zoom = Math.min(width / (spanX * PADDING), above / (spanY * PADDING));
  // Aim LABELS / 2 pixels below the box's centre, so the box is centred in
  // the part above the labels.
  const up = new Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
  camera.position.addScaledVector(up, -LABELS / 2 / camera.zoom);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  return camera;
}

// Step 3: each captured stage's groups and texture, from the physical
// pixels its width spans on screen.
function drawStrips(
  placed: readonly Placed[],
  words: Float32Array,
  scales: PassScales,
  view: OrthographicCamera,
  size: Size,
): readonly (Strip | null)[] {
  return placed.map(({ stage, x0, x1, length }) => {
    if (!isCaptured(stage.id) || !stage.captured) return null;
    const fullAt = scales[stage.id];
    const middle = (x0 + x1) / 2;
    const a = new Vector3(middle, 0, 0).project(view);
    const b = new Vector3(middle, 0, -length).project(view);
    // Along the screen's major axis, as rasterization samples a line.
    const css = Math.max(
      Math.abs(((b.x - a.x) / 2) * size.width),
      Math.abs(((b.y - a.y) / 2) * size.height),
    );
    const pixels = Math.max(1, Math.floor(css * size.dpr));
    const group = groupFor(stage.width, pixels);
    const rows = stage.rows.length;
    const count = Math.ceil(stage.width / group);
    const data = new Uint8Array(rows * count * 4);
    const marks = stage.rows.map((row, r) => {
      const first = row.first ?? 0;
      const values = words.subarray(first, first + stage.width);
      const chosen = strongestOfGroups(values, group);
      const painted = paintStrip(values, chosen, fullAt, DECK_PALETTE);
      for (let m = 0; m < count; m++) {
        data.set(painted.subarray(m * 4, m * 4 + 4), (m * rows + r) * 4);
      }
      return chosen;
    });
    const texture = new DataTexture(data, rows, count, RGBAFormat);
    texture.magFilter = NearestFilter;
    texture.minFilter = NearestFilter;
    texture.colorSpace = SRGBColorSpace;
    texture.needsUpdate = true;
    return { group, marks, count, texture };
  });
}

// The stages that hold captured values, and so have a scale.
function isCaptured(id: StageId): id is CapturedId {
  return (
    id === "residual" ||
    id === "query" ||
    id === "attention" ||
    id === "attentionAdded" ||
    id === "neurons" ||
    id === "mlpAdded"
  );
}

// Each row's rectangle, captured stages' and the others' apart.
function outlinesOf(placed: readonly Placed[]): {
  readonly captured: BufferGeometry;
  readonly not: BufferGeometry;
} {
  const captured: number[] = [];
  const not: number[] = [];
  for (const { stage, x0, length } of placed) {
    const into = stage.captured ? captured : not;
    stage.rows.forEach((_, r) => {
      const a = x0 + r * ROW;
      const b = a + ROW;
      const z = -length;
      into.push(a, 0, 0, b, 0, 0, b, 0, 0, b, 0, z);
      into.push(b, 0, z, a, 0, z, a, 0, z, a, 0, 0);
    });
  }
  const geometry = (points: number[]) => {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(points, 3));
    return g;
  };
  return { captured: geometry(captured), not: geometry(not) };
}

// Step 5: the reading of what the pointer is over.
function reading(
  pass: Pass,
  placed: readonly Placed[],
  strips: readonly (Strip | null)[] | null,
  captures: Taken,
  spans: readonly Span[],
  pointed: Pointed | null,
): ReactNode {
  if (pointed === null) return "Point at a mark to read it";
  const p = placed[pointed.stage];
  if (p === undefined) return "Point at a mark to read it";
  const { stage } = p;
  const head = `Layer ${pass.layer} · ${stage.name}`;
  const strip = strips?.[pointed.stage] ?? null;
  const row = stage.rows[pointed.row];
  const index = strip?.marks[pointed.row]?.[pointed.mark];
  if (
    strip === null ||
    row === undefined ||
    row.first === null ||
    index === undefined
  ) {
    return `${head} · not captured`;
  }
  const value = captures.words[row.first + index] ?? Number.NaN;
  const a = strip.group * pointed.mark;
  const b = Math.min(a + strip.group, stage.width) - 1;
  if (stage.id === "attention") {
    const position = pass.firstKey + index;
    const said = readingOf(spanAt(spans, position), position);
    return (
      <>
        {`${head} · ${row.label} · `}
        <data value={String(value)}>{value.toFixed(2)}</data>
        {` on ${said} · position ${position}`}
        {strip.group > 1 &&
          ` · strongest of positions ${pass.firstKey + a}–${pass.firstKey + b}`}
      </>
    );
  }
  const kind = stage.unit === "neuron" ? "neuron" : "dimension";
  const label = row.label === "" ? "" : ` · ${row.label}`;
  return (
    <>
      {`${head}${label} · ${kind} ${index} of ${COUNT.format(stage.width)} · `}
      <data value={String(value)}>{signed(value)}</data>
      {strip.group > 1 && ` · strongest of ${a}–${b}`}
    </>
  );
}

// The scene's CSS size and the canvas's pixel ratio (react-three-fiber's
// dpr [1, 2]), measured; null before the first measure or at no size.
function useSize(element: RefObject<HTMLDivElement | null>): Size | null {
  const [size, setSize] = useState<Size | null>(null);
  useEffect(() => {
    const target = element.current;
    if (target === null) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry === undefined) return;
      const { width, height } = entry.contentRect;
      const dpr = Math.min(2, Math.max(1, window.devicePixelRatio));
      setSize((old) =>
        width < 1 || height < 1
          ? null
          : old !== null &&
              old.width === width &&
              old.height === height &&
              old.dpr === dpr
            ? old
            : { width, height, dpr },
      );
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [element]);
  return size;
}

// Whether this browser gives WebGL 2, as the deck asks (Deck.tsx).
function hasWebGL(): boolean {
  const context = document.createElement("canvas").getContext("webgl2");
  if (context === null) return false;
  context.getExtension("WEBGL_lose_context")?.loseContext();
  return true;
}

// Whether two pointings name the same mark.
function same(a: Pointed | null, b: Pointed | null): boolean {
  if (a === null || b === null) return a === b;
  return a.stage === b.stage && a.row === b.row && a.mark === b.mark;
}
