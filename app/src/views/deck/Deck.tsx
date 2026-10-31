// Axis H (view): the layer deck, the workbench's centre column. Every
// layer's neurons at the branch point, as one side used them (cards.ts,
// neuronsOf): one card per layer, its neurons in a grid you point at, click and
// drag across, zoom and pan, the front card sliding away to show the next
// (docs/design/concepts/deck.html, the reviewed concept).
//
// Contract
//   - Deck({neurons, cards, faces, previewing, layer, onLayer, pins,
//     selection, onSelect, autoFocus}): `neurons` the side shown
//     (cards.ts), `cards` its layers, null when it has none; `faces` the pixels each card
//     shows (cards.ts, facesOf); `previewing` whether those show what the
//     pins will send (the reply's cards, not the steered reply's); `layer` is the front
//     card's number. With autoFocus, the deck takes the focus once it has
//     cards.
//   - Without cards: status "Deck" says why, in the page's words (cards.ts,
//     neuronsOf), and nothing is drawn.
//   - With cards: status "Deck" reads "{n} layers". Each card shows "Layer
//     {l}" and, when any pin is held in it, "● {k} pinned"; its neurons in
//     the grid fold.ts gives, its face (a heat map of each neuron's size
//     against the layer's own scale, paint.ts), with a gap between cells;
//     an outline round the neurons pinned (session.pins) and round the
//     selection, in colours the heat never takes (theme/palette.ts), and a
//     light tint over a selection of more than one.
//   - Pointing at a neuron on the front card shows a tip: "#{n} {value}",
//     the value with its sign and two places, and for a pinned one
//     " · pinned", or, while previewing, " · pinned → sends {v}" (≈ before
//     a scale's when an earlier layer is pinned too: the run's own value
//     there may differ from this card's).
//   - Selecting, on the front card once it has settled (onSelect, one
//     layer's neurons): a click selects that neuron alone; Shift+click
//     adds it to the selection in this layer, or takes it out. A drag of
//     more than 4 pixels from anywhere on the card's face (its grid,
//     margins or header) selects the box of cells from there to the
//     pointer, clipped to the grid (fold.ts, neuronsIn), shown while
//     dragging, chosen on release; with Shift, added to the selection in
//     this layer. A box that holds no neuron changes nothing.
//   - Zoom: Ctrl or ⌘ with the wheel, a trackpad's pinch, + and − with
//     the deck focused, or group "Zoom": buttons "Zoom out", "Zoom in",
//     "Fit", and status "Zoom level" ("×{z}" to one place). From ×1, the
//     fitted view, to ×24; the wheel and pinch zoom about the pointer, the
//     rest about the centre; 0 or Fit go back to ×1. Zoomed, Space+drag or
//     a middle-button drag pans, never past the fitted view's edges. The
//     zoom stays as the layers change.
//   - Moving: a strip of buttons "Layer 0" … "Layer {n-1}" (the front one
//     pressed, a dot on each with a pin held); with the deck focused, the
//     arrow keys and Page Up / Page Down; the wheel over the deck (without
//     Ctrl or ⌘), one layer per notch-sized turn. Never past either end.
//   - The legend: the heat ramp from 0 to s, s = the front layer's scale
//     to two significant digits (shape/numbers.ts), "size of the push,
//     either sign · each layer on its own scale, the strongest 1% at full",
//     the outlines' meaning (", showing what it will send" after "pinned"
//     while previewing with pins held), and a line on the gestures.
//   - Motion: the front card slides away to the left and the next comes
//     forward, settling in about half a second at any frame rate; under
//     prefers-reduced-motion the deck moves at once, with no slide.
//   - No WebGL 2: status "Deck" reads "The deck needs WebGL, which this
//     browser did not give", and nothing else is drawn.
//
// Algorithm
//   1. Each card: a face (the card colour, a rule round it), a header
//      strip (a canvas texture: the layer, its pins), and the grid: a plane
//      drawn by one shader over two textures, the painted pixels (one texel
//      a neuron) and the marks (one byte a neuron: 0 none, 1 pinned, 2
//      selected, 3 both).
//   2. The shader, per fragment: the cell under it (p = uv × grid, row 0 at
//      the top); its colour from the pixels at the cell's centre, with
//      gradients from the plane's own uv so far cards take their mipmaps;
//      a gap one screen pixel wide at the cell's edges, faded out where
//      cells are smaller than about four pixels; rings about two pixels
//      wide just outside a marked cell (the selected one innermost, a pin's
//      round it), so they show at any cell size; all times the card's
//      dimming and opacity.
//   3. Placement for d = l - position, position easing toward the front:
//      d ≥ 0 recedes up, right and back; 0 > d > -1 slides out to the left,
//      turning; d ≤ -1 is a thin stack at the left edge. Cards with d ≥ 12
//      are not drawn.
//   4. Pointing: listeners on the canvas itself. A ray from the camera
//      through the pointer meets the front card's plane, in the card's
//      own frame, which gives the cell in grid units, unclipped so a drag
//      may leave the card; fold.ts's neuronAt gives the neuron. The front
//      card only, once it has settled.
//   5. Zoom: the fitted view is drawn z times its size and the canvas
//      shows a window of it (the camera's view offset), so picking, the
//      cell gaps and the outlines work unchanged at any zoom.
//
// Design and cost
//   - three.js through react-three-fiber, the WebGL renderer on the page,
//     apart from the engine's WebGPU device in the worker. three.js is
//     imported only in views/ (scripts/boundaries.py).
//   - Rendered on demand: a frame only while a card moves or something
//     shown changes, never continuously.
//   - Textures: per card 27,648 bytes of pixels and 6,912 of marks, about
//     900 KiB for 26 cards, uploaded when the faces change (the captures
//     or, previewing, the pins) or the marks do (the pins or selection);
//     headers redrawn when the pins change.
//   - Zoom as a window on the fitted view, not a camera moved nearer the
//     card: the deck behind keeps its place, the view survives a resize
//     (it is kept as fractions), and the ray through the pointer is the
//     camera's own, offset included.
//   - Colours are passed to the shader as the sRGB bytes they are, and
//     written out unconverted, as the page's CSS colours are (no tone
//     mapping: Canvas flat).

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  CanvasTexture,
  DataTexture,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  LinearFilter,
  LinearMipmapLinearFilter,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  NearestFilter,
  PerspectiveCamera,
  PlaneGeometry,
  RGBAFormat,
  Raycaster,
  RedFormat,
  ShaderMaterial,
  SRGBColorSpace,
  Vector2,
  Vector3,
  type WebGLRenderer,
} from "three";

import type { Selection } from "../../App.tsx";
import type { Edit } from "../../engine/edits.ts";
import { sentBy } from "../../session/pins.ts";
import { neuronAt, neuronsIn } from "../../shape/fold.ts";
import { signed, significant } from "../../shape/numbers.ts";
import {
  CARD,
  CARD_RULE,
  DECK_PALETTE,
  RING_PIN,
  RING_SELECTED,
} from "../../theme/palette.ts";
import type { Card, Neurons } from "./cards.ts";

// The scene's units: a card's grid is GRID_WIDTH wide, its height the
// grid's proportion of that, with a header strip above it and a margin of
// card face round both.
const GRID_WIDTH = 4;
const HEADER = 0.28;
const MARGIN = 0.12;
const DRAWN_AHEAD = 12;
const SETTLE_SECONDS = 0.09;
const WHEEL_STEP = 70;
const LINE_PIXELS = 40;
const DRAG_PIXELS = 4;
const ZOOM_STEP = 1.5;
const ZOOM_MOST = 24;
const ZOOM_PER_PIXEL = 0.004; // a mouse's notch, about 100 pixels: ×1.5
const SELECTION_TINT = 0.22;
const PAGE = "#0d1422";
const INK = "#eee6d3";
const PIN = "#ffcf73";

// With the deck focused: + and = zoom in, − out, 0 back to the fit.
const ZOOM_KEYS: Readonly<Record<string, number>> = {
  "+": ZOOM_STEP,
  "=": ZOOM_STEP,
  "-": 1 / ZOOM_STEP,
  "0": 0,
};

const KEYS: Readonly<Record<string, number>> = {
  ArrowDown: 1,
  ArrowRight: 1,
  PageDown: 1,
  ArrowUp: -1,
  ArrowLeft: -1,
  PageUp: -1,
};

interface Tip {
  readonly neuron: number;
  readonly x: number;
  readonly y: number;
}

// The view's zoom: z times the fitted view, showing the window whose top
// left is (x, y), as fractions of the fitted view, so it survives a resize.
interface View {
  readonly z: number;
  readonly x: number;
  readonly y: number;
}

const FIT: View = { z: 1, x: 0, y: 0 };

// A cell under the pointer, in grid units, not rounded and not clipped: a
// drag may leave the card.
interface At {
  readonly column: number;
  readonly row: number;
}

export function Deck(props: {
  readonly neurons: Neurons;
  readonly cards: readonly Card[] | null;
  readonly faces: readonly Uint8Array[] | null;
  readonly previewing: boolean;
  readonly layer: number;
  readonly onLayer: (layer: number) => void;
  readonly pins: readonly Edit[];
  readonly selection: Selection | null;
  readonly onSelect: (chosen: Selection) => void;
  readonly autoFocus?: boolean;
}) {
  const { neurons, cards, faces, previewing, layer, onLayer } = props;
  const { pins, selection, onSelect, autoFocus = false } = props;
  const [webgl] = useState(hasWebGL);
  const reduced = useReducedMotion();
  const [tip, setTip] = useState<Tip | null>(null);
  const [view, setView] = useState<View>(FIT);
  // The box being dragged, shown before it is let go.
  const [dragged, setDragged] = useState<Selection | null>(null);
  const [panning, setPanning] = useState(false);
  const space = useRef(false);
  const host = useRef<HTMLDivElement>(null);
  const count = cards?.length ?? 0;
  const step = (by: number) => {
    setTip(null);
    onLayer(Math.max(0, Math.min(count - 1, layer + by)));
  };

  // Zoom by `factor` about the point (fx, fy) of the deck, as fractions of
  // its width and height: that point stays where it is.
  const zoomAt = (factor: number, fx: number, fy: number) =>
    setView((v) => {
      const z = Math.max(1, Math.min(ZOOM_MOST, v.z * factor));
      if (z === 1) return FIT;
      const room = 1 - 1 / z;
      return {
        z,
        x: clamp(v.x + fx / v.z - fx / z, 0, room),
        y: clamp(v.y + fy / v.z - fy / z, 0, room),
      };
    });
  const pan = (dx: number, dy: number) => {
    const box = host.current?.getBoundingClientRect();
    if (box === undefined) return;
    setView((v) => {
      if (v.z === 1) return v;
      const room = 1 - 1 / v.z;
      return {
        z: v.z,
        x: clamp(v.x - dx / (box.width * v.z), 0, room),
        y: clamp(v.y - dy / (box.height * v.z), 0, room),
      };
    });
  };

  // The wheel steps layers, or zooms at the pointer with Ctrl or ⌘ held
  // (a trackpad's pinch arrives so), so the page must not scroll under it:
  // a listener that is not passive, which React's onWheel cannot be.
  const wheel = useRef({ turned: 0, step, zoomAt });
  wheel.current.step = step;
  wheel.current.zoomAt = zoomAt;
  useEffect(() => {
    const element = host.current;
    if (element === null || cards === null) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const w = wheel.current;
      const delta =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? event.deltaY * LINE_PIXELS
          : event.deltaY;
      if (event.ctrlKey || event.metaKey) {
        const box = element.getBoundingClientRect();
        w.zoomAt(
          Math.exp(-delta * ZOOM_PER_PIXEL),
          (event.clientX - box.left) / box.width,
          (event.clientY - box.top) / box.height,
        );
        return;
      }
      w.turned += delta;
      if (Math.abs(w.turned) >= WHEEL_STEP) {
        w.step(Math.sign(w.turned));
        w.turned = 0;
      }
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [cards]);

  // The focus, once there are cards to work on (Workbench.tsx).
  const hasCards = cards !== null;
  useEffect(() => {
    if (autoFocus && hasCards) host.current?.focus({ preventScroll: true });
  }, [autoFocus, hasCards]);

  const why = webgl
    ? neurons.kind === "none"
      ? neurons.why
      : null
    : "The deck needs WebGL, which this browser did not give";
  const card = cards?.[layer];
  const pinOf = (neuron: number) =>
    pins.find((p) => p.layer === layer && p.neuron === neuron);
  const shown = dragged ?? selection;

  // A click on a neuron: it alone, or with Shift, in or out of the
  // selection of its layer.
  const pick = (neuron: number, adding: boolean) => {
    if (!adding || selection === null || selection.layer !== layer) {
      onSelect({ layer, neurons: [neuron] });
      return;
    }
    const rest = selection.neurons.filter((n) => n !== neuron);
    if (rest.length === selection.neurons.length) {
      onSelect({ layer, neurons: [...rest, neuron].sort((a, b) => a - b) });
    } else if (rest.length > 0) {
      onSelect({ layer, neurons: rest });
    }
  };
  // A box: its neurons, or with Shift, added to the selection of its layer.
  const boxed = (a: At, b: At, adding: boolean, done: boolean) => {
    if (card === undefined) return;
    const inBox = neuronsIn(card.grid, a, b);
    const base =
      adding && selection !== null && selection.layer === layer
        ? selection.neurons
        : [];
    const neurons = [...new Set([...base, ...inBox])].sort((x, y) => x - y);
    const chosen = neurons.length === 0 ? null : { layer, neurons };
    if (!done) {
      setDragged(chosen);
      return;
    }
    setDragged(null);
    if (chosen !== null) onSelect(chosen);
  };

  return (
    <section className="column deck-column" aria-label="Layer deck">
      <h2>
        Layer deck{" "}
        <output aria-label="Deck" className="muted">
          {why ?? `${count} layers`}
        </output>
      </h2>
      <div
        ref={host}
        className={panning ? "deck panning" : "deck"}
        tabIndex={cards === null ? -1 : 0}
        role="group"
        aria-label="Deck cards"
        onKeyDown={(event) => {
          if (cards === null) return;
          if (event.key === " ") {
            event.preventDefault();
            space.current = true;
            return;
          }
          const zoom = ZOOM_KEYS[event.key];
          if (zoom !== undefined) {
            event.preventDefault();
            if (zoom === 0) setView(FIT);
            else zoomAt(zoom, 0.5, 0.5);
            return;
          }
          const by = KEYS[event.key];
          if (by === undefined) return;
          event.preventDefault();
          step(by);
        }}
        onKeyUp={(event) => {
          if (event.key === " ") space.current = false;
        }}
        onBlur={() => {
          space.current = false;
        }}
      >
        {webgl && cards !== null && faces !== null ? (
          <Canvas
            flat
            frameloop="demand"
            dpr={[1, 2]}
            camera={{ fov: 30, near: 0.1, far: 100 }}
            aria-hidden
          >
            <color attach="background" args={[PAGE]} />
            <Scene
              cards={cards}
              faces={faces}
              front={layer}
              reduced={reduced}
              pins={pins}
              selection={shown}
              view={view}
              space={space}
              onHover={setTip}
              onPick={pick}
              onBox={boxed}
              onPan={pan}
              onPanning={setPanning}
            />
          </Canvas>
        ) : (
          <p className="deck-empty">
            {why === "No run yet"
              ? "Run a prompt to see every layer's neurons here."
              : why}
          </p>
        )}
        {cards !== null && (
          <div className="zoom" role="group" aria-label="Zoom">
            <button
              type="button"
              aria-label="Zoom out"
              disabled={view.z === 1}
              onClick={() => zoomAt(1 / ZOOM_STEP, 0.5, 0.5)}
            >
              −
            </button>
            <output aria-label="Zoom level">×{view.z.toFixed(1)}</output>
            <button
              type="button"
              aria-label="Zoom in"
              disabled={view.z === ZOOM_MOST}
              onClick={() => zoomAt(ZOOM_STEP, 0.5, 0.5)}
            >
              +
            </button>
            <button
              type="button"
              aria-label="Fit"
              disabled={view.z === 1}
              onClick={() => setView(FIT)}
            >
              Fit
            </button>
          </div>
        )}
        {tip !== null && card !== undefined && (
          <div className="tip" style={{ left: tip.x + 14, top: tip.y + 10 }}>
            #{tip.neuron} {signed(card.values[tip.neuron] ?? Number.NaN)}
            {tipPin(
              pinOf(tip.neuron),
              previewing,
              card.values[tip.neuron] ?? Number.NaN,
              pins.some((p) => p.layer < layer),
            )}
          </div>
        )}
      </div>
      {cards !== null && (
        <>
          <div className="strip" role="group" aria-label="Layers">
            {cards.map((_, l) => (
              <button
                key={l}
                type="button"
                aria-label={`Layer ${l}`}
                aria-pressed={l === layer}
                className={
                  pins.some((p) => p.layer === l) ? "pinned" : undefined
                }
                onClick={() => {
                  setTip(null);
                  onLayer(l);
                }}
              >
                {l}
              </button>
            ))}
          </div>
          {card !== undefined && (
            <Legend
              scale={card.painted.fullAt}
              previewing={previewing && pins.length > 0}
            />
          )}
        </>
      )}
    </section>
  );
}

// What the tip adds for a pinned neuron: on the reply's card, the
// value it will send (≈ for a scale when an earlier layer is pinned too,
// as the run's own value there may differ from this card's).
function tipPin(
  pin: Edit | undefined,
  previewing: boolean,
  used: number,
  earlierPinned: boolean,
): string {
  if (pin === undefined) return "";
  if (!previewing) return " · pinned";
  const about = pin.kind === "scale" && earlierPinned ? "≈ " : "";
  return ` · pinned → sends ${about}${signed(sentBy(pin, used))}`;
}

function Legend(props: {
  readonly scale: number;
  readonly previewing: boolean;
}): ReactNode {
  const { scale, previewing } = props;
  const rgb = (c: readonly number[]) => `rgb(${c.join(",")})`;
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
        <span>{significant(scale)}</span>
        <span className="muted">
          size of the push, either sign · each layer on its own scale, the
          strongest 1% at full ·{" "}
          <span style={{ color: RING_PIN }}>
            ▢ pinned{previewing ? ", showing what it will send" : ""}
          </span>{" "}
          · <span style={{ color: RING_SELECTED }}>▢ selected</span>
        </span>
      </div>
      <p className="legend muted">
        Drag to select several, Shift to add · ⌘ or Ctrl + scroll, or pinch, to
        zoom · Space + drag to pan
      </p>
    </>
  );
}

// The cards in the scene, placed by the front layer, sliding toward it.
function Scene(props: {
  readonly cards: readonly Card[];
  readonly faces: readonly Uint8Array[];
  readonly front: number;
  readonly reduced: boolean;
  readonly pins: readonly Edit[];
  readonly selection: Selection | null;
  readonly view: View;
  readonly space: { readonly current: boolean };
  readonly onHover: (tip: Tip | null) => void;
  readonly onPick: (neuron: number, adding: boolean) => void;
  readonly onBox: (a: At, b: At, adding: boolean, done: boolean) => void;
  readonly onPan: (dx: number, dy: number) => void;
  readonly onPanning: (panning: boolean) => void;
}) {
  const { cards, faces, front, reduced, pins, selection, view } = props;
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const built = useMemo(() => cards.map((c) => build(c, gl)), [cards, gl]);
  const position = useRef(front);
  const fonts = useFontsReady();

  useEffect(() => () => built.forEach(dispose), [built]);
  useEffect(() => invalidate(), [front, built, reduced, invalidate]);

  // The faces follow the pins (cards.ts, facesOf).
  useEffect(() => {
    built.forEach((b, l) => {
      const face = faces[l];
      if (face === undefined || b.pixels.image.data === face) return;
      b.pixels.image.data = face;
      b.pixels.needsUpdate = true;
    });
    invalidate();
  }, [built, faces, invalidate]);

  // Marks and headers follow the pins and the selection.
  useEffect(() => {
    built.forEach((b, l) => {
      const data = b.markBytes;
      data.fill(0);
      for (const p of pins) {
        if (p.layer === l && p.neuron < b.width) data[p.neuron] = 1;
      }
      if (selection !== null && selection.layer === l) {
        for (const n of selection.neurons) {
          if (n < b.width) data[n] = (data[n] ?? 0) | 2;
        }
      }
      b.marks.needsUpdate = true;
      (b.grid.uniforms["tint"] as { value: number }).value =
        selection !== null &&
        selection.layer === l &&
        selection.neurons.length > 1
          ? SELECTION_TINT
          : 0;
      drawHeader(b, l, pins.filter((p) => p.layer === l).length);
    });
    invalidate();
  }, [built, pins, selection, fonts, invalidate]);

  // Fit the front card, with room for the deck behind it; then the zoom,
  // as a window on that fitted view.
  useLayoutEffect(() => {
    if (!(camera instanceof PerspectiveCamera)) return;
    const height = Math.max(...built.map((b) => b.faceHeight));
    const width = GRID_WIDTH + 2 * MARGIN;
    const tan = Math.tan((camera.fov * Math.PI) / 360);
    const aspect = size.width / Math.max(1, size.height);
    const distance = Math.max(
      (height * 1.42) / (2 * tan),
      (width * 1.3) / (2 * tan * aspect),
    );
    // Aimed up and right of the front card's centre, where the deck
    // behind it recedes, with the front card whole below.
    camera.position.set(0.12, 0.25, distance);
    camera.lookAt(0.6, 0.3, -1.2);
    if (view.z === 1) {
      camera.clearViewOffset();
    } else {
      const w = size.width * view.z;
      const h = size.height * view.z;
      camera.setViewOffset(
        w,
        h,
        view.x * w,
        view.y * h,
        size.width,
        size.height,
      );
    }
    camera.updateProjectionMatrix();
    invalidate();
  }, [camera, size, built, view, invalidate]);

  useFrame((_, delta) => {
    const gap = front - position.current;
    position.current =
      reduced || Math.abs(gap) < 0.001
        ? front
        : position.current +
          gap * (1 - Math.exp(-Math.min(delta, 0.5) / SETTLE_SECONDS));
    built.forEach((b, l) => place(b, l - position.current));
    if (position.current !== front) invalidate();
  });

  // The pointer, on the canvas itself: hovering, clicking and dragging on
  // the front card once it has settled, and panning.
  const latest = useRef(props);
  latest.current = props;
  useEffect(() => {
    const element = gl.domElement;
    const ray = new Raycaster();
    const ndc = new Vector2();
    const inverse = new Matrix4();
    // The front card's cell under a point of the page, or null.
    const at = (clientX: number, clientY: number): At | null => {
      const b = built[latest.current.front];
      if (b === undefined) return null;
      if (Math.abs(position.current - latest.current.front) > 0.02) {
        return null;
      }
      const box = element.getBoundingClientRect();
      ndc.set(
        ((clientX - box.left) / box.width) * 2 - 1,
        -((clientY - box.top) / box.height) * 2 + 1,
      );
      ray.setFromCamera(ndc, camera);
      b.cells.updateWorldMatrix(true, false);
      inverse.copy(b.cells.matrixWorld).invert();
      const local = ray.ray.clone().applyMatrix4(inverse);
      if (Math.abs(local.direction.z) < 1e-9) return null;
      const t = -local.origin.z / local.direction.z;
      if (t < 0) return null;
      const x = local.origin.x + t * local.direction.x;
      const y = local.origin.y + t * local.direction.y;
      const height = (GRID_WIDTH * b.rows) / b.columns;
      return {
        column: (x / GRID_WIDTH + 0.5) * b.columns,
        row: (0.5 - y / height) * b.rows,
      };
    };
    const neuronUnder = (a: At | null): number | null => {
      const b = built[latest.current.front];
      if (a === null || b === undefined) return null;
      return neuronAt(
        { width: b.width, columns: b.columns, rows: b.rows },
        Math.floor(a.column),
        Math.floor(a.row),
      );
    };

    // Whether a point is on the front card's face: the grid, its margins
    // and its header, in grid units.
    const onFace = (a: At): boolean => {
      const b = built[latest.current.front];
      if (b === undefined) return false;
      const perUnit = b.columns / GRID_WIDTH;
      const margin = MARGIN * perUnit;
      return (
        a.column >= -margin &&
        a.column <= b.columns + margin &&
        a.row >= -(HEADER + MARGIN) * perUnit &&
        a.row <= b.rows + margin
      );
    };

    type Gesture =
      | { readonly kind: "pan"; x: number; y: number }
      | {
          readonly kind: "drag";
          readonly from: At;
          readonly x: number;
          readonly y: number;
          readonly adding: boolean;
          moved: boolean;
        };
    let gesture: Gesture | null = null;

    const hover = (event: PointerEvent) => {
      const neuron = neuronUnder(at(event.clientX, event.clientY));
      element.style.cursor = latest.current.space.current
        ? "grab"
        : neuron === null
          ? ""
          : "crosshair";
      latest.current.onHover(
        neuron === null ? null : { neuron, x: event.offsetX, y: event.offsetY },
      );
    };
    const down = (event: PointerEvent) => {
      const p = latest.current;
      if (event.button === 1 || (event.button === 0 && p.space.current)) {
        event.preventDefault();
        gesture = { kind: "pan", x: event.clientX, y: event.clientY };
        element.setPointerCapture(event.pointerId);
        p.onHover(null);
        p.onPanning(true);
        return;
      }
      if (event.button !== 0) return;
      const from = at(event.clientX, event.clientY);
      if (from === null || !onFace(from)) return;
      gesture = {
        kind: "drag",
        from,
        x: event.clientX,
        y: event.clientY,
        adding: event.shiftKey,
        moved: false,
      };
      element.setPointerCapture(event.pointerId);
    };
    const move = (event: PointerEvent) => {
      const p = latest.current;
      if (gesture === null) {
        hover(event);
        return;
      }
      if (gesture.kind === "pan") {
        p.onPan(event.clientX - gesture.x, event.clientY - gesture.y);
        gesture.x = event.clientX;
        gesture.y = event.clientY;
        return;
      }
      if (
        !gesture.moved &&
        Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) <
          DRAG_PIXELS
      ) {
        return;
      }
      gesture.moved = true;
      p.onHover(null);
      const to = at(event.clientX, event.clientY);
      if (to !== null) p.onBox(gesture.from, to, gesture.adding, false);
    };
    const up = (event: PointerEvent) => {
      const p = latest.current;
      const g = gesture;
      gesture = null;
      if (element.hasPointerCapture(event.pointerId)) {
        element.releasePointerCapture(event.pointerId);
      }
      if (g === null) return;
      if (g.kind === "pan") {
        p.onPanning(false);
        return;
      }
      if (g.moved) {
        p.onBox(
          g.from,
          at(event.clientX, event.clientY) ?? g.from,
          g.adding,
          true,
        );
        return;
      }
      const neuron = neuronUnder(g.from);
      if (neuron !== null) p.onPick(neuron, g.adding);
    };
    const leave = () => {
      if (gesture !== null) return;
      element.style.cursor = "";
      latest.current.onHover(null);
    };
    element.addEventListener("pointerdown", down);
    element.addEventListener("pointermove", move);
    element.addEventListener("pointerup", up);
    element.addEventListener("pointercancel", up);
    element.addEventListener("pointerleave", leave);
    return () => {
      element.removeEventListener("pointerdown", down);
      element.removeEventListener("pointermove", move);
      element.removeEventListener("pointerup", up);
      element.removeEventListener("pointercancel", up);
      element.removeEventListener("pointerleave", leave);
    };
  }, [gl, camera, built]);

  return (
    <>
      {built.map((b, l) => (
        <primitive key={l} object={b.group} />
      ))}
    </>
  );
}

const clamp = (v: number, low: number, high: number) =>
  Math.max(low, Math.min(high, v));

// A card's objects.
interface Built {
  readonly group: Group;
  readonly cells: Mesh;
  readonly faceHeight: number;
  readonly width: number;
  readonly columns: number;
  readonly rows: number;
  readonly pixels: DataTexture;
  readonly marks: DataTexture;
  readonly markBytes: Uint8Array;
  readonly header: CanvasTexture;
  readonly headerCanvas: HTMLCanvasElement;
  readonly grid: ShaderMaterial;
  readonly face: MeshBasicMaterial;
  readonly title: MeshBasicMaterial;
  readonly edge: LineBasicMaterial;
  readonly geometries: readonly (PlaneGeometry | EdgesGeometry)[];
}

const VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAGMENT = /* glsl */ `
uniform sampler2D pixels;
uniform sampler2D marks;
uniform vec2 grid;
uniform vec3 face;
uniform vec3 pin;
uniform vec3 chosen;
uniform float dim;
uniform float opacity;
uniform float tint;
varying vec2 vUv;
void main() {
  vec2 at = vec2(vUv.x, 1.0 - vUv.y);
  vec2 p = at * grid;
  vec2 cell = min(floor(p), grid - 1.0);
  vec3 rgb = textureGrad(pixels, (cell + 0.5) / grid, dFdx(at), dFdy(at)).rgb;
  vec2 w = max(fwidth(p), vec2(1e-6));
  vec2 f = fract(p);
  float edge = min(min(f.x, 1.0 - f.x) / w.x, min(f.y, 1.0 - f.y) / w.y);
  float cellPixels = 1.0 / max(w.x, w.y);
  float gap = (1.0 - smoothstep(0.35, 1.0, edge)) * clamp((cellPixels - 3.0) / 2.0, 0.0, 1.0);
  float own = texelFetch(marks, ivec2(cell), 0).r * 255.0;
  bool ownChosen = own > 1.5;
  bool ownPinned = mod(own, 2.0) > 0.5;
  if (ownChosen) rgb = mix(rgb, chosen, tint);
  rgb = mix(rgb, face, gap);
  // Rings round marked regions, drawn outside them so they show at any
  // cell size, and only in cells outside the region, so a box of many
  // reads as one outline: the selected nearest, a pin's outside it.
  float pinRing = 0.0;
  float chosenRing = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 n = cell + vec2(float(dx), float(dy));
      if (n.x < 0.0 || n.y < 0.0 || n.x >= grid.x || n.y >= grid.y) continue;
      float m = texelFetch(marks, ivec2(n), 0).r * 255.0;
      if (m < 0.5) continue;
      float d = length(max(max(n - p, p - (n + 1.0)), vec2(0.0)) / w);
      float inner = smoothstep(0.2, 0.8, d) * (1.0 - smoothstep(2.0, 2.6, d));
      float outer = smoothstep(2.0, 2.6, d) * (1.0 - smoothstep(3.8, 4.4, d));
      bool isChosen = m > 1.5;
      bool isPinned = mod(m, 2.0) > 0.5;
      if (isChosen && !ownChosen) chosenRing = max(chosenRing, inner);
      if (isPinned && !ownPinned) {
        pinRing = max(pinRing, isChosen && !ownChosen ? outer : inner);
      }
    }
  }
  rgb = mix(rgb, pin, pinRing);
  rgb = mix(rgb, chosen, chosenRing);
  gl_FragColor = vec4(rgb * dim, opacity);
}`;

const srgb = (hex: string) => {
  const n = Number.parseInt(hex.slice(1), 16);
  return new Vector3(
    ((n >> 16) & 255) / 255,
    ((n >> 8) & 255) / 255,
    (n & 255) / 255,
  );
};

function build(card: Card, gl: WebGLRenderer): Built {
  const { grid, painted } = card;
  const height = (GRID_WIDTH * grid.rows) / grid.columns;
  const faceWidth = GRID_WIDTH + 2 * MARGIN;
  const faceHeight = height + HEADER + 2 * MARGIN;

  const pixels = new DataTexture(
    painted.pixels,
    grid.columns,
    grid.rows,
    RGBAFormat,
  );
  pixels.magFilter = NearestFilter;
  pixels.minFilter = LinearMipmapLinearFilter;
  pixels.generateMipmaps = true;
  pixels.anisotropy = gl.capabilities.getMaxAnisotropy();
  pixels.needsUpdate = true;
  const markBytes = new Uint8Array(grid.columns * grid.rows);
  const marks = new DataTexture(markBytes, grid.columns, grid.rows, RedFormat);
  marks.magFilter = NearestFilter;
  marks.minFilter = NearestFilter;
  marks.needsUpdate = true;

  const gridMaterial = new ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    depthWrite: false,
    uniforms: {
      pixels: { value: pixels },
      marks: { value: marks },
      grid: { value: new Vector2(grid.columns, grid.rows) },
      face: { value: srgb(CARD) },
      pin: { value: srgb(RING_PIN) },
      chosen: { value: srgb(RING_SELECTED) },
      dim: { value: 1 },
      opacity: { value: 1 },
      tint: { value: 0 },
    },
  });

  const headerCanvas = document.createElement("canvas");
  headerCanvas.width = 1024;
  headerCanvas.height = Math.round((1024 * HEADER) / GRID_WIDTH);
  const header = new CanvasTexture(headerCanvas);
  header.colorSpace = SRGBColorSpace;
  header.minFilter = LinearFilter;
  header.anisotropy = gl.capabilities.getMaxAnisotropy();

  const translucent = { transparent: true, depthWrite: false } as const;
  const faceMaterial = new MeshBasicMaterial({ ...translucent, color: CARD });
  const titleMaterial = new MeshBasicMaterial({ ...translucent, map: header });
  const edgeMaterial = new LineBasicMaterial({
    ...translucent,
    color: CARD_RULE,
  });

  const faceGeometry = new PlaneGeometry(faceWidth, faceHeight);
  const gridGeometry = new PlaneGeometry(GRID_WIDTH, height);
  const titleGeometry = new PlaneGeometry(GRID_WIDTH, HEADER);
  const edgeGeometry = new EdgesGeometry(faceGeometry);

  const group = new Group();
  const face = new Mesh(faceGeometry, faceMaterial);
  const cells = new Mesh(gridGeometry, gridMaterial);
  cells.position.set(0, -faceHeight / 2 + MARGIN + height / 2, 0.001);
  const title = new Mesh(titleGeometry, titleMaterial);
  title.position.set(0, faceHeight / 2 - MARGIN * 0.6 - HEADER / 2, 0.001);
  const edge = new LineSegments(edgeGeometry, edgeMaterial);
  edge.position.z = 0.002;
  group.add(face, cells, title, edge);
  return {
    group,
    cells,
    faceHeight,
    width: grid.width,
    columns: grid.columns,
    rows: grid.rows,
    pixels,
    marks,
    markBytes,
    header,
    headerCanvas,
    grid: gridMaterial,
    face: faceMaterial,
    title: titleMaterial,
    edge: edgeMaterial,
    geometries: [faceGeometry, gridGeometry, titleGeometry, edgeGeometry],
  };
}

function drawHeader(b: Built, layer: number, pinned: number): void {
  const c = b.headerCanvas;
  const g = c.getContext("2d");
  if (g === null) return;
  g.fillStyle = CARD;
  g.fillRect(0, 0, c.width, c.height);
  g.textBaseline = "middle";
  g.font = `500 ${Math.round(c.height * 0.62)}px "Plex Mono", monospace`;
  g.fillStyle = INK;
  g.textAlign = "left";
  g.fillText(`Layer ${layer}`, 0, c.height / 2);
  if (pinned > 0) {
    g.fillStyle = PIN;
    g.textAlign = "right";
    g.fillText(`● ${pinned} pinned`, c.width, c.height / 2);
  }
  b.header.needsUpdate = true;
}

function dispose(b: Built): void {
  b.pixels.dispose();
  b.marks.dispose();
  b.header.dispose();
  b.grid.dispose();
  b.face.dispose();
  b.title.dispose();
  b.edge.dispose();
  b.geometries.forEach((g) => g.dispose());
}

// A card's place for d = layer - position, fractional while sliding (the
// header's Algorithm, step 3; the reviewed concept's deck.js).
function place(b: Built, d: number): void {
  let x: number;
  let y: number;
  let z: number;
  let turn: number;
  let opacity: number;
  let dim: number;
  if (d >= 0) {
    x = d * 0.3;
    y = d * 0.17;
    z = -d * 0.7;
    turn = 0;
    opacity = d < 1 ? 1 - d * 0.2 : Math.max(0.15, 0.8 - (d - 1) * 0.06);
    dim = d < 1 ? 1 - d * 0.3 : 0.7;
  } else if (d > -1) {
    const t = -d;
    x = -t * GRID_WIDTH * 1.08;
    y = -t * 0.12;
    z = t * 0.35;
    turn = t * 0.45;
    opacity = 1 - t * 0.85;
    dim = 1 - t * 0.4;
  } else {
    const k = -d - 1;
    x = -GRID_WIDTH * 1.08 - k * 0.05;
    y = -0.12;
    z = 0.35 - k * 0.02;
    turn = 0.45;
    opacity = Math.max(0, 0.15 - k * 0.006);
    dim = 0.6;
  }
  b.group.position.set(x, y, z);
  b.group.rotation.y = turn;
  b.group.visible = opacity > 0.01 && d < DRAWN_AHEAD;
  (b.grid.uniforms["opacity"] as { value: number }).value = opacity;
  (b.grid.uniforms["dim"] as { value: number }).value = dim;
  for (const m of [b.face, b.title, b.edge]) m.opacity = opacity;
  b.face.color.set(CARD).multiplyScalar(dim);
  b.title.color.setScalar(dim);
  b.edge.color.set(CARD_RULE).multiplyScalar(dim);
  // Nearer cards draw last, over the ones behind.
  const order = Math.round(1000 - Math.abs(d) * 10) * 4;
  b.group.children.forEach((child, k) => (child.renderOrder = order + k));
}

// Whether the browser gives a WebGL 2 context, which three.js needs; the
// trial context is released at once.
function hasWebGL(): boolean {
  const context = document.createElement("canvas").getContext("webgl2");
  if (context === null) return false;
  context.getExtension("WEBGL_lose_context")?.loseContext();
  return true;
}

function useReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const list = matchMedia(query);
    const onChange = () => setReduced(list.matches);
    list.addEventListener("change", onChange);
    return () => list.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

// Whether the headers' font has loaded, so they are drawn in it.
function useFontsReady(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let live = true;
    void document.fonts.load('500 40px "Plex Mono"').then(() => {
      if (live) setReady(true);
    });
    return () => {
      live = false;
    };
  }, []);
  return ready;
}
