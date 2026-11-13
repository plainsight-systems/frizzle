// Axis H (view): the Map, the workbench's other view in the deck's place:
// every layer at the branch point, one row a layer, so a person can see
// which layer to open (docs/design/concepts/explore.html, its Map).
//
// Contract
//   - LayerMap({neurons, pins, layer, onLayer, onSelect}): over
//     the captures of the side the workbench shows (cards.ts, neuronsOf),
//     at the branch point, as the run computed them: no pin's value is
//     overlaid, unlike the deck's preview, so every mark is a captured
//     value.
//   - Without captures: status "Map" with the page's words for why
//     (cards.ts: "Waiting", "Not captured: the run was cancelled", "Not
//     read back: {reason}", "None: {failure}", …), and no rows.
//   - The columns, in the order a layer computes them (capture/capture.h):
//       "Residual in"       the residual stream coming into the layer, X
//                           before its first norm; layer 0's is the
//                           embedding
//       "Attention added"   what the attention block added into it
//       "MLP neurons"       the neurons, after the layer's edits, as the
//                           down projection read them
//       "MLP added"         what the MLP block added into it
//     Under each heading, the column's legend, the Map's own (the deck's
//     speaks of a scale a layer and its outlines): the deck's heat ramp
//     from 0 to s, s its scale (shape/strip.ts, scaleAcross over that column of
//     every layer) to two significant digits (shape/numbers.ts, significant),
//     then "{g} a mark", the column's group.
//     Under the headings, one line: "size, either sign · one scale a column
//     across the layers, the strongest 1% at full · each mark the strongest
//     of its group, sized so every mark has a pixel".
//   - A row a layer, layer 0 first: a button "Layer {l}", with "● {p}
//     pinned" when p of its neurons are pinned (`pins`); then the four
//     strips, each a mark per group of g values, the group's strongest
//     (shape/strip.ts, strongestOfGroups). A column's g is the smallest
//     that gives each mark at least one of the screen's physical pixels
//     across the column's width (shape/strip.ts, groupFor), and changes as
//     that width does: for 6,912 neurons, 26 in 275 CSS pixels at 1×, 13 at
//     2×. The row of the front layer `layer` is outlined. The selection is not drawn on the Map: the inspector beside
//     it shows the selected neuron, and the deck its cell.
//   - Pointing at a mark shows, in status "Map reading": "Layer {l} ·
//     {column} · {kind} {i} of {width} · {value}", kind "neuron" or
//     "dimension", the value with its sign and two places
//     (shape/numbers.ts, signed), the exact f32 in a <data value>, and
//     "strongest of {a}–{b}", the group's indices. Pointing at nothing reads
//     "Point at a mark to read it".
//   - Clicking a neuron mark selects that neuron alone, in its layer
//     (onSelect, which brings the layer to the front); the inspector beside
//     the Map shows it. Clicking anywhere else in a row, a residual or added
//     mark included, brings that layer to the front (onLayer). The Map stays
//     shown either way; the bar's switch goes back to the deck. Residual
//     dimensions are not editable (research/open.md, slice 7).
//   - Under 900 CSS pixels wide, a row is the layer's button and the
//     neurons' strip; the other columns are not drawn.
//
// Algorithm
//   1. When the shown captures change: for each column, each layer's slot
//      as a view of the words (layers[l].{residual, attentionAdded, neurons,
//      mlpAdded}, never copied), and the column's scale, scaleAcross over
//      its 26 slots.
//   2. Each column's width in physical pixels, from a ResizeObserver on its
//      heading: devicePixelContentBoxSize where the browser gives it, else
//      the CSS width × devicePixelRatio, rounded down. A column not drawn
//      (under 900 pixels) is 0 and gets no strips.
//   3. When step 1's slots or a column's pixels change: g = groupFor(width,
//      pixels) for its slots' width; each slot's marks,
//      strongestOfGroups(slot, g); each strip's pixels, paintStrip at the
//      column's scale with the deck's palette (theme/palette.ts,
//      DECK_PALETTE).
//   4. Each strip is a canvas as many pixels wide as it has marks, one
//      high, drawn with putImageData and stretched by CSS without
//      smoothing: no more marks than physical pixels, so stretching never
//      drops one.
//   5. Pointer over a strip: the mark at floor(x / width × marks); its
//      index in the slot from step 3; the group g × mark to g × mark + g −
//      1, the last cut at the slot's end; the reading from the slot's
//      words.
//
// Design and cost
//   - In the workbench, beside the inspector, not a page of its own: the
//     side radio, the selected neuron and the pins are the inspector's, and
//     the Map is how to find a layer and a neuron worth opening.
//   - One scale for a column across the layers, magnitude not sign, the
//     strongest of a group selected not combined, the group sized to the
//     pixels: shape/strip.ts says why.
//   - Physical pixels, not CSS pixels: every value that fits is drawn, so a
//     2× screen shows twice the detail of a 1× one. A mark can then be half
//     a CSS pixel, finer than a pointer is placed; the reading names the
//     mark the pointer is over and its group, so what is read and clicked
//     is always stated.
//   - The heading measures the column, as every row's strip has its width.
//     A change of devicePixelRatio with no change of size (a window moved
//     to another screen) is seen where devicePixelContentBoxSize is given
//     (Chrome, Firefox); elsewhere the groups stay until the next resize.
//   - Attention's weights and statistics cross with the store but are not
//     drawn here: one row a layer has no room for 4 heads over up to
//     32,768 keys, and showing them is not this slice's (the slice plan's
//     omissions).
//   - Global and local layers are not marked: the page would have to
//     reckon it from each layer's key count, which the concept's made-up
//     data did and the captures do not state.
//   - 104 canvases, drawn when the captures or a column's pixels change,
//     and the strip pass of shape/strip.ts; while a window is resized, a
//     pass each time a column's pixels change. No WebGL, so the deck's
//     context is released while the Map is shown (Workbench.tsx mounts one
//     of the two).

import {
  type MouseEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { Selection } from "../../App.tsx";
import type { Edit } from "../../engine/edits.ts";
import type { LayerSlots } from "../../protocol/messages.ts";
import { signed, significant } from "../../shape/numbers.ts";
import {
  groupFor,
  paintStrip,
  scaleAcross,
  strongestOfGroups,
} from "../../shape/strip.ts";
import { DECK_PALETTE } from "../../theme/palette.ts";
import type { Neurons } from "./cards.ts";

// The columns, in the order a layer computes them (the header).
const COLUMNS = [
  { slot: "residual", label: "Residual in", kind: "dimension" },
  { slot: "attentionAdded", label: "Attention added", kind: "dimension" },
  { slot: "neurons", label: "MLP neurons", kind: "neuron" },
  { slot: "mlpAdded", label: "MLP added", kind: "dimension" },
] as const satisfies readonly {
  slot: keyof LayerSlots;
  label: string;
  kind: string;
}[];

// One strip: its slot's words, the index each mark selected, its pixels.
interface Strip {
  readonly values: Float32Array;
  readonly marks: Int32Array;
  readonly pixels: Uint8Array;
}

// A column's slots, a view a layer, and its scale across them.
interface Slots {
  readonly scale: number;
  readonly slots: readonly Float32Array[];
}

// A column as drawn: its scale, its group (0 before it is measured, or
// while it is not drawn) and a strip a layer.
interface Column {
  readonly scale: number;
  readonly group: number;
  readonly strips: readonly Strip[];
}

// A column as drawn, with what it was drawn from.
interface Drawn {
  readonly from: Slots;
  readonly pixels: number;
  readonly column: Column;
}

const NO_MARKS = new Int32Array(0);
const NO_PIXELS = new Uint8Array(0);

// What the pointer is over: a layer, a column and a mark.
interface Pointed {
  readonly layer: number;
  readonly column: number;
  readonly mark: number;
}

export function LayerMap(props: {
  readonly neurons: Neurons;
  readonly pins: readonly Edit[];
  readonly layer: number;
  readonly onLayer: (layer: number) => void;
  readonly onSelect: (chosen: Selection) => void;
}): ReactNode {
  const { neurons, pins, layer, onLayer, onSelect } = props;
  const captures = neurons.kind === "taken" ? neurons.captures : null;
  const [pointed, setPointed] = useState<Pointed | null>(null);

  const heads = useRef<(HTMLSpanElement | null)[]>([]);
  const [widths, setWidths] = useState<readonly number[]>(() =>
    COLUMNS.map(() => 0),
  );

  // The header's Algorithm, step 1.
  const sliced = useMemo((): readonly Slots[] | null => {
    if (captures === null) return null;
    const { layers, words } = captures;
    return COLUMNS.map(({ slot }) => {
      const slots = layers.map(({ [slot]: { first, width } }) =>
        words.subarray(first, first + width),
      );
      return { scale: scaleAcross(slots), slots };
    });
  }, [captures]);

  // The header's Algorithm, step 2: the headings exist while there are
  // captures.
  const shown = sliced !== null;
  useEffect(() => {
    if (!shown) return;
    const observer = new ResizeObserver((entries) => {
      setWidths((before) => {
        const after = [...before];
        for (const entry of entries) {
          const c = heads.current.indexOf(entry.target as HTMLSpanElement);
          if (c >= 0) after[c] = physicalWidth(entry);
        }
        return after.every((w, c) => w === before[c]) ? before : after;
      });
    });
    for (const head of heads.current) {
      if (head === null) continue;
      try {
        observer.observe(head, { box: "device-pixel-content-box" });
      } catch {
        // A browser without device-pixel boxes: CSS pixels, scaled.
        observer.observe(head);
      }
    }
    return () => {
      observer.disconnect();
      // Headings mounted hidden are never measured, so none keeps a width.
      setWidths(COLUMNS.map(() => 0));
    };
  }, [shown]);

  // The header's Algorithm, step 3, a column at a time: a column whose
  // slots and pixels are as they were is the one drawn then. The cache
  // holds only what its inputs determine, so a render that is thrown away
  // leaves nothing wrong in it.
  const drawn = useRef<(Drawn | undefined)[]>([]);
  const columns = useMemo((): readonly Column[] | null => {
    if (sliced === null) return null;
    return sliced.map((from, c) => {
      const pixels = widths[c] ?? 0;
      const kept = drawn.current[c];
      if (kept?.from === from && kept.pixels === pixels) return kept.column;
      const column = drawColumn(from, pixels);
      drawn.current[c] = { from, pixels, column };
      return column;
    });
  }, [sliced, widths]);

  // A reading of the strips shown only: a pointer kept from an earlier
  // run, or from before the groups changed, names a mark not under it.
  useEffect(() => setPointed(null), [columns]);

  const count = captures?.layers.length ?? 0;
  const status = neurons.kind === "none" ? neurons.why : `${count} layers`;

  return (
    <section className="column map-column" aria-label="Map">
      <h2>
        Every layer{" "}
        <output aria-label="Map" className="muted">
          {status}
        </output>
      </h2>
      {columns !== null && (
        <>
          <p className="map-reading">
            <output aria-label="Map reading">
              {reading(columns, pointed)}
            </output>
          </p>
          <div className="map-heads">
            <span>Layer</span>
            {COLUMNS.map(({ label }, c) => (
              <span
                key={label}
                ref={(head) => {
                  heads.current[c] = head;
                }}
                className={columnClass(c)}
              >
                {label}
                <MapLegend
                  scale={columns[c]?.scale ?? 0}
                  group={columns[c]?.group ?? 0}
                />
              </span>
            ))}
          </div>
          <p className="legend muted">
            size, either sign · one scale a column across the layers, the
            strongest 1% at full · each mark the strongest of its group, sized
            so every mark has a pixel
          </p>
          <div className="map-rows">
            {Array.from({ length: count }, (_, l) => {
              const pinned = pins.filter((p) => p.layer === l).length;
              return (
                <div
                  key={l}
                  className={l === layer ? "map-row front" : "map-row"}
                  onClick={() => onLayer(l)}
                >
                  <button
                    type="button"
                    aria-pressed={l === layer}
                    onClick={(event) => {
                      event.stopPropagation();
                      onLayer(l);
                    }}
                  >
                    Layer {l}
                    {pinned > 0 && (
                      <small className="map-pinned"> ● {pinned} pinned</small>
                    )}
                  </button>
                  {columns.map((column, c) => {
                    const strip = column.strips[l];
                    if (strip === undefined) return null;
                    const kind = COLUMNS[c]?.kind;
                    return (
                      <StripCanvas
                        key={c}
                        className={columnClass(c)}
                        label={`Layer ${l}'s ${COLUMNS[c]?.label ?? ""}`}
                        pixels={strip.pixels}
                        marks={strip.marks.length}
                        onPoint={(mark) =>
                          setPointed(
                            mark === null
                              ? null
                              : { layer: l, column: c, mark },
                          )
                        }
                        onPick={(mark, event) => {
                          if (kind !== "neuron") return;
                          const neuron = strip.marks[mark];
                          if (neuron === undefined) return;
                          event.stopPropagation();
                          onSelect({ layer: l, neurons: [neuron] });
                        }}
                      />
                    );
                  })}
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

// A column's strips at its pixels (the header's Algorithm, step 3); none
// before it is measured, or while it is not drawn.
function drawColumn({ scale, slots }: Slots, pixels: number): Column {
  if (pixels < 1) {
    return {
      scale,
      group: 0,
      strips: slots.map((values) => ({
        values,
        marks: NO_MARKS,
        pixels: NO_PIXELS,
      })),
    };
  }
  const widest = Math.max(0, ...slots.map((v) => v.length));
  const group = groupFor(widest, pixels);
  return {
    scale,
    group,
    strips: slots.map((values) => {
      const marks = strongestOfGroups(values, group);
      return {
        values,
        marks,
        pixels: paintStrip(values, marks, scale, DECK_PALETTE),
      };
    }),
  };
}

// The neuron column is drawn at every width; the others only above 900px
// (theme.css).
function columnClass(c: number): string | undefined {
  return COLUMNS[c]?.kind === "neuron" ? undefined : "map-wide";
}

// The header's "Map reading".
function reading(
  columns: readonly Column[],
  pointed: Pointed | null,
): ReactNode {
  const column = pointed === null ? undefined : columns[pointed.column];
  const strip = column?.strips[pointed?.layer ?? -1];
  const index = strip?.marks[pointed?.mark ?? -1];
  const named = pointed === null ? undefined : COLUMNS[pointed.column];
  if (
    pointed === null ||
    column === undefined ||
    strip === undefined ||
    index === undefined ||
    named === undefined
  ) {
    return "Point at a mark to read it";
  }
  const value = strip.values[index] ?? Number.NaN;
  const first = pointed.mark * column.group;
  const last = Math.min(first + column.group, strip.values.length) - 1;
  return (
    <>
      Layer {pointed.layer} · {named.label} · {named.kind} {index} of{" "}
      {strip.values.length} · <data value={String(value)}>{signed(value)}</data>{" "}
      · strongest of {first}–{last}
    </>
  );
}

// A column's physical width (the header's Algorithm, step 2).
function physicalWidth(entry: ResizeObserverEntry): number {
  const box = entry.devicePixelContentBoxSize?.[0];
  if (box !== undefined) return Math.floor(box.inlineSize);
  return Math.floor(entry.contentRect.width * devicePixelRatio);
}

// A column's legend: the deck's ramp from 0 to the column's scale, and its
// group once measured.
function MapLegend(props: {
  readonly scale: number;
  readonly group: number;
}): ReactNode {
  const rgb = (c: readonly number[]) => `rgb(${c.join(",")})`;
  return (
    <span className="map-legend">
      <span>0</span>
      <span
        className="ramp"
        style={{
          background: `linear-gradient(90deg, ${DECK_PALETTE.heat.map(rgb).join(", ")})`,
        }}
      />
      <span>{significant(props.scale)}</span>
      {props.group > 0 && (
        <span className="map-group">{props.group} a mark</span>
      )}
    </span>
  );
}

// One strip as a canvas a pixel a mark, stretched without smoothing (the
// header's Algorithm, steps 4 and 5).
function StripCanvas(props: {
  readonly className: string | undefined;
  readonly label: string;
  readonly pixels: Uint8Array;
  readonly marks: number;
  readonly onPoint: (mark: number | null) => void;
  readonly onPick: (mark: number, event: MouseEvent) => void;
}): ReactNode {
  const { className, label, pixels, marks, onPoint, onPick } = props;
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (context === null || context === undefined || marks === 0) return;
    const clamped = new Uint8ClampedArray(
      pixels.buffer as ArrayBuffer,
      pixels.byteOffset,
      pixels.length,
    );
    context.putImageData(new ImageData(clamped, marks, 1), 0, 0);
  }, [pixels, marks]);
  const markAt = (event: MouseEvent<HTMLCanvasElement>): number => {
    const box = event.currentTarget.getBoundingClientRect();
    const at = Math.floor(((event.clientX - box.left) / box.width) * marks);
    return Math.max(0, Math.min(marks - 1, at));
  };
  return (
    <canvas
      ref={canvas}
      className={className}
      width={Math.max(1, marks)}
      height={1}
      aria-label={label}
      onPointerMove={(event) => onPoint(markAt(event))}
      onPointerLeave={() => onPoint(null)}
      onClick={(event) => onPick(markAt(event), event)}
    />
  );
}
