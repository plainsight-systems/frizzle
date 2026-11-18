// Axis H (view): the Ride's picture: a night road through one glass pane a
// layer, drawn on a 2D canvas (docs/assets/frizzle-readme.png is the scene;
// research-era concept: The Ride). Plain drawing, no React.
//
// Contract
//   - drawScene(x, frame): one frame, in CSS pixels, of a viewer `frame.t`
//     layers along the road, looking ahead. Each pane is a layer: its best
//     guess (shape/ride.ts) written on the glass, glowing when the reply goes
//     on to write it, and its strongest neurons as points of light where
//     they sit in the layer's grid, amber positive, blue negative. Pinned
//     neurons are ringed, with a brass dial beside them.
//   - With `ghost`, a second road: past `forkAt` the two part, the ridden
//     one (`main`) to the right and the other to the left, its panes dim,
//     `fork` from 0 to 1 how far they have parted.
//   - At the end, once `arrived`, a lit pane with the next word.
//   - Returns the frame's targets: every light and pin on the pane in front,
//     as circles in CSS pixels, for the pointer.
//
// Design and cost
//   - Panes always face the viewer, so each is a scaled rectangle and no 3D
//     library is needed: a point projects by F / distance. About ten panes
//     a frame, each a rectangle, up to 24 × 15 dots, six gradients and a
//     word. One road or two, about 40 quads each.

import type { Road } from "../../shape/ride.ts";
import { onTheWay } from "../../shape/ride.ts";
import { RIDE } from "../../theme/palette.ts";

export const GAP = 9; // between panes, in world units
export const AHEAD = 7.6; // how far ahead the pane in front stands
const EYE = 1.75;
const PANE_WIDTH = 6.2;
const PANE_HEIGHT = 3.9;
const SPREAD = 4.6; // how far the two roads part, each side
const ROAD_HALF = 1.7;
const SERIF = '"Plex Serif", Georgia, serif';
const MONO = '"Plex Mono", ui-monospace, monospace';

export interface SceneFrame {
  readonly width: number;
  readonly height: number;
  readonly t: number;
  readonly fork: number;
  readonly clock: number;
  readonly still: boolean;
  readonly moving: number; // 0 at rest, up to 1 driving
  readonly lean: number;
  readonly arrived: boolean;
  readonly main: Road;
  readonly ghost: Road | null;
  readonly forkAt: number;
  readonly upcoming: readonly string[]; // main's words from the point on
  readonly ghostUpcoming: readonly string[];
  readonly pins: ReadonlyMap<number, readonly number[]>; // by layer
}

export interface Target {
  readonly x: number;
  readonly y: number;
  readonly r: number;
  readonly layer: number;
  readonly neuron: number;
}

const ease = (v: number): number => {
  const c = Math.max(0, Math.min(1, v));
  return c * c * (3 - 2 * c);
};

// The stars' places, fixed: fractions of the sky.
const STARS = Array.from({ length: 130 }, (_, i) => ({
  a: (i * 197.3) % 1,
  b: ((i * i * 0.0137) % 1) * 0.92,
  r: 0.35 + ((i * 7) % 5) / 7,
}));

export function drawScene(
  x: CanvasRenderingContext2D,
  f: SceneFrame,
): { readonly targets: readonly Target[]; readonly cameraX: number } {
  const { width: W, height: H, t, fork, main, ghost, forkAt } = f;
  const last = main.layers.length - 1;
  const two = ghost !== null && fork > 0.015;
  const laneX = (i: number, side: number): number =>
    side * SPREAD * fork * ease((i - (forkAt - 0.75)) / 2.3);
  const ahead = t + AHEAD / GAP;
  const cx = two ? laneX(ahead - 0.15, 1) : 0;
  const cz = t * GAP - AHEAD;
  const horizon = H * 0.4;
  const F = Math.min(W * 0.92, H * 1.25);
  const targets: Target[] = [];

  const project = (px: number, py: number, pz: number) => {
    const dz = pz - cz;
    if (dz < 0.25) return null;
    const s = F / dz;
    return { x: W / 2 + (px - cx) * s, y: horizon + (EYE - py) * s, s, dz };
  };

  const ridge = (colour: string, amp: number, par: number, seed: number) => {
    x.fillStyle = colour;
    x.beginPath();
    x.moveTo(0, horizon + 1);
    for (let px = 0; px <= W + 12; px += 12) {
      const u = ((px + cx * par * 6 + t * par * 3) / W) * 7 + seed;
      const h =
        (Math.sin(u) * 0.5 +
          Math.sin(u * 2.3 + seed) * 0.3 +
          Math.sin(u * 5.1) * 0.14 +
          0.75) *
        H *
        amp;
      x.lineTo(px, horizon - h);
    }
    x.lineTo(W, horizon + 1);
    x.fill();
  };

  // The sky, the dusk at the road's end, the hills and the ground.
  const sky = x.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, RIDE.skyTop);
  sky.addColorStop(1, RIDE.skyLow);
  x.fillStyle = sky;
  x.fillRect(0, 0, W, horizon + 1);
  x.fillStyle = RIDE.star;
  for (const { a, b, r } of STARS) {
    x.globalAlpha = 0.25 + r * 0.45;
    x.fillRect((a * W * 1.1 - cx * 2 + W) % W, b * horizon, r * 1.5, r * 1.5);
  }
  x.globalAlpha = 1;
  const moonX = W * 0.12 - cx * 3;
  const moonY = horizon * 0.27;
  const moonR = Math.min(W, H) * 0.045;
  x.fillStyle = RIDE.moon;
  x.beginPath();
  x.arc(moonX, moonY, moonR, 0, 7);
  x.fill();
  x.fillStyle = RIDE.moonShade;
  x.beginPath();
  x.arc(moonX + moonR * 0.42, moonY - moonR * 0.12, moonR * 0.9, 0, 7);
  x.fill();
  const near = ease((t - last * 0.56) / (last * 0.44));
  const dusk = x.createRadialGradient(
    W / 2,
    horizon,
    0,
    W / 2,
    horizon,
    W * (0.25 + 0.28 * near),
  );
  dusk.addColorStop(0, `rgba(${RIDE.dusk},${0.3 + 0.42 * near})`);
  dusk.addColorStop(1, `rgba(${RIDE.dusk},0)`);
  x.fillStyle = dusk;
  x.fillRect(0, 0, W, H);
  ridge(RIDE.ridgeFar, 0.075, 0.9, 11);
  ridge(RIDE.ridgeNear, 0.05, 1.5, 29);
  const ground = x.createLinearGradient(0, horizon, 0, H);
  ground.addColorStop(0, RIDE.groundTop);
  ground.addColorStop(1, RIDE.groundLow);
  x.fillStyle = ground;
  x.fillRect(0, horizon, W, H - horizon);

  const road = (side: number, dim: boolean) => {
    const end = GAP * (last + 1.5);
    let previous: {
      a: NonNullable<ReturnType<typeof project>>;
      b: NonNullable<ReturnType<typeof project>>;
    } | null = null;
    for (
      let z = cz + 0.5;
      z < Math.min(cz + GAP * 13, end);
      z += Math.max(0.8, (z - cz) * 0.11)
    ) {
      const mid = laneX(z / GAP, side);
      const a = project(mid - ROAD_HALF, 0, z);
      const b = project(mid + ROAD_HALF, 0, z);
      if (a === null || b === null) continue;
      if (previous !== null) {
        const fade = Math.max(0, 1 - (z - cz) / (GAP * 12));
        x.globalAlpha = (dim ? 0.55 : 1) * Math.min(1, fade * 1.5);
        x.fillStyle = RIDE.road;
        x.beginPath();
        x.moveTo(previous.a.x, previous.a.y);
        x.lineTo(previous.b.x, previous.b.y);
        x.lineTo(b.x, b.y);
        x.lineTo(a.x, a.y);
        x.fill();
        x.strokeStyle = RIDE.roadEdge;
        x.lineWidth = Math.max(0.6, a.s * 0.05);
        x.beginPath();
        x.moveTo(previous.a.x, previous.a.y);
        x.lineTo(a.x, a.y);
        x.moveTo(previous.b.x, previous.b.y);
        x.lineTo(b.x, b.y);
        x.stroke();
        if (Math.floor(z / 1.7) % 2 === 0) {
          x.strokeStyle = RIDE.roadLine;
          x.lineWidth = Math.max(0.7, a.s * 0.07);
          x.beginPath();
          x.moveTo((previous.a.x + previous.b.x) / 2, previous.a.y);
          x.lineTo((a.x + b.x) / 2, a.y);
          x.stroke();
        }
      }
      previous = { a, b };
    }
    x.globalAlpha = 1;
  };

  const fitted = (word: string, most: number, size: number, weight: number) => {
    x.font = `${weight} ${size}px ${SERIF}`;
    const w = x.measureText(word).width;
    if (w <= most) return size;
    const smaller = (size * most) / w;
    x.font = `${weight} ${smaller}px ${SERIF}`;
    return smaller;
  };

  const pane = (
    i: number,
    side: number,
    from: Road,
    upcoming: readonly string[],
    dim: boolean,
  ) => {
    const layer = from.layers[i];
    if (layer === undefined) return;
    const z = i * GAP;
    const px = laneX(i, side);
    const tl = project(px - PANE_WIDTH / 2, PANE_HEIGHT, z);
    const br = project(px + PANE_WIDTH / 2, 0, z);
    if (tl === null || br === null) return;
    const { dz } = tl;
    if (dz < 0.8 || dz > GAP * 9.6) return;
    const alpha =
      Math.min(1, (dz - 0.8) / 2) *
      Math.exp(-Math.max(0, dz - AHEAD) / (GAP * 3.4)) *
      (dim ? 0.5 : 1);
    x.globalAlpha = alpha;
    const w = br.x - tl.x;
    const h = br.y - tl.y;
    const focus = !dim && Math.abs(dz - AHEAD) < GAP * 0.55;
    x.fillStyle = focus
      ? `rgba(${RIDE.glassFocus},.72)`
      : `rgba(${RIDE.glass},.075)`;
    x.fillRect(tl.x, tl.y, w, h);
    x.strokeStyle = `rgba(${RIDE.paneEdge},.5)`;
    x.lineWidth = Math.max(0.6, tl.s * 0.035);
    x.strokeRect(tl.x, tl.y, w, h);
    if (w > 70) {
      x.fillStyle = `rgba(${RIDE.paneEdge},.17)`;
      const across = w > 230 ? 24 : 12;
      const down = Math.round(across * 0.63);
      const dot = Math.max(0.7, (w / across) * 0.055);
      for (let u = 0; u < across; u++) {
        for (let v = 0; v < down; v++) {
          x.fillRect(
            tl.x + ((u + 0.5) / across) * w,
            tl.y + ((v + 0.5) / down) * h,
            dot,
            dot,
          );
        }
      }
    }
    const { columns, rows } = layer.grid;
    const at = (neuron: number) => ({
      x: tl.x + (((neuron % columns) + 0.5) / columns) * w,
      y: tl.y + ((Math.floor(neuron / columns) + 0.5) / rows) * h,
    });
    const strongest = Math.max(
      1e-6,
      ...layer.lights.map((l) => Math.abs(l.value)),
    );
    for (const { neuron, value } of layer.lights) {
      const p = at(neuron);
      const r = Math.max(
        1.3,
        w * (0.006 + (0.024 * Math.abs(value)) / strongest),
      );
      const colour = value < 0 ? RIDE.negative : RIDE.positive;
      const light = x.createRadialGradient(p.x, p.y, 0, p.x, p.y, r);
      light.addColorStop(0, `rgba(${colour},1)`);
      light.addColorStop(0.3, `rgba(${colour},.6)`);
      light.addColorStop(1, `rgba(${colour},0)`);
      x.fillStyle = light;
      x.beginPath();
      x.arc(p.x, p.y, r, 0, 7);
      x.fill();
      if (focus) {
        targets.push({ ...p, r: Math.max(9, r), layer: i, neuron });
      }
    }
    const word = layer.guesses[0] ?? "";
    const lit = onTheWay(word, upcoming);
    const size = fitted(
      word,
      w * 0.84,
      Math.min(h * 0.34, w * 0.2),
      lit ? 500 : 400,
    );
    x.textAlign = "center";
    x.textBaseline = "middle";
    const paneAlpha = x.globalAlpha;
    // A pane being driven through keeps its word out of the way.
    if (dz < AHEAD - GAP * 0.55) x.globalAlpha = 0;
    else if (!focus && !dim) x.globalAlpha = paneAlpha * 0.34;
    if (lit) {
      x.shadowColor = RIDE.glow;
      x.shadowBlur = focus ? size * 0.4 : 0;
      x.fillStyle = RIDE.wordLit;
    } else x.fillStyle = `rgba(${RIDE.word},.8)`;
    x.fillText(word, tl.x + w / 2, tl.y + h * 0.5);
    x.shadowBlur = 0;
    x.globalAlpha = paneAlpha;
    if (w > 150) {
      x.font = `${Math.max(9, w * 0.024)}px ${MONO}`;
      x.textAlign = "left";
      x.fillStyle = `rgba(${RIDE.word},.62)`;
      x.fillText(`LAYER ${i}`, tl.x + w * 0.03, tl.y + h * 0.07);
    }
    if (!dim) {
      for (const neuron of f.pins.get(i) ?? []) {
        const p = at(neuron);
        const box = w * 0.024;
        const dial = Math.max(9, w * 0.052);
        const dx = Math.min(tl.x + w - dial * 1.3, p.x + dial * 2.6);
        x.strokeStyle = RIDE.pin;
        x.lineWidth = 1.5;
        x.strokeRect(p.x - box, p.y - box, box * 2, box * 2);
        x.beginPath();
        x.moveTo(p.x + box, p.y);
        x.lineTo(dx - dial, p.y);
        x.stroke();
        const brass = x.createRadialGradient(
          dx - dial * 0.3,
          p.y - dial * 0.3,
          1,
          dx,
          p.y,
          dial,
        );
        brass.addColorStop(0, RIDE.dialLight);
        brass.addColorStop(1, RIDE.dialDark);
        x.fillStyle = brass;
        x.beginPath();
        x.arc(dx, p.y, dial, 0, 7);
        x.fill();
        x.strokeStyle = RIDE.dialInk;
        x.stroke();
        const angle = 1.3;
        x.lineWidth = Math.max(2, dial * 0.2);
        x.beginPath();
        x.moveTo(dx, p.y);
        x.lineTo(
          dx + Math.cos(angle) * dial * 0.78,
          p.y + Math.sin(angle) * dial * 0.78,
        );
        x.stroke();
        if (focus) {
          targets.push({ x: dx, y: p.y, r: dial * 1.4, layer: i, neuron });
        }
      }
    }
    x.globalAlpha = 1;
  };

  const destination = (side: number, word: string) => {
    if (!f.arrived) return;
    const z = (last + 1) * GAP;
    const px = laneX(last + 1, side);
    const tl = project(px - PANE_WIDTH / 2, PANE_HEIGHT, z);
    const br = project(px + PANE_WIDTH / 2, 0, z);
    if (tl === null || br === null) return;
    const w = br.x - tl.x;
    const h = br.y - tl.y;
    x.globalAlpha = ease((t - (last - 0.7)) / 0.7);
    x.fillStyle = `rgba(${RIDE.positive},.2)`;
    x.fillRect(tl.x, tl.y, w, h);
    x.strokeStyle = RIDE.pin;
    x.lineWidth = 1.5;
    x.strokeRect(tl.x, tl.y, w, h);
    fitted(word, w * 0.8, h * 0.4, 500);
    x.textAlign = "center";
    x.textBaseline = "middle";
    x.shadowColor = RIDE.glow;
    x.shadowBlur = 18;
    x.fillStyle = RIDE.wordLit;
    x.fillText(word, tl.x + w / 2, tl.y + h / 2);
    x.shadowBlur = 0;
    x.globalAlpha = 1;
  };

  const coach = () => {
    const w = Math.min(W * 0.2, H * 0.34, 178);
    const h = w * 0.83;
    const bob = f.still ? 0 : Math.sin(f.clock * 9) * (0.15 + f.moving * 0.95);
    const cxs = W / 2;
    const by = H - h * 0.2 + bob;
    const round = (
      a: number,
      b: number,
      c: number,
      d: number,
      r: number | number[],
    ) => {
      x.beginPath();
      x.roundRect(a, b, c, d, r);
    };
    const beam = x.createLinearGradient(0, by - h * 0.3, 0, horizon + 10);
    beam.addColorStop(0, `rgba(${RIDE.headlamp},.3)`);
    beam.addColorStop(1, `rgba(${RIDE.headlamp},0)`);
    x.fillStyle = beam;
    x.beginPath();
    x.moveTo(cxs - w * 0.42, by - h * 0.15);
    x.lineTo(cxs + w * 0.42, by - h * 0.15);
    x.lineTo(cxs + w * 0.09, horizon + H * 0.1);
    x.lineTo(cxs - w * 0.09, horizon + H * 0.1);
    x.fill();
    x.save();
    x.translate(cxs, by);
    x.rotate(Math.max(-0.05, Math.min(0.05, f.lean)));
    x.fillStyle = "rgba(0,0,0,.45)";
    x.beginPath();
    x.ellipse(0, h * 0.06, w * 0.56, h * 0.08, 0, 0, 7);
    x.fill();
    x.fillStyle = RIDE.coachTyre;
    round(-w * 0.45, -h * 0.08, w * 0.17, h * 0.16, 4);
    x.fill();
    round(w * 0.28, -h * 0.08, w * 0.17, h * 0.16, 4);
    x.fill();
    const body = x.createLinearGradient(0, -h, 0, 0);
    body.addColorStop(0, RIDE.coachTop);
    body.addColorStop(0.55, RIDE.coachMiddle);
    body.addColorStop(1, RIDE.coachLow);
    x.fillStyle = body;
    round(-w / 2, -h * 0.94, w, h * 0.9, [
      w * 0.2,
      w * 0.2,
      w * 0.07,
      w * 0.07,
    ]);
    x.fill();
    x.strokeStyle = RIDE.coachLine;
    x.lineWidth = 1.3;
    x.stroke();
    x.fillStyle = RIDE.coachRack;
    round(-w * 0.34, -h, w * 0.68, h * 0.075, 3);
    x.fill();
    x.fillStyle = RIDE.coachGlass;
    round(-w * 0.39, -h * 0.78, w * 0.78, h * 0.31, w * 0.05);
    x.fill();
    x.fillStyle = RIDE.coachLit;
    round(-w * 0.36, -h * 0.75, w * 0.34, h * 0.25, 3);
    x.fill();
    round(w * 0.02, -h * 0.75, w * 0.34, h * 0.25, 3);
    x.fill();
    x.strokeStyle = RIDE.coachFrame;
    x.lineWidth = 2.4;
    x.beginPath();
    x.moveTo(0, -h * 0.78);
    x.lineTo(0, -h * 0.47);
    x.stroke();
    x.fillStyle = RIDE.coachStripe;
    x.fillRect(-w / 2 + 1, -h * 0.39, w - 2, h * 0.05);
    x.fillStyle = RIDE.coachLine;
    round(-w * 0.13, -h * 0.28, w * 0.26, h * 0.1, 2);
    x.fill();
    x.fillStyle = RIDE.star;
    x.font = `${w * 0.058}px ${MONO}`;
    x.textAlign = "center";
    x.textBaseline = "middle";
    x.fillText("FRIZ", 0, -h * 0.23);
    for (const s of [-1, 1]) {
      x.shadowColor = RIDE.coachLamp;
      x.shadowBlur = 11;
      x.fillStyle = RIDE.coachLamp;
      x.beginPath();
      x.arc(s * w * 0.37, -h * 0.2, w * 0.042, 0, 7);
      x.fill();
      x.shadowBlur = 0;
    }
    x.fillStyle = RIDE.coachLine;
    round(-w * 0.52, -h * 0.08, w * 1.04, h * 0.055, 2);
    x.fill();
    x.restore();
  };

  if (two) {
    road(-1, true);
    road(1, false);
    destination(-1, ghost.next);
    destination(1, main.next);
  } else {
    road(0, false);
    destination(0, main.next);
  }
  const first = Math.max(0, Math.floor(t) - 1);
  const end = Math.min(last, Math.ceil(t) + 9);
  for (let i = end; i >= first; i--) {
    if (two && i >= forkAt) {
      pane(i, -1, ghost, f.ghostUpcoming, true);
      pane(i, 1, main, f.upcoming, false);
    } else pane(i, 0, main, f.upcoming, false);
  }
  coach();
  return { targets, cameraX: cx };
}
