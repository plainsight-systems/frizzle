// Concept: one Gemma 3 1B forward pass as the matrices it computes, drawn at their real shapes.
// Rows are tokens (T = 6), columns are dimensions. Values are made up; grey = magnitude.
const stage = document.getElementById('stage'), labelsEl = document.getElementById('labels'), panel = document.getElementById('panel');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); stage.insertBefore(renderer.domElement, labelsEl);
const scene = new THREE.Scene();
const VIEW = 14;
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -200, 400);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableRotate = false; controls.screenSpacePanning = true;
function fit() { const w = Math.max(stage.clientWidth, 1), h = Math.max(stage.clientHeight, 1); renderer.setSize(w, h); const a = w / h; camera.left = -VIEW * a; camera.right = VIEW * a; camera.top = VIEW; camera.bottom = -VIEW; camera.updateProjectionMatrix(); }
new ResizeObserver(fit).observe(stage); fit();

function rng(s) { return function () { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const T = 6, L = 26, D = 1152, H = 4, HD = 256, F = 6912, V = 262144;
const ROW = 0.3;          // one token row, along the flow axis
const COL = 0.004;        // one dimension column, along the depth axis
const SQ = ROW;           // attention score cells are T x T, drawn square
let layer = 14, selected = null;

// Texture is T wide (tokens, along x) by cols high (dimensions, along depth).
function texture(cols, rows, fill) {
  const d = new Uint8Array(cols * rows * 4);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const v = fill(r, c); const i = (c * rows + r) * 4;
    if (v === null) { d[i] = 250; d[i + 1] = 249; d[i + 2] = 245; d[i + 3] = 255; continue; }
    const g = 244 - Math.min(1, Math.abs(v)) * 170; d[i] = g; d[i + 1] = g - 2; d[i + 2] = g - 8; d[i + 3] = 255;
  }
  const t = new THREE.DataTexture(d, rows, cols, THREE.RGBAFormat); t.magFilter = THREE.NearestFilter; t.minFilter = THREE.LinearFilter; t.needsUpdate = true; return t;
}
const sparse = (seed, scale) => { const r = rng(seed); return () => (r() < 0.5 ? -1 : 1) * Math.pow(r(), 12) * scale; };

// The pass, in order. Each slot is one step along the flow; a slot may hold several matrices side by side.
// mats: [label, cols, kind]. kind 'scores' draws a T x T causal grid.
function stages() {
  return [
    { id: 'emb', once: true, name: 'Embedding lookup', op: 'one row of the embedding table per token, × √1152', mats: [['X', D]] , shape: 'T × 1,152' },
    { id: 'x', name: 'Residual in', op: 'the residual stream entering this layer', mats: [['X', D]], shape: 'T × 1,152' },
    { id: 'n1', name: 'RMSNorm', op: 'each row scaled to unit RMS, times a learned gain', mats: [['', D]], shape: 'T × 1,152' },
    { id: 'q', name: 'Q, K, V', op: '<code>Q = X·W_q</code> (1,152 × 1,024), split into 4 heads; <code>K = X·W_k</code>, <code>V = X·W_v</code> (1,152 × 256 each), one head shared by all four; QK-norm and RoPE applied', mats: [['Q₀', HD], ['Q₁', HD], ['Q₂', HD], ['Q₃', HD], ['K', HD], ['V', HD]], shape: '4 × (T × 256) · T × 256 · T × 256' },
    { id: 's', name: 'Scores, softmaxed', op: '<code>softmax(Q·Kᵀ / √256)</code> per head, causal: a token sees only itself and earlier tokens' + '', mats: [['h₀', 0, 'scores'], ['h₁', 0, 'scores'], ['h₂', 0, 'scores'], ['h₃', 0, 'scores']], shape: '4 × (T × T)' },
    { id: 'o', name: 'Heads × V', op: 'each head\'s weights times V, the four results side by side', mats: [['', H * HD]], shape: 'T × 1,024' },
    { id: 'op', name: 'Out projection', op: '<code>· W_o</code> (1,024 × 1,152), then a post-norm', mats: [['', D]], shape: 'T × 1,152' },
    { id: 'a1', name: 'Residual + attention', op: 'added to the residual stream', mats: [['', D]], shape: 'T × 1,152' },
    { id: 'n2', name: 'RMSNorm', op: 'each row scaled again before the MLP', mats: [['', D]], shape: 'T × 1,152' },
    { id: 'gate', name: 'Gate', op: '<code>X·W_gate</code> (1,152 × 6,912)', mats: [['', F]], shape: 'T × 6,912' },
    { id: 'up', name: 'Up', op: '<code>X·W_up</code> (1,152 × 6,912)', mats: [['', F]], shape: 'T × 6,912' },
    { id: 'act', name: 'Neurons', op: '<code>GELU(gate) ⊙ up</code>, elementwise. Each column is one neuron; this is what you poke', mats: [['', F]], shape: 'T × 6,912' },
    { id: 'down', name: 'Down projection', op: '<code>· W_down</code> (6,912 × 1,152), then a post-norm', mats: [['', D]], shape: 'T × 1,152' },
    { id: 'a2', name: 'Residual out', op: 'added to the residual stream; the next layer starts here', mats: [['X', D]], shape: 'T × 1,152' },
    { id: 'fn', once: true, name: 'Final RMSNorm', op: 'after layer 25', mats: [['', D]], shape: 'T × 1,152' },
    { id: 'logits', once: true, name: 'Logits', op: '<code>· Eᵀ</code>, the embedding table reused (1,152 × 262,144). Drawn: the first 4,096 of 262,144 columns', mats: [['', 4096, 'elided']], shape: 'T × 262,144' },
    { id: 'p', once: true, name: 'Softmax', op: 'over the vocabulary, last row only: the next-token probabilities', mats: [['', 4096, 'lastrow']], shape: '1 × 262,144' },
  ];
}

const group = new THREE.Group(); scene.add(group);
const EDGE = new THREE.LineBasicMaterial({ color: 0xB4B2A9 }), SELE = new THREE.LineBasicMaterial({ color: 0xBA7517 });
let S = [], pick = [];
const GAP = 2.6, MGAP = 0.25;

function build() {
  while (group.children.length) group.remove(group.children[0]);
  pick = []; S = stages(); let x = 0;
  S.forEach((st, si) => {
    st.x = x; let z = 0; st.edges = [];
    st.mats.forEach(([lab, cols, kind], mi) => {
      const seed = layer * 997 + si * 31 + mi;
      let w, rows = T, tex;
      if (kind === 'scores') {
        w = T * SQ; const r = rng(seed);
        const row = [...Array(T)].map((_, i) => { const a = [...Array(i + 1)].map((_, j) => j === 0 ? 1.4 : (i === 5 && j === 4 && layer > 8 ? 1.8 : r() * 0.6)); const s = a.reduce((p, q) => p + q, 0); return a.map(v => v / s); });
        tex = texture(T, T, (i, j) => j > i ? null : Math.min(1, row[i][j] * 1.6));
      } else if (kind === 'lastrow') {
        w = cols * COL; const g = sparse(seed, 0.5);
        tex = texture(cols, T, (i, c) => i < T - 1 ? null : (c === 1977 ? 1 : (c % 97 === 3 ? 0.3 : g())));
      } else {
        w = cols * COL; tex = texture(cols, T, sparse(seed, 1.3 + (st.id === 'act' ? 0.8 : 0)));
      }
      const geo = new THREE.PlaneGeometry(rows * ROW, w);
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide }));
      // Flat on the ground: rows run along x (flow), columns along -z (depth).
      m.rotation.x = -Math.PI / 2;
      m.position.set(x + rows * ROW / 2, 0, -(z + w / 2)); m.userData.stage = st; group.add(m); pick.push(m);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), EDGE); e.rotation.copy(m.rotation); e.position.copy(m.position); group.add(e); st.edges.push(e);
      if (kind === 'elided') { const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints([0, 1, 2].map(k => new THREE.Vector3(x + rows * ROW / 2, 0, -(z + w + 0.4 + k * 0.35)))), new THREE.PointsMaterial({ color: 0x888780, size: 3, sizeAttenuation: false })); group.add(dots); }
      z += w + MGAP;
    });
    st.depth = z; x += T * ROW + GAP;
  });
  // One baseline under the flow, so the order reads.
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.4, 0, 0.35), new THREE.Vector3(x - GAP + 0.4, 0, 0.35)]), EDGE));
  makeLabels();
  document.getElementById('layerName').textContent = 'Layer ' + layer;
  document.getElementById('layerKind').textContent = (layer + 1) % 6 === 0 ? 'global attention: every token visible' : 'local attention: 512 tokens back';
  if (selected) select(S.find(s => s.id === selected.id), false); else overview();
}

let labelEls = [];
function makeLabels() {
  labelsEl.replaceChildren(); labelEls = S.map(st => { const d = document.createElement('div'); d.className = st.once ? 'once' : ''; d.innerHTML = `<b>${st.name}${st.once ? ' (once)' : ''}</b><br>${st.shape}`; labelsEl.appendChild(d); return d; });
}
function placeLabels() {
  const w = stage.clientWidth, h = stage.clientHeight;
  S.forEach((st, i) => { const v = new THREE.Vector3(st.x + T * ROW / 2, 0, 0.6).project(camera); labelEls[i].style.left = ((v.x + 1) / 2 * w) + 'px'; labelEls[i].style.top = ((1 - v.y) / 2 * h + (i % 2) * 30) + "px"; labelEls[i].style.fontWeight = selected === st ? '500' : ''; });
}
function overview() {
  panel.innerHTML = `<h2>Layer ${layer} of 25</h2><p class="shape">6 tokens · Gemma 3 1B</p>
  <p>Every step is a matrix: one row per token, one column per dimension, drawn at its real width. The residual is 1,152 wide; attention splits into 4 heads of 256; the MLP widens to 6,912 and comes back to 1,152.</p>
  <p>The embedding runs once before layer 0; the final norm, logits and softmax run once after layer 25. The steps between repeat in every layer.</p>
  <p class="hint">Click a matrix to open it. ‹ › or the arrow keys step through layers. Scroll to zoom, drag to pan.</p>`;
}
function select(st, move = true) {
  selected = st; S.forEach(s => s.edges.forEach(e => e.material = s === st ? SELE : EDGE));
  panel.innerHTML = `<h2>${st.name}${st.once ? ' (once)' : ''}</h2><p class="shape">${st.shape}</p><p>${st.op}.</p>` +
    (st.id === 'act' ? `<p class="hint">Next: pick a column (a neuron) to see the tokens it pushes toward, and set, scale or zero it.</p>` : '') +
    `<button id="back">Back to the whole pass</button>`;
  document.getElementById('back').onclick = () => { selected = null; S.forEach(s => s.edges.forEach(e => e.material = EDGE)); overview(); focus(null); };
  if (move) focus(st);
}
const ISO = new THREE.Vector3(0.18, 1, 1).normalize().multiplyScalar(80);
let goal;
function focus(st) {
  if (!st) { const last = S[S.length - 1]; const extent = last.x + T * ROW; goal = { t: new THREE.Vector3(extent / 2, 0, -6), zoom: Math.min(1, (camera.right - camera.left) / (extent * 1.12)) }; return; }
  const span = Math.max(st.depth, T * ROW); goal = { t: new THREE.Vector3(st.x + T * ROW / 2, 0, -st.depth / 2), zoom: Math.min(4, 22 / span) };
}
build(); focus(null); controls.target.copy(goal.t); camera.zoom = goal.zoom;
controls.addEventListener('start', () => { goal = null; });

const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(); let downAt = null;
renderer.domElement.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect(); mouse.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(mouse, camera); const hit = ray.intersectObjects(pick)[0]; if (hit) select(hit.object.userData.stage);
});
const step = d => { layer = Math.max(0, Math.min(L - 1, layer + d)); build(); };
document.getElementById('prev').onclick = () => step(-1); document.getElementById('next').onclick = () => step(1);
addEventListener('keydown', e => { if (e.key === 'ArrowLeft') step(-1); if (e.key === 'ArrowRight') step(1); if (e.key === 'Escape' && selected) document.getElementById('back').click(); });

(function loop() {
  if (goal) { controls.target.lerp(goal.t, 0.12); camera.zoom += (goal.zoom - camera.zoom) * 0.12; camera.updateProjectionMatrix(); }
  camera.position.copy(controls.target).add(ISO); camera.lookAt(controls.target); controls.update();
  renderer.render(scene, camera); placeLabels(); requestAnimationFrame(loop);
})();
