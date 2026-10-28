// Frizzle layer-deck concept. Made-up values; nothing here runs a model.
(() => {
  const LAYERS = 26, COLS = 96, ROWS = 72, N = COLS * ROWS;
  const CELL = 6, PAD = 16, HEAD = 34;
  const TW = COLS * CELL + PAD * 2, TH = HEAD + ROWS * CELL + PAD;
  const CARD_W = 4, CARD_H = CARD_W * TH / TW;
  const isGlobal = (l) => l % 6 === 5;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Deterministic made-up activations: mostly near zero, a few strong ones.
  function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  const base = [];
  for (let l = 0; l < LAYERS; l++) {
    const r = rng(1000 + l * 7919), v = new Float32Array(N), scale = 0.6 + l * 0.12;
    for (let i = 0; i < N; i++) {
      const g = (r() + r() + r() - 1.5) * 0.25;
      v[i] = r() < 0.018 ? (r() < 0.8 ? 1 : -0.6) * (1 + r() * 5) * scale : g * scale;
    }
    base.push(v);
  }
  const WORDS = [' Whiskers', ' Luna', ' Milo', ' Oliver', ' Pepper', ' Shadow', ' Mochi', ' Ginger', ' Biscuit', ' Tiger',
    ' Smokey', ' Cleo', ' Jasper', ' Nala', 'ington', ' Sir', ' Captain', ' Professor', ' Mr', ' Duchess'];
  const pushFor = (l, i) => { const r = rng(l * 100003 + i); return [0, 1, 2].map(() => WORDS[Math.floor(r() * WORDS.length)]); };

  // The edit set: what the app owns and sends whole with every request.
  const pins = new Map(); // key "l:i" -> {l, i, mode, value}
  const key = (l, i) => l + ':' + i;
  pins.set(key(6, 2417), { l: 6, i: 2417, mode: 'set', value: 8 });
  pins.set(key(14, 5120), { l: 14, i: 5120, mode: 'zero', value: 0 });
  const edited = (l, i) => {
    const p = pins.get(key(l, i)), v = base[l][i];
    if (!p) return v;
    return p.mode === 'set' ? p.value : p.mode === 'scale' ? v * p.value : 0;
  };

  // Scene.
  const host = document.getElementById('deck'), tip = document.getElementById('tip');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x0d1422);
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);

  const cards = [];
  for (let l = 0; l < LAYERS; l++) {
    const canvas = document.createElement('canvas'); canvas.width = TW; canvas.height = TH;
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(CARD_W, CARD_H), mat);
    mesh.userData.layer = l;
    scene.add(mesh);
    cards.push({ canvas, tex, mat, mesh });
  }

  const mix = (a, b, t) => a.map((x, k) => Math.round(x + (b[k] - x) * t));
  const ZERO = [24, 35, 58], POS = [255, 195, 90], NEG = [159, 192, 232];
  function cellColor(v, vmax) {
    const t = Math.min(1, Math.pow(Math.abs(v) / vmax, 0.6));
    return 'rgb(' + mix(ZERO, v >= 0 ? POS : NEG, t).join(',') + ')';
  }
  const vmaxOf = (l) => { let m = 0; for (let i = 0; i < N; i++) m = Math.max(m, Math.abs(base[l][i])); return m * 0.8; };
  const VMAX = base.map((_, l) => vmaxOf(l));

  let selected = { l: 6, i: 2417 };
  function drawCard(l) {
    const { canvas, tex } = cards[l], g = canvas.getContext('2d');
    g.fillStyle = '#121b2c'; g.fillRect(0, 0, TW, TH);
    g.strokeStyle = '#2b3a55'; g.lineWidth = 2; g.strokeRect(1, 1, TW - 2, TH - 2);
    g.fillStyle = '#eee6d3'; g.font = '500 17px "Plex Mono", monospace'; g.textBaseline = 'middle';
    g.fillText('Layer ' + l, PAD, HEAD / 2 + 2);
    g.fillStyle = isGlobal(l) ? '#e2b866' : '#9aa0a8'; g.font = '14px "Plex Mono", monospace';
    g.fillText(isGlobal(l) ? 'global attention' : 'local attention', PAD + 96, HEAD / 2 + 2);
    const n = [...pins.values()].filter((p) => p.l === l).length;
    if (n) { g.fillStyle = '#ffcf73'; g.textAlign = 'right'; g.fillText('● ' + n + ' pinned', TW - PAD, HEAD / 2 + 2); g.textAlign = 'left'; }
    for (let i = 0; i < N; i++) {
      const x = PAD + (i % COLS) * CELL, y = HEAD + Math.floor(i / COLS) * CELL;
      g.fillStyle = cellColor(edited(l, i), VMAX[l]);
      g.fillRect(x, y, CELL - 1, CELL - 1);
    }
    for (const p of pins.values()) if (p.l === l) {
      const x = PAD + (p.i % COLS) * CELL, y = HEAD + Math.floor(p.i / COLS) * CELL;
      g.strokeStyle = '#ffcf73'; g.lineWidth = 2; g.strokeRect(x - 3, y - 3, CELL + 5, CELL + 5);
    }
    if (selected.l === l) {
      const x = PAD + (selected.i % COLS) * CELL, y = HEAD + Math.floor(selected.i / COLS) * CELL;
      g.strokeStyle = '#f1ebdc'; g.lineWidth = 1; g.strokeRect(x - 5.5, y - 5.5, CELL + 10, CELL + 10);
    }
    tex.needsUpdate = true;
  }

  // Deck placement as a function of d = layer − position (fractional while sliding).
  function place(card, d) {
    const m = card.mesh;
    let x, y, z, rot, op, dim;
    if (d >= 0) {            // ahead: receding into the deck
      x = d * 0.30; y = d * 0.17; z = -d * 0.7; rot = 0;
      op = d < 1 ? 1 - d * 0.35 : Math.max(0.06, 0.65 - (d - 1) * 0.07);
      dim = d < 1 ? 1 - d * 0.55 : 0.42;
    } else if (d > -1) {     // the card leaving: slides out to the left
      const t = -d;
      x = -t * CARD_W * 1.08; y = -t * 0.12; z = t * 0.35; rot = t * 0.45;
      op = 1 - t * 0.88; dim = 1 - t * 0.5;
    } else {                 // passed: a thin stack at the left edge
      const k = -d - 1;
      x = -CARD_W * 1.08 - k * 0.05; y = -0.12; z = 0.35 - k * 0.02; rot = 0.45;
      op = Math.max(0, 0.12 - k * 0.006); dim = 0.5;
    }
    m.position.set(x, y, z); m.rotation.y = rot;
    card.mat.opacity = op; card.mat.color.setScalar(dim);
    m.renderOrder = Math.round(1000 - Math.abs(d) * 10);
    m.visible = op > 0.01 && d < 12;
  }

  let pos = 6, target = 6;
  function layout() { cards.forEach((c, l) => place(c, l - pos)); }

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Fit the front card with room for the deck behind it.
    const vFov = camera.fov * Math.PI / 180;
    const needH = CARD_H * 1.32, needW = CARD_W * 1.5;
    const dist = Math.max(needH / (2 * Math.tan(vFov / 2)), needW / (2 * Math.tan(vFov / 2) * camera.aspect));
    camera.position.set(0.1, 0.25, dist);
    camera.lookAt(0.55, 0.38, -1.2);
    camera.updateProjectionMatrix();
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.5, (now - last) / 1000); last = now;
    const diff = target - pos;
    // Time-based easing: settles in about half a second at any frame rate.
    pos = Math.abs(diff) < 0.001 || reduced ? target : pos + diff * (1 - Math.exp(-dt / 0.09));
    layout();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  // Navigation.
  function go(l) {
    target = Math.max(0, Math.min(LAYERS - 1, l));
    tip.style.display = 'none';
    renderSide();
  }
  let wheelAcc = 0;
  host.addEventListener('wheel', (e) => {
    e.preventDefault();
    wheelAcc += e.deltaY;
    if (Math.abs(wheelAcc) > 70) { go(target + Math.sign(wheelAcc)); wheelAcc = 0; }
  }, { passive: false });
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (['ArrowDown', 'ArrowRight', 'PageDown'].includes(e.key)) { go(target + 1); e.preventDefault(); }
    if (['ArrowUp', 'ArrowLeft', 'PageUp'].includes(e.key)) { go(target - 1); e.preventDefault(); }
  });

  // Picking neurons on the front card only.
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  function pick(e) {
    if (Math.abs(target - pos) > 0.05) return null;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(cards[target].mesh)[0];
    if (!hit) return null;
    const px = hit.uv.x * TW - PAD, py = (1 - hit.uv.y) * TH - HEAD;
    const c = Math.floor(px / CELL), rw = Math.floor(py / CELL);
    if (c < 0 || c >= COLS || rw < 0 || rw >= ROWS) return null;
    return { l: target, i: rw * COLS + c, x: e.clientX - r.left, y: e.clientY - r.top };
  }
  renderer.domElement.addEventListener('mousemove', (e) => {
    const p = pick(e);
    if (!p) { tip.style.display = 'none'; renderer.domElement.style.cursor = 'default'; return; }
    renderer.domElement.style.cursor = 'crosshair';
    const v = edited(p.l, p.i);
    tip.textContent = '#' + p.i + '  ' + (v >= 0 ? '+' : '') + v.toFixed(2) + (pins.has(key(p.l, p.i)) ? '  pinned' : '');
    tip.style.left = (p.x + 14) + 'px'; tip.style.top = (p.y + 10) + 'px'; tip.style.display = 'block';
  });
  renderer.domElement.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  renderer.domElement.addEventListener('click', (e) => { const p = pick(e); if (p) select(p.l, p.i); });

  function select(l, i) {
    const prev = selected.l; selected = { l, i };
    drawCard(prev); drawCard(l);
    if (l !== target) go(l); else renderSide();
  }

  // Side panel.
  const $ = (id) => document.getElementById(id);
  const fmt = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
  const tokenHtml = (t) => t.startsWith(' ') ? '<span class="sp">␣</span>' + t.slice(1) : '<span class="sp">…</span>' + t;

  function renderScrub() {
    const s = $('scrub'); s.innerHTML = '';
    for (let l = 0; l < LAYERS; l++) {
      const b = document.createElement('button');
      b.textContent = l;
      b.title = 'Layer ' + l + (isGlobal(l) ? ', global attention' : '');
      b.className = (l === target ? 'on ' : '') + (isGlobal(l) ? 'global ' : '') + ([...pins.values()].some((p) => p.l === l) ? 'pin' : '');
      b.onclick = () => go(l);
      s.appendChild(b);
    }
  }

  function renderSide() {
    const l = target;
    $('strongestTitle').textContent = 'Strongest in layer ' + l;
    const idx = Array.from({ length: N }, (_, i) => i).sort((a, b) => Math.abs(edited(l, b)) - Math.abs(edited(l, a))).slice(0, 8);
    const top = Math.abs(edited(l, idx[0]));
    $('list').innerHTML = idx.map((i) => {
      const v = edited(l, i), w = Math.round(Math.abs(v) / top * 100);
      return '<div class="item' + (selected.l === l && selected.i === i ? ' sel' : '') + '" data-i="' + i + '"><span>#' + i + (pins.has(key(l, i)) ? ' <span style="color:var(--edit)">●</span>' : '') + '</span><div class="bar"><i class="' + (v < 0 ? 'neg' : '') + '" style="width:' + w + '%"></i></div><span>' + fmt(v) + '</span></div>';
    }).join('');
    $('list').querySelectorAll('.item').forEach((el) => el.onclick = () => select(l, +el.dataset.i));

    const s = selected, p = pins.get(key(s.l, s.i));
    $('card').innerHTML =
      '<div class="title">Neuron #' + s.i + ' <span class="sp">· layer ' + s.l + '</span></div>' +
      '<div class="row mono"><span>Now</span><span>' + fmt(edited(s.l, s.i)) + '</span></div>' +
      '<div class="row mono"><span>Without edits</span><span>' + fmt(base[s.l][s.i]) + '</span></div>' +
      '<div class="row"><span>Pushes toward</span><span></span></div>' +
      '<div class="push mono">' + pushFor(s.l, s.i).map((t) => '<span>' + tokenHtml(t) + '</span>').join('') + '</div>' +
      '<div class="btns"><input id="val" class="mono" value="' + (p && p.mode !== 'zero' ? p.value : 4) + '">' +
      '<button data-m="set">Set</button><button data-m="scale">Scale ×</button><button data-m="zero">Zero</button>' +
      (p ? '<button data-m="unpin">Unpin</button>' : '') + '</div>';
    $('card').querySelectorAll('button').forEach((b) => b.onclick = () => {
      const m = b.dataset.m, k = key(s.l, s.i), v = parseFloat($('val').value);
      if (m === 'unpin') pins.delete(k);
      else if (m === 'zero') pins.set(k, { l: s.l, i: s.i, mode: 'zero', value: 0 });
      else if (Number.isFinite(v)) pins.set(k, { l: s.l, i: s.i, mode: m, value: v });
      drawCard(s.l); renderSide();
    });

    const list = [...pins.values()].sort((a, b) => a.l - b.l || a.i - b.i);
    $('pins').innerHTML = list.length ? list.map((q) =>
      '<div><a data-l="' + q.l + '" data-i="' + q.i + '"><span class="dot">●</span> L' + q.l + ' #' + q.i + '</a><span>' +
      (q.mode === 'set' ? 'set ' + q.value : q.mode === 'scale' ? '× ' + q.value : 'zero') + '</span><button data-k="' + key(q.l, q.i) + '" aria-label="Unpin">×</button></div>').join('')
      : '<span class="sp">Nothing pinned.</span>';
    $('pins').querySelectorAll('a').forEach((a) => a.onclick = () => select(+a.dataset.l, +a.dataset.i));
    $('pins').querySelectorAll('button').forEach((b) => b.onclick = () => { const [pl] = b.dataset.k.split(':'); pins.delete(b.dataset.k); drawCard(+pl); renderSide(); });

    renderOut();
    renderScrub();
  }

  // A pretend next-token readout that moves with the pins.
  function renderOut() {
    const before = [[' How', 2.1], [' Here', 1.3], [' Oh', 0.6], [' What', 0.4], [' Naming', 0.2], [' Whiskers', -0.4], [' Luna', -0.6]];
    const after = before.map(([t, z]) => [t, z]);
    for (const p of pins.values()) {
      const shift = (edited(p.l, p.i) - base[p.l][p.i]) * 0.35;
      for (const t of pushFor(p.l, p.i)) {
        const e = after.find((a) => a[0] === t);
        if (e) e[1] += shift; else after.push([t, -1 + shift]);
      }
    }
    const soft = (rows) => { const m = Math.max(...rows.map((r) => r[1])), s = rows.reduce((a, r) => a + Math.exp(r[1] - m), 0); return new Map(rows.map((r) => [r[0], Math.exp(r[1] - m) / s])); };
    const pb = soft(before), pa = soft(after);
    const rows = [...pa.keys()].sort((a, b) => pa.get(b) - pa.get(a)).slice(0, 6);
    $('out').innerHTML = rows.map((t) =>
      '<div class="row"><span>' + tokenHtml(t) + '</span><div class="bar"><i class="before" style="width:' + Math.round((pb.get(t) || 0) * 100) + '%"></i><i style="width:' + Math.round(pa.get(t) * 100) + '%;opacity:.85;height:3px;top:2px"></i></div><span>' + Math.round(pa.get(t) * 100) + '%</span></div>').join('');
  }

  document.fonts.ready.then(() => { for (let l = 0; l < LAYERS; l++) drawCard(l); });
  for (let l = 0; l < LAYERS; l++) drawCard(l);
  new ResizeObserver(resize).observe(host);
  resize(); renderSide(); requestAnimationFrame(frame);
})();
