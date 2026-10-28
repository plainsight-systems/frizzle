// Concept: the MLP workbench for one layer. All values, labels and effects are made up.
const tokens = ['<bos>', 'The', '▁capital', '▁of', '▁France', '▁is'];
const T = tokens.length, F = 6912, COLS = 96, ROWS = 72, CELL = 7, L = 26;
let layer = 14, tok = 5, sel = 1768;
let edits = [];   // { layer, neuron, mode: 'set'|'scale'|'zero', value, scope: 'this'|'all', token }

function rng(s) { return function () { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const cache = {};
function base(l, t) {
  const k = l + ':' + t; if (cache[k]) return cache[k];
  const r = rng(l * 7919 + t * 104729 + 3), a = new Float32Array(F);
  for (let i = 0; i < F; i++) a[i] = (r() < 0.35 ? -1 : 1) * Math.pow(r(), 14) * 1.7;
  if (l === 14) { a[1768] = t === 5 ? 2.3 : (t === 4 ? 1.1 : 0.05 * t); a[4056] = t === 5 ? -2.3 : -0.2; }
  return (cache[k] = a);
}
function editFor(l, t, i) { let e = null; edits.forEach(x => { if (x.layer === l && x.neuron === i && (x.scope === 'all' || x.token === t)) e = x; }); return e; }
function value(l, t, i) { const v = base(l, t)[i], e = editFor(l, t, i); if (!e) return v; return e.mode === 'zero' ? 0 : e.mode === 'set' ? e.value : v * e.value; }
const fmt = v => (v >= 0 ? ' ' : '−') + Math.abs(v).toFixed(2);
const grey = v => 246 - Math.min(1, Math.abs(v) / 2.4) * 190;

function drawFolded(canvas, l, t, cell, outline) {
  const c = canvas.getContext('2d'); c.fillStyle = '#fff'; c.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < F; i++) { const g = grey(value(l, t, i)); c.fillStyle = `rgb(${g},${g - 2},${g - 8})`; c.fillRect((i % COLS) * cell, Math.floor(i / COLS) * cell, cell, cell); }
  if (!outline) return;
  c.lineWidth = 1.5;
  edits.forEach(e => { if (e.layer !== l || (e.scope === 'this' && e.token !== t)) return; c.strokeStyle = '#BA7517'; c.strokeRect((e.neuron % COLS) * cell + 0.75, Math.floor(e.neuron / COLS) * cell + 0.75, cell - 1.5, cell - 1.5); });
  c.strokeStyle = '#2C2C2A'; c.lineWidth = 2; c.strokeRect((sel % COLS) * cell - 1, Math.floor(sel / COLS) * cell - 1, cell + 2, cell + 2);
}

const grid = document.getElementById('grid'), tip = document.getElementById('tip');
function renderTokens() {
  const box = document.getElementById('tokens'); box.replaceChildren();
  tokens.forEach((name, t) => {
    const b = document.createElement('button'); b.className = 'tok' + (t === tok ? ' on' : ''); b.setAttribute('aria-label', 'Show neurons for token ' + name);
    const cv = document.createElement('canvas'); cv.width = COLS; cv.height = ROWS; drawFolded(cv, layer, t, 1, false);
    const s = document.createElement('span'); s.textContent = name; b.append(cv, s); b.onclick = () => { tok = t; render(); }; box.appendChild(b);
  });
}
function renderList() {
  const a = Array.from({ length: F }, (_, i) => i).sort((i, j) => Math.abs(value(layer, tok, j)) - Math.abs(value(layer, tok, i))).slice(0, 12);
  const max = Math.abs(value(layer, tok, a[0])) || 1;
  const list = document.getElementById('list'); list.replaceChildren();
  a.forEach(i => {
    const v = value(layer, tok, i), d = document.createElement('div'); d.className = 'item' + (i === sel ? ' sel' : '');
    d.innerHTML = `<span>#${i}${editFor(layer, tok, i) ? ' <span class="ed">•</span>' : ''}</span><div class="bar"><i style="width:${Math.round(Math.abs(v) / max * 100)}%"></i></div><span>${fmt(v)}</span>`;
    d.onclick = () => { sel = i; render(); }; list.appendChild(d);
  });
}
const vocab = ['▁Paris', '▁France', '▁French', '▁capital', '▁city', '▁Lyon', '▁Europe', '▁the', '▁of', '▁river', '▁king', '▁wine', '▁Berlin', '▁London', '▁Rome', '▁is', '▁was', '▁located', '▁known', '▁famous'];
function pushes(i) { if (i === 1768) return ['▁Paris', '▁French', '▁France', '▁Lyon', '▁Seine']; const r = rng(i + 5); return [0, 1, 2, 3, 4].map(() => vocab[Math.floor(r() * vocab.length)]).filter((v, k, s) => s.indexOf(v) === k); }
let mode = 'scale';
function renderCard() {
  const vals = tokens.map((_, t) => value(layer, t, sel)), max = Math.max(...vals.map(Math.abs), 0.01);
  const e = editFor(layer, tok, sel);
  document.getElementById('card').innerHTML = `
  <div>
    <h2>Neuron #${sel}</h2><div class="muted">Layer ${layer} · row ${Math.floor(sel / COLS)}, column ${sel % COLS} of the fold</div>
    <div style="margin-top:10px" class="muted">Value on each token (its column in the T × 6,912 matrix)</div>
    <div class="cols">${vals.map((v, t) => `<div class="c${t === tok ? ' cur' : ''}" title="${tokens[t]}: ${fmt(v)}"><i style="height:${Math.max(2, Math.abs(v) / max * 100)}%"></i></div>`).join('')}</div>
    <div class="colnames">${tokens.map(n => `<span>${n}</span>`).join('')}</div>
    <div style="margin-top:10px" class="muted">When it fires, it pushes the output toward</div>
    <div class="push">${pushes(sel).map(p => `<span>${p}</span>`).join('')}</div>
  </div>
  <div class="edit">
    <h3>Change it</h3>
    <div class="muted">Now ${fmt(value(layer, tok, sel))} on ${tokens[tok]}${e ? ' (edited)' : ''}</div>
    <div class="modes"><button data-m="set" class="${mode === 'set' ? 'on' : ''}">Set</button><button data-m="scale" class="${mode === 'scale' ? 'on' : ''}">Scale ×</button><button data-m="zero" class="${mode === 'zero' ? 'on' : ''}">Zero</button></div>
    <div class="val" style="${mode === 'zero' ? 'visibility:hidden' : ''}"><input id="amt" type="number" step="0.1" value="${mode === 'scale' ? 3 : 4}"><span class="muted">${mode === 'scale' ? 'times its value' : 'new value'}</span></div>
    <div><label><input type="radio" name="scope" value="this" checked> This token</label><label><input type="radio" name="scope" value="all"> All tokens</label></div>
    <div style="margin-top:8px;display:flex;gap:6px"><button id="apply">Apply</button>${e ? '<button id="undo">Remove edit</button>' : ''}</div>
  </div>`;
  document.querySelectorAll('.modes button').forEach(b => b.onclick = () => { mode = b.dataset.m; renderCard(); });
  document.getElementById('apply').onclick = () => {
    const scope = document.querySelector('input[name=scope]:checked').value, amt = parseFloat(document.getElementById('amt').value);
    if (mode !== 'zero' && !Number.isFinite(amt)) return;
    edits = edits.filter(x => !(x.layer === layer && x.neuron === sel && (x.scope === 'all' || x.token === tok)));
    edits.push({ layer, neuron: sel, mode, value: mode === 'zero' ? 0 : amt, scope, token: tok }); render();
  };
  const u = document.getElementById('undo'); if (u) u.onclick = () => { edits = edits.filter(x => x !== e); render(); };
}
function renderEdits() {
  const box = document.getElementById('edits');
  if (!edits.length) { box.innerHTML = '<div class="muted">None yet.</div>'; return; }
  box.replaceChildren(...edits.map(e => { const d = document.createElement('div'); const what = e.mode === 'zero' ? 'zero' : e.mode === 'set' ? 'set ' + e.value : '× ' + e.value; d.innerHTML = `<span>L${e.layer} #${e.neuron} ${what} · ${e.scope === 'all' ? 'all tokens' : tokens[e.token]}</span>`; const b = document.createElement('button'); b.textContent = '×'; b.setAttribute('aria-label', 'Remove edit'); b.onclick = () => { edits = edits.filter(x => x !== e); render(); }; d.appendChild(b); return d; }));
}
function renderOut() {
  // Simulated: neuron 1768 at layer 14 drives "▁Paris"; anything else nudges the distribution a little.
  const before = { '▁Paris': 0.62, '▁a': 0.11, '▁the': 0.07, '▁located': 0.04, '▁Lyon': 0.01 };
  const after = { ...before };
  edits.forEach(e => { const k = e.layer === 14 && e.neuron === 1768 ? 1 : 0.15; const v = value(e.layer, 5, e.neuron), b = base(e.layer, 5)[e.neuron]; const d = Math.tanh((v - b) * k); after['▁Paris'] = Math.max(0.01, Math.min(0.97, after['▁Paris'] + d * 0.3)); after['▁Lyon'] = Math.max(0.005, after['▁Lyon'] + Math.max(0, -d) * 0.15); });
  const s = Object.values(after).reduce((a, b) => a + b, 0); Object.keys(after).forEach(k => after[k] /= s);
  document.getElementById('out').innerHTML = Object.keys(before).map(k => `<div class="row"><span>${k}</span><div class="bar"><i class="before" style="width:${before[k] * 100}%"></i><i style="width:${after[k] * 100}%;opacity:${edits.length ? 1 : 0}"></i></div><span>${Math.round(before[k] * 100)}%${edits.length ? ' → ' + Math.round(after[k] * 100) + '%' : ''}</span></div>`).join('');
}
function render() {
  document.getElementById('layerName').textContent = 'Layer ' + layer;
  document.getElementById('tokName').textContent = tokens[tok];
  drawFolded(grid, layer, tok, CELL, true); renderTokens(); renderList(); renderCard(); renderEdits(); renderOut();
}
function cellAt(ev) { const r = grid.getBoundingClientRect(); const x = Math.floor((ev.clientX - r.left) / r.width * COLS), y = Math.floor((ev.clientY - r.top) / r.height * ROWS); return x < 0 || y < 0 || x >= COLS || y >= ROWS ? null : y * COLS + x; }
grid.addEventListener('mousemove', ev => { const i = cellAt(ev); if (i === null) { tip.style.display = 'none'; return; } const r = grid.getBoundingClientRect(); tip.style.display = 'block'; tip.textContent = `#${i}  ${fmt(value(layer, tok, i))}`; tip.style.left = (ev.clientX - r.left + 12) + 'px'; tip.style.top = (ev.clientY - r.top + 12) + 'px'; });
grid.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
grid.addEventListener('click', ev => { const i = cellAt(ev); if (i !== null) { sel = i; render(); } });
document.getElementById('go').onclick = () => { const n = parseInt(document.getElementById('find').value, 10); if (n >= 0 && n < F) { sel = n; render(); } };
document.getElementById('find').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('go').click(); });
const step = d => { layer = Math.max(0, Math.min(L - 1, layer + d)); render(); };
document.getElementById('prev').onclick = () => step(-1); document.getElementById('next').onclick = () => step(1);
render();
