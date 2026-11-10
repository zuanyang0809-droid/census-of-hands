// iPad side: collect handwriting samples for training.
//
// Shows one symbol at a time in random order (weighted toward the symbols
// that still need samples). Write it in the box; after a short pause with the
// pencil lifted, the sample is saved and the next prompt appears. The pause
// lets multi-stroke symbols like 4 and 7 finish before saving.
//
// Saved: raw strokes in box pixels, [x, y, t ms since first touch, pressure].

import { LABELS, TARGET_PER_LABEL } from './labels.js';
import { INK, autoFit, debugOverlay, drawStroke, lockPage, pressureOf, round, trackStrokes } from './ink.js';

const SAVE_DELAY_MS = 700;
const MIN_POINTS = 3; // ignore accidental taps

lockPage();

// Local date and time of this sitting, e.g. "2026-11-10 14:05-k3x9".
const session = (() => {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return `${stamp}-${Math.random().toString(36).slice(2, 6)}`;
})();
const box = document.getElementById('box');
const boxWrap = document.getElementById('box-wrap');
const ctx = box.getContext('2d');
const glyphEl = document.getElementById('glyph');
const hintEl = document.getElementById('hint');
const progressEl = document.getElementById('progress');
const timerEl = document.getElementById('timer');
const toastEl = document.getElementById('toast');

const counts = Object.fromEntries(LABELS.map((l) => [l.label, 0]));
let label = null;
let lastLabel = null;
let strokes = [];
let current = null;
let pointerType = '';
let saveTimer = null;
let history = []; // samples saved in this session, newest last
let size = { w: 1, h: 1 };
let dirty = true;

function toast(text) {
  toastEl.textContent = text;
}

function pickLabel() {
  let pool = LABELS.map((l) => ({ label: l.label, weight: Math.max(0, TARGET_PER_LABEL - counts[l.label]) }));
  if (pool.every((p) => p.weight === 0)) pool = pool.map((p) => ({ ...p, weight: 1 })); // targets met: keep going evenly
  const others = pool.filter((p) => p.label !== lastLabel && p.weight > 0);
  if (others.length) pool = others; // avoid the same symbol twice in a row
  let r = Math.random() * pool.reduce((sum, p) => sum + p.weight, 0);
  for (const p of pool) {
    r -= p.weight;
    if (r < 0) return p.label;
  }
  return pool[pool.length - 1].label;
}

function showPrompt(next) {
  label = next;
  const info = LABELS.find((l) => l.label === label);
  glyphEl.textContent = label === 'other' ? '✕' : label;
  hintEl.textContent = info.hint;
  renderProgress();
}

function renderProgress() {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  progressEl.innerHTML = '';
  for (const { label: l } of LABELS) {
    const cell = document.createElement('div');
    cell.className = `cell${l === label ? ' now' : ''}`;
    const pct = Math.min(100, (counts[l] / TARGET_PER_LABEL) * 100);
    cell.innerHTML = `<b></b><span class="muted">${counts[l]}/${TARGET_PER_LABEL}</span><div class="meter"><i style="width:${pct}%"></i></div>`;
    cell.querySelector('b').textContent = l === 'other' ? '✕' : l;
    progressEl.append(cell);
  }
  const all = LABELS.length * TARGET_PER_LABEL;
  const status = document.createElement('div');
  status.className = 'cell muted';
  status.style.width = 'auto';
  status.innerHTML = `<b>${total}</b>of ${all}`;
  progressEl.append(status);
}

function clearBox() {
  strokes = [];
  current = null;
  cancelSave();
  dirty = true;
}

function cancelSave() {
  clearTimeout(saveTimer);
  saveTimer = null;
  timerEl.style.transition = 'none';
  timerEl.style.transform = 'scaleX(0)';
}

function scheduleSave() {
  cancelSave();
  // Restart the bar full, then shrink it over the save delay.
  timerEl.style.transform = 'scaleX(1)';
  void timerEl.offsetWidth;
  timerEl.style.transition = `transform ${SAVE_DELAY_MS}ms linear`;
  timerEl.style.transform = 'scaleX(0)';
  saveTimer = setTimeout(save, SAVE_DELAY_MS);
}

async function save() {
  saveTimer = null;
  const pointCount = strokes.reduce((n, s) => n + s.length, 0);
  if (pointCount < MIN_POINTS) {
    clearBox();
    return;
  }
  const t0 = strokes[0][0][2];
  const sample = {
    label,
    session,
    pointerType,
    box: { w: round(size.w, 1), h: round(size.h, 1) },
    strokes: strokes.map((s) => s.map(([x, y, t, p]) => [x, y, t - t0, p])),
  };

  // Move on at once so writing never waits for the network.
  const savedLabel = label;
  counts[savedLabel]++;
  lastLabel = savedLabel;
  clearBox();
  boxWrap.classList.add('saved');
  setTimeout(() => boxWrap.classList.remove('saved'), 250);
  showPrompt(pickLabel());

  try {
    const res = await fetch('/api/samples', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(sample),
    });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const { id } = await res.json();
    history.push({ id, label: savedLabel });
    toast(`Saved ${savedLabel === 'other' ? '✕' : savedLabel}`);
  } catch (err) {
    counts[savedLabel]--;
    renderProgress();
    toast(`Could not save that ${savedLabel}: ${err.message}`);
  }
}

async function deleteLast() {
  const last = history.pop();
  if (!last) {
    toast('Nothing saved in this session yet');
    return;
  }
  try {
    const res = await fetch(`/api/samples/${encodeURIComponent(last.id)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    counts[last.label]--;
    clearBox();
    showPrompt(last.label); // write it again
    toast(`Deleted the last ${last.label === 'other' ? '✕' : last.label} — write it again`);
  } catch (err) {
    history.push(last);
    toast(`Could not delete: ${err.message}`);
  }
}

trackStrokes(box, {
  toPoint(e) {
    const r = box.getBoundingClientRect();
    return [round(e.clientX - r.left, 1), round(e.clientY - r.top, 1), Math.round(e.timeStamp), round(pressureOf(e), 2)];
  },
  onStart(pt, e) {
    cancelSave();
    pointerType = e.pointerType;
    current = [pt];
    strokes.push(current);
    dirty = true;
  },
  onPoints(pts) {
    if (!current) return;
    current.push(...pts);
    dirty = true;
  },
  onEnd() {
    if (!current) return;
    current = null;
    scheduleSave();
  },
});

document.getElementById('redo').addEventListener('click', () => {
  clearBox();
  toast('');
});
document.getElementById('skip').addEventListener('click', () => {
  clearBox();
  lastLabel = label;
  showPrompt(pickLabel());
});
document.getElementById('undo').addEventListener('click', deleteLast);

autoFit(box, (s) => {
  // Points are stored in box pixels, so a size change mid-sample would mix scales.
  clearBox();
  size = s;
});
debugOverlay(box, () => `ink size   ${Math.round(size.w)} x ${Math.round(size.h)}`);

function frame() {
  requestAnimationFrame(frame); // first, so one bad draw cannot stop the loop
  if (dirty) {
    dirty = false;
    ctx.clearRect(0, 0, size.w, size.h);
    ctx.strokeStyle = ctx.fillStyle = INK;
    for (const s of strokes) drawStroke(ctx, s, (x, y) => [x, y], 5);
  }
}

async function init() {
  try {
    Object.assign(counts, await (await fetch('/api/stats')).json());
  } catch {
    toast('Could not load counts from the Mac — is the server running?');
  }
  showPrompt(pickLabel());
  requestAnimationFrame(frame);
}

init();
