// Census of Hands — the Mac screen.
//
// Write a digit on the iPad. The model reads it, takes a fingerprint of the
// handwriting, and finds the 12 people in the MNIST archive who wrote that
// digit most like you. They gather around your digit, closest first, each
// known only by their row number in the archive. Along the bottom, every
// number you have written is rewritten by the stranger closest to you.

import { autoFit } from './ink.js';
import { rasterize } from './preprocess.js';
import { CLASSES, fingerprintModel, fingerprints, loadModel, predict } from './model.js';
import { MNIST_TOTAL, inkCanvas, loadIndex, loadMnist, nearest, strangerName } from './archive.js';
import { Reader } from './reader.js';

const STRANGERS = 12;
const SETTLE_MS = 450; // wait after the pen lifts, so a 4's first stroke is not read alone
const MIN_CONFIDENCE = 0.5;
const ARRIVE_GAP_MS = 110; // between one stranger and the next
const ARRIVE_MS = 700;
const CELL = 46; // background archive grid
const STRIP_HEIGHT = 165; // "your numbers, in their hands" along the bottom
const PAPER = '#f2efe6';
const INK = [29, 29, 27];
const RED = '#a8322a';
const MUTED = '#8b877d';
const MONO = 'ui-monospace, Menlo, monospace';

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const statusDot = document.getElementById('status');
const statusText = document.getElementById('status-text');
const archiveText = document.getElementById('archive-text');

let size = { w: 1, h: 1 };
let page = { aspect: 0.75, rows: 4 };
let strokes = new Map();
let read = new Set();
let ws = null;
let pens = 0;
let penUrl = '';
let model = null;
let fpModel = null;
let mnist = null;
let index = null;
let settleTimer = null;

// symbol id -> { key, digit, strangers: [{ row, similarity }] }
const found = new Map();
let focus = null; // { id, start }
let background = [];

const reader = new Reader((glyphStrokes) => {
  if (!model) return null;
  const probs = predict(model, rasterize(glyphStrokes));
  const best = probs.indexOf(Math.max(...probs));
  return { label: CLASSES[best], confidence: probs[best] };
});

const ready = () => model && index && mnist;

function setStatus() {
  const open = ws?.readyState === WebSocket.OPEN;
  statusDot.className = `dot ${open ? (pens ? 'live' : 'waiting') : ''}`;
  if (!open) statusText.textContent = 'Not connected to server';
  else if (pens) statusText.textContent = 'iPad connected';
  else statusText.textContent = penUrl ? `Waiting for iPad — open ${penUrl}` : 'Waiting for iPad';
}

// ---- Reading and searching ----

function isDigit(g) {
  return g.result && g.result.label !== 'other' && g.result.confidence >= MIN_CONFIDENCE;
}

function search(g) {
  const digit = Number(g.result.label);
  fpModel ??= fingerprintModel(model);
  const query = fingerprints(fpModel, rasterize(g.strokes), 1);
  // one digit per person, so the 12 around you are 12 different people
  return { digit, strangers: nearest(index, mnist, query, digit, STRANGERS, { people: true }) };
}

const keyOf = (g) => `${g.strokes.length}:${g.result?.label}`;

// Look up every symbol that is new or changed; the last of them takes the stage.
function settle({ animate = true } = {}) {
  settleTimer = null;
  if (!ready()) return;
  let latest = null;
  for (const g of reader.glyphs) {
    const key = keyOf(g);
    if (found.get(g.id)?.key === key) continue;
    found.set(g.id, { key, ...(isDigit(g) ? search(g) : { digit: null, strangers: [] }) });
    latest = g.id;
  }
  if (latest !== null) focus = { id: latest, start: animate ? performance.now() : -1e9 };
}

function scheduleSettle() {
  clearTimeout(settleTimer);
  settleTimer = setTimeout(settle, SETTLE_MS);
}

function rebuild(done) {
  reader.reset();
  reader.setPage(page);
  read = new Set();
  found.clear();
  focus = null;
  for (const [id, pts] of strokes) {
    if (!done.has(id)) continue;
    reader.addStroke(pts);
    read.add(id);
  }
  settle({ animate: false });
}

function handle(msg) {
  switch (msg.type) {
    case 'peers':
      pens = msg.pens;
      setStatus();
      return;
    case 'sync':
      page = { aspect: msg.aspect, rows: msg.rows };
      strokes = new Map(msg.strokes.map((s) => [s.id, s.pts]));
      rebuild(new Set(msg.strokes.filter((s) => s.done !== false).map((s) => s.id)));
      return;
    case 'start':
      strokes.set(msg.id, [msg.pt]);
      clearTimeout(settleTimer); // still writing
      return;
    case 'points':
      strokes.get(msg.id)?.push(...msg.pts);
      return;
    case 'end':
      if (strokes.has(msg.id) && !read.has(msg.id)) {
        reader.addStroke(strokes.get(msg.id));
        read.add(msg.id);
        scheduleSettle();
      }
      return;
    case 'clear':
      strokes.clear();
      read.clear();
      reader.reset();
      found.clear();
      focus = null;
      clearTimeout(settleTimer);
      return;
    default:
  }
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws?role=display`);
  ws.onopen = setStatus;
  ws.onmessage = (e) => {
    try {
      handle(JSON.parse(e.data));
    } catch (err) {
      console.warn('Bad message', err);
    }
  };
  ws.onclose = () => {
    setStatus();
    setTimeout(connect, 1000);
  };
}

// ---- Drawing ----

const ease = (t) => 1 - (1 - Math.max(0, Math.min(1, t))) ** 3;

function drawStranger(row, x, y, cell, alpha) {
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(inkCanvas(mnist, row, INK), x - cell / 2, y - cell / 2, cell, cell);
  ctx.globalAlpha = 1;
}

// The faint archive behind everything: digits fading in and out.
function layoutBackground() {
  background = [];
  if (!mnist) return;
  const cols = Math.ceil(size.w / CELL);
  const rows = Math.ceil(size.h / CELL);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      background.push({
        x: c * CELL + CELL / 2,
        y: r * CELL + CELL / 2,
        row: Math.floor(Math.random() * mnist.rows),
        speed: 0.15 + Math.random() * 0.35,
        phase: Math.random() * Math.PI * 2,
        low: false,
      });
    }
  }
}

function drawBackground(t, strength) {
  for (const b of background) {
    const s = 0.5 + 0.5 * Math.sin(t * b.speed + b.phase);
    if (s < 0.03 && !b.low) {
      b.row = Math.floor(Math.random() * mnist.rows); // a different stranger next time
      b.low = true;
    } else if (s > 0.1) b.low = false;
    drawStranger(b.row, b.x, b.y, CELL * 0.62, s * strength);
  }
}

// Your symbol, fitted into a square of `box` px centred on (cx, cy).
function drawYours(g, cx, cy, box, alpha) {
  const pts = g.strokes.flat();
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const w = Math.max(...xs) - minX;
  const h = Math.max(...ys) - minY;
  const s = box / Math.max(w, h, 1e-6);
  const ox = cx - (w * s) / 2;
  const oy = cy - (h * s) / 2;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = ctx.fillStyle = RED;
  ctx.lineWidth = Math.max(3, box * 0.045);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const stroke of g.strokes) {
    ctx.beginPath();
    stroke.forEach(([x, y], i) => (i ? ctx.lineTo(ox + (x - minX) * s, oy + (y - minY) * s) : ctx.moveTo(ox + (x - minX) * s, oy + (y - minY) * s)));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function text(str, x, y, { size: px = 11, color = MUTED, align = 'center', alpha = 1, font = MONO, weight = 400 } = {}) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.font = `${weight} ${px}px ${font}`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(str, x, y);
  ctx.globalAlpha = 1;
}

function drawStage(now) {
  const g = reader.glyphs.find((x) => x.id === focus?.id);
  const info = g && found.get(g.id);
  const stripTop = size.h - STRIP_HEIGHT;
  const cx = size.w / 2;
  const cy = 120 + (stripTop - 120) / 2;
  const unit = Math.min(size.w * 0.9, stripTop - 150);
  const box = unit * 0.3;

  if (!g || !info) {
    text('write a number on the iPad', cx, cy, { size: 14, alpha: 0.8 });
    return;
  }
  const age = now - focus.start;
  drawYours(g, cx, cy, box, ease(age / 300));

  if (info.digit === null) {
    text('not a number — no one in the archive wrote this', cx, cy + box * 0.8, { size: 13, alpha: ease((age - 300) / 600) });
    return;
  }

  const radius = unit * 0.38;
  const cell = unit * 0.12;
  // Closest first, clockwise from the top. (No similarity figures: they are
  // all within a percent of each other, so the order says more.)
  info.strangers.forEach(({ row }, i) => {
    const p = ease((age - 250 - i * ARRIVE_GAP_MS) / ARRIVE_MS);
    if (p <= 0) return;
    const a = -Math.PI / 2 + (i / info.strangers.length) * Math.PI * 2;
    const x = cx + Math.cos(a) * radius * p;
    const y = cy + Math.sin(a) * radius * p;
    drawStranger(row, x, y, cell, p);
    text(strangerName(row), x, y + cell * 0.62, { size: 10, alpha: p * 0.9, color: '#5c5952' });
  });
  const people = info.strangers.length;
  text(`the ${people} people in the archive who write ${info.digit} most like you`, cx, cy + radius + cell * 1.15, {
    size: 12,
    alpha: ease((age - 250 - people * ARRIVE_GAP_MS) / 600),
  });
}

// Everything you have written, each digit in the hand of its closest stranger.
function drawStrip() {
  const items = reader.glyphs
    .filter((g) => found.get(g.id)?.strangers.length)
    .sort((a, b) => a.row - b.row || a.box.x0 - b.box.x0);
  const top = size.h - STRIP_HEIGHT;
  ctx.strokeStyle = 'rgba(0,0,0,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(36, top + 0.5);
  ctx.lineTo(size.w - 36, top + 0.5);
  ctx.stroke();
  text('YOUR NUMBERS, IN THEIR HANDS', 36, top + 20, { align: 'left', size: 10 });
  if (!items.length) return;

  const cell = 56;
  const gap = 18;
  const fit = Math.max(1, Math.floor((size.w - 72) / (cell + gap)));
  const shown = items.slice(-fit);
  let x = 36 + cell / 2;
  let prevRow = shown[0].row;
  for (const g of shown) {
    if (g.row !== prevRow) x += gap; // a little more space between lines of writing
    prevRow = g.row;
    const s = found.get(g.id).strangers[0];
    drawStranger(s.row, x, top + 66, cell, 1);
    text(strangerName(s.row), x, top + 104, { size: 9 });
    x += cell + gap;
  }
}

function frame(nowMs) {
  requestAnimationFrame(frame);
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, size.w, size.h);
  if (!ready()) {
    text(archiveText.textContent || 'Opening the archive…', size.w / 2, size.h / 2, { size: 13 });
    return;
  }
  drawBackground(nowMs / 1000, focus ? 0.035 : 0.07);
  drawStage(nowMs);
  drawStrip();
}

// ---- Start ----

autoFit(canvas, (s) => {
  size = s;
  layoutBackground();
});

fetch('/api/info')
  .then((r) => r.json())
  .then(({ addresses, port }) => {
    if (addresses[0]) penUrl = `http://${addresses[0]}:${port}/pen.html`;
    setStatus();
  })
  .catch(() => {});

async function openArchive() {
  try {
    const saved = await loadModel();
    if (!saved) throw new Error('No model yet — train one at /train.html');
    model = saved.model;
    index = await loadIndex();
    if (!index) throw new Error('The archive is not indexed yet — press Train at /train.html');
    mnist = await loadMnist(MNIST_TOTAL, (done, total) => {
      archiveText.textContent = `· opening the archive ${Math.round((done / total) * 100)}%`;
    });
    archiveText.textContent = `· ${mnist.rows.toLocaleString()} digits by ${mnist.people.length} people in the archive`;
    layoutBackground();
    reader.rerecognize();
    settle({ animate: false });
  } catch (err) {
    archiveText.textContent = `· ${err.message}`;
  }
}

connect();
setStatus();
openArchive();
requestAnimationFrame(frame);
