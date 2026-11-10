// Census of Hands, drawn with the mouse on the Mac — no iPad needed.
//
// Draw a digit anywhere. When you stop, the model reads it and takes its
// fingerprint. The size of your digit sets the size of one grid cell, and
// the strangers from the archive who write that digit most like you fill an
// invisible grid around it in every direction — the closest right beside
// you — each one decoding into place out of flickering characters.
//
// Two fingers move around the archive, a pinch zooms, space brings you back.
// The strangers are still; only your digit, now and then, writes itself
// again the way you wrote it.

import { rasterize } from './preprocess.js';
import { CLASSES, fingerprintModel, fingerprints, loadModel, predict } from './model.js';
import { MNIST_TOTAL, inkCanvas, loadIndex, loadMnist, rankAll, strangerName, writerOf } from './archive.js';

const SETTLE_MS = 700; // pause after the last stroke before reading
const MIN_CONFIDENCE = 0.5;
const MIN_UNIT = 36;
const BRUSH = 15; // stroke width in CSS px while you draw
const LOW = 4; // the brush is drawn at 1/LOW resolution and blown up, for MNIST's soft, blocky edge
const SWAP_MS = 450; // your strokes fading into the 28×28 image the model saw
const RING_MS = 110; // ms per cell of distance for strangers to arrive
const DECODE = 650; // ms for one cell to decode top to bottom
const SCRAMBLE = '01|:.+#%/\\<>=*';
const FONT = "Futura, 'Futura PT', 'Century Gothic', sans-serif"; // every label, and the decoding characters
const REPLAY_WAIT = 3500; // ms your digit rests before writing itself again
const REPLAY_FADE = 250;
const REPLAY_HOLD = 500;
const MAX_GAP = 300; // longest pause kept between strokes in a replay
const MAX_VISIBLE = 3000; // cells on screen at the furthest zoom
const FLY_MS = 700;

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const archiveText = document.getElementById('archive');
const readingText = document.getElementById('reading');
const low = document.createElement('canvas');
const lowCtx = low.getContext('2d');

let model = null;
let fpModel = null;
let mnist = null;
let index = null;

let strokes = []; // [[ [x, y, t], ... ], ...] in world pixels, t in ms
let drawing = null;
let settleTimer = null;
let result = null; // see read()
// The view: screen = (world − cam) × zoom.
const cam = { x: 0, y: 0 };
let zoom = 1;
let flight = null; // an animated return to your digit
let dpr = 1;

// ---- Canvas ----

let W = 0;
let H = 0;
function resize() {
  dpr = window.devicePixelRatio || 1;
  W = canvas.clientWidth;
  H = canvas.clientHeight;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  low.width = Math.ceil(W / LOW);
  low.height = Math.ceil(H / LOW);
}
new ResizeObserver(resize).observe(canvas);
resize();

// ---- Drawing with the mouse ----

const pointOf = (e) => {
  const r = canvas.getBoundingClientRect();
  return [cam.x + (e.clientX - r.left) / zoom, cam.y + (e.clientY - r.top) / zoom, performance.now()];
};

// Once there is a reading, a press might be a click (pick a writer) or the
// start of a new digit: it only becomes a stroke once it moves a few pixels.
const CLICK_SLOP = 5;
let pending = null; // { point, sx, sy } while deciding

const screenOf = (e) => {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
};

function startStroke(point) {
  if (result) clearAll(); // a new stroke after a reading starts a new digit
  drawing = [point];
  strokes.push(drawing);
}

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  clearTimeout(settleTimer);
  try {
    canvas.setPointerCapture(e.pointerId);
  } catch {}
  const [sx, sy] = screenOf(e);
  if (result) pending = { point: pointOf(e), sx, sy };
  else startStroke(pointOf(e));
});

canvas.addEventListener('pointermove', (e) => {
  if (pending) {
    const [sx, sy] = screenOf(e);
    if (Math.hypot(sx - pending.sx, sy - pending.sy) < CLICK_SLOP) return;
    startStroke(pending.point);
    pending = null;
  }
  if (!drawing) return;
  const coalesced = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  const events = coalesced.length ? coalesced : [e];
  for (const ev of events) drawing.push(pointOf(ev));
});

// Clicking a digit picks out its writer: everything else they wrote stays
// lit, the rest of the archive dims. Clicking it again, or anywhere that is
// not a stranger, lets go.
let picked = null; // { writer, row, wx, wy } — wx, wy: where you clicked, in the world
function pickAt(point) {
  const [wx, wy] = point;
  let next = null;
  if (result && result.digit !== null) {
    const i = Math.floor((wx - result.you.x) / result.unit);
    const j = Math.floor((wy - result.you.y) / result.unit);
    const c = i || j ? result.cells.get(`${i},${j}`) : null;
    if (c) next = { writer: mnist.writers[c.row], row: c.row, wx, wy };
  }
  picked = next && picked && next.row === picked.row ? null : next;
}

const endStroke = () => {
  if (pending) {
    pickAt(pending.point);
    pending = null;
    return;
  }
  if (!drawing) return;
  drawing = null;
  settleTimer = setTimeout(read, SETTLE_MS);
};
canvas.addEventListener('pointerup', endStroke);
canvas.addEventListener('pointercancel', () => {
  pending = null;
  endStroke();
});

// Two fingers on the trackpad move the canvas; a pinch (which the browser
// reports as a wheel event with ctrlKey) zooms around the cursor.
canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    flight = null;
    const k = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? H : 1;
    if (e.ctrlKey) {
      const r = canvas.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      const wx = cam.x + px / zoom;
      const wy = cam.y + py / zoom;
      zoom = clampZoom(zoom * Math.exp(-e.deltaY * k * 0.01));
      cam.x = wx - px / zoom;
      cam.y = wy - py / zoom;
    } else {
      cam.x += (e.deltaX * k) / zoom;
      cam.y += (e.deltaY * k) / zoom;
    }
  },
  { passive: false },
);

function clampZoom(z) {
  // never so far out that there are too many cells to draw
  const unit = result ? result.unit : 100;
  const min = Math.max(0.05, Math.sqrt((W * H) / (MAX_VISIBLE * unit * unit)));
  return Math.min(8, Math.max(min, z));
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' || e.key === 'Backspace') clearAll();
  if (e.key === ' ') {
    e.preventDefault();
    comeBack();
  }
});

// Fly back to your digit, centred, at its original size.
function comeBack() {
  if (!strokes.length) return;
  const box = result ? result.box : boxOf(strokes);
  const cx = (box.x0 + box.x1) / 2;
  const cy = (box.y0 + box.y1) / 2;
  flight = {
    start: performance.now(),
    from: { x: cam.x, y: cam.y, zoom },
    to: { x: cx - W / 2, y: cy - H / 2, zoom: 1 },
  };
}

function fly(now) {
  if (!flight) return;
  const k = ease((now - flight.start) / FLY_MS);
  const { from, to } = flight;
  // interpolate the centre of the view, so the zoom feels anchored
  zoom = 1 / (1 / from.zoom + (1 / to.zoom - 1 / from.zoom) * k);
  const fcx = from.x + W / 2 / from.zoom;
  const fcy = from.y + H / 2 / from.zoom;
  const tcx = to.x + W / 2 / to.zoom;
  const tcy = to.y + H / 2 / to.zoom;
  cam.x = fcx + (tcx - fcx) * k - W / 2 / zoom;
  cam.y = fcy + (tcy - fcy) * k - H / 2 / zoom;
  if (k >= 1) flight = null;
}

function clearAll() {
  clearTimeout(settleTimer);
  strokes = [];
  drawing = null;
  result = null;
  picked = null;
  showReading();
}

// ---- Reading ----

function boxOf(list) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of list) {
    for (const [x, y] of s) {
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x);
      y1 = Math.max(y1, y);
    }
  }
  return { x0, y0, x1, y1 };
}

function read() {
  settleTimer = null;
  if (!strokes.length || !model || !index || !mnist) return;
  const image = rasterize(strokes);
  const probs = predict(model, image);
  const best = probs.indexOf(Math.max(...probs));
  const label = CLASSES[best];
  const box = boxOf(strokes);
  const base = { box, place: image.place, mine: imageCanvas(image), replay: timeline(strokes), start: performance.now(), digit: null, ranked: [] };
  if (label === 'other' || probs[best] < MIN_CONFIDENCE) {
    result = layout(base);
    showReading();
    return;
  }
  const digit = Number(label);
  fpModel ??= fingerprintModel(model);
  const query = fingerprints(fpModel, image, 1);
  const ranked = rankAll(index, mnist, query, digit);
  // how many of this digit each person wrote, for the note on a picked writer
  const perWriter = new Map();
  for (const { row } of ranked) perWriter.set(mnist.writers[row], (perWriter.get(mnist.writers[row]) ?? 0) + 1);
  result = layout({ ...base, digit, ranked, perWriter });
  showReading();
}

// When each point of your digit was drawn, for replaying it: your real speed,
// with long pauses between strokes shortened.
function timeline(list) {
  const times = [];
  let clock = 0;
  let prev = null;
  for (const s of list) {
    const ts = [];
    for (const p of s) {
      if (prev !== null) clock += Math.min(MAX_GAP, Math.max(0, p[2] - prev));
      prev = p[2];
      ts.push(clock);
    }
    times.push(ts);
  }
  return { times, duration: Math.max(clock, 400) };
}

function showReading() {
  if (!result) readingText.textContent = '';
  else if (result.digit === null) readingText.textContent = 'not a number — no one in the archive wrote this';
  else {
    const n = result.ranked.length.toLocaleString('en-US');
    readingText.textContent = `${n} ${result.digit}s by ${result.perWriter.size} people · further out, less like you`;
  }
}

// An invisible grid, one cell the size of your digit, centred on it and
// running on in every direction. Cells are numbered by distance from yours:
// the n-th closest cell holds the n-th most similar stranger, until everyone
// who wrote this digit has a place.
function layout(r) {
  // Your cell is exactly where the 28×28 image lands over your strokes —
  // MNIST centres a digit by its centre of mass, not its outline — so the
  // swap from strokes to image does not jump.
  const { place } = r;
  const unit = Math.max(MIN_UNIT, place.size);
  const cx = place.x + place.size / 2;
  const cy = place.y + place.size / 2;
  const you = { x: cx - unit / 2, y: cy - unit / 2 };
  const R = Math.ceil(Math.sqrt(r.ranked.length / Math.PI)) + 2;
  const slots = [];
  for (let j = -R; j <= R; j++) {
    for (let i = -R; i <= R; i++) {
      if (i || j) slots.push({ i, j, d: Math.hypot(i, j) + Math.random() * 0.01 });
    }
  }
  slots.sort((p, q) => p.d - q.d);
  const rankOf = new Map();
  slots.slice(0, r.ranked.length).forEach((s, rank) => rankOf.set(`${s.i},${s.j}`, rank));
  return { ...r, unit, you, rankOf, cells: new Map() };
}

// The cell at grid position (i, j), made the first time it comes into view.
function cellAt(r, i, j, t) {
  const key = `${i},${j}`;
  let c = r.cells.get(key);
  if (c) return c;
  const rank = r.rankOf.get(key);
  if (rank === undefined) return null; // past the edge of the archive
  const d = Math.hypot(i, j);
  c = {
    row: r.ranked[rank].row,
    similarity: r.ranked[rank].similarity,
    rank,
    x: r.you.x + i * r.unit,
    y: r.you.y + j * r.unit,
    // a ripple outward from you; cells you scroll to later decode as they appear
    delay: Math.max(200 + d * RING_MS, t) + Math.random() * RING_MS * 2,
    chars: charGrid(r.ranked[rank].row, r.unit),
  };
  r.cells.set(key, c);
  return c;
}

// The stranger's 28×28 digit divided into n×n squares, each with its ink and
// a moment (after the cell's arrival) when it stops flickering and shows the
// real handwriting underneath.
const gridCache = new Map();
function charGrid(row, unit) {
  const n = Math.max(8, Math.min(28, Math.round(unit / 7)));
  const key = `${row}:${n}`;
  if (gridCache.has(key)) return gridCache.get(key);
  const ink = new Float32Array(n * n);
  const base = row * 784;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const i = Math.floor((y * n) / 28) * n + Math.floor((x * n) / 28);
      ink[i] = Math.max(ink[i], mnist.pixels[base + y * 28 + x] / 255);
    }
  }
  const lock = new Float32Array(n * n);
  let done = 0;
  for (let i = 0; i < n * n; i++) {
    lock[i] = (Math.floor(i / n) / n) * DECODE * 0.65 + Math.random() * DECODE * 0.35;
    done = Math.max(done, lock[i]);
  }
  const grid = { n, ink, lock, done };
  gridCache.set(key, grid);
  return grid;
}

// ---- Drawing the scene ----

const ease = (t) => 1 - (1 - Math.min(1, Math.max(0, t))) ** 3;
const randomChar = () => SCRAMBLE[(Math.random() * SCRAMBLE.length) | 0];

// Your strokes as a soft brush: drawn small, then blown up with smoothing,
// so the edge blurs the way an enlarged MNIST digit does. `until` (ms into
// the replay) draws only what had been written by then.
function drawStrokes(alpha, until = Infinity) {
  if (alpha <= 0 || !strokes.length) return;
  const times = until === Infinity ? null : result.replay.times;
  const s = zoom / LOW;
  lowCtx.setTransform(1, 0, 0, 1, 0, 0);
  lowCtx.clearRect(0, 0, low.width, low.height);
  lowCtx.setTransform(s, 0, 0, s, -cam.x * s, -cam.y * s);
  lowCtx.strokeStyle = '#fff';
  lowCtx.fillStyle = '#fff';
  lowCtx.lineWidth = BRUSH;
  lowCtx.lineCap = 'round';
  lowCtx.lineJoin = 'round';
  strokes.forEach((st, k) => {
    const n = times ? times[k].findIndex((t) => t > until) : -1;
    const count = n === -1 ? st.length : n;
    if (!count) return;
    if (count === 1) {
      lowCtx.beginPath();
      lowCtx.arc(st[0][0], st[0][1], BRUSH / 2, 0, Math.PI * 2);
      lowCtx.fill();
      return;
    }
    lowCtx.beginPath();
    lowCtx.moveTo(st[0][0], st[0][1]);
    for (let i = 1; i < count; i++) lowCtx.lineTo(st[i][0], st[i][1]);
    lowCtx.stroke();
  });
  screenSpace();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(low, 0, 0, low.width * LOW, low.height * LOW);
  ctx.globalAlpha = 1;
  worldSpace();
}

const screenSpace = () => ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
const worldSpace = () => ctx.setTransform(dpr * zoom, 0, 0, dpr * zoom, -cam.x * dpr * zoom, -cam.y * dpr * zoom);

// The 28×28 picture the model actually read, as white ink on transparent.
function imageCanvas(pixels) {
  const c = document.createElement('canvas');
  c.width = c.height = 28;
  const g = c.getContext('2d');
  const img = g.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255;
    img.data[i * 4 + 3] = Math.round(pixels[i] * 255);
  }
  g.putImageData(img, 0, 0);
  return c;
}

function drawCell(r, c, t) {
  const since = t - c.delay;
  if (since < 0) return;
  const { x, y } = c;
  const img = inkCanvas(mnist, c.row, [255, 255, 255]);
  // still white, but fainter the less alike: zoomed out, the rings show
  ctx.globalAlpha = 1 - 0.7 * (c.rank / r.ranked.length);
  if (picked) ctx.globalAlpha = mnist.writers[c.row] === picked.writer ? 1 : 0.12;
  const base = ctx.globalAlpha;
  const { n, ink, lock, done } = c.chars;
  if (r.unit * zoom < 24) {
    // too small on screen to read characters: just fade in
    ctx.globalAlpha = base * ease(since / DECODE);
    ctx.drawImage(img, x, y, r.unit, r.unit);
  } else if (since >= done) {
    ctx.drawImage(img, x, y, r.unit, r.unit);
  } else {
    const step = r.unit / n;
    const src = 28 / n;
    ctx.font = `${Math.max(5, step * 1.05)}px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n * n; i++) {
      const col = i % n;
      const row = Math.floor(i / n);
      if (since >= lock[i]) {
        // decoded: the stranger's own ink shows through this square
        if (ink[i] > 0.02) {
          ctx.drawImage(img, col * src, row * src, src, src, x + col * step, y + row * step, step, step);
        }
      } else if (ink[i] > 0.1 || Math.random() < 0.04) {
        // still decoding: flicker, with a little noise in the empty space
        ctx.fillStyle = `rgba(255,255,255,${0.25 + 0.5 * Math.random()})`;
        ctx.fillText(randomChar(), x + (col + 0.5) * step, y + (row + 0.5) * step);
      }
    }
  }
  ctx.globalAlpha = 1;
  // who they are, and how close: rank among everyone who wrote this digit,
  // and how alike the two fingerprints are
  const px = r.unit * zoom;
  if (px >= 60) {
    const a = ease((since - done) / 400);
    if (a > 0) {
      const fs = (px >= 110 ? 10 : 8) / zoom;
      const pad = 4 / zoom;
      ctx.font = `${fs}px ${FONT}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      const dim = picked && mnist.writers[c.row] !== picked.writer ? 0.2 : 1;
      ctx.fillStyle = `rgba(255,255,255,${0.55 * a * dim})`;
      ctx.fillText(`#${c.rank + 1} · ${(c.similarity * 100).toFixed(1)}%`, x + pad, y + r.unit - pad);
      if (px >= 110) {
        ctx.fillStyle = `rgba(255,255,255,${0.3 * a * dim})`;
        ctx.fillText(strangerName(c.row), x + pad, y + r.unit - pad - fs * 1.3);
      }
    }
  }
}

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  fly(now);
  screenSpace();
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  worldSpace();
  if (!result) {
    drawStrokes(1);
    return;
  }
  const t = now - result.start;
  if (result.digit !== null) {
    const { unit, you } = result;
    const i0 = Math.floor((cam.x - you.x) / unit);
    const i1 = Math.ceil((cam.x + W / zoom - you.x) / unit);
    const j0 = Math.floor((cam.y - you.y) / unit);
    const j1 = Math.ceil((cam.y + H / zoom - you.y) / unit);
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        if (!i && !j) continue;
        const c = cellAt(result, i, j, t);
        if (c) drawCell(result, c, t);
      }
    }
  }
  drawYou(t);
  drawPickedNote();
}

// Who you picked: their writer number, which group they came from, and how
// many times they wrote this digit for the archive. Pinned where you clicked.
function drawPickedNote() {
  if (!picked) return;
  const w = writerOf(picked.row, mnist);
  const n = result.perWriter.get(picked.writer) ?? 0;
  const sx = (picked.wx - cam.x) * zoom;
  const sy = (picked.wy - cam.y) * zoom;
  const lines = [
    `Writer ${w.id.toLocaleString('en-US')}`,
    w.student ? 'high-school student' : 'adult employee',
    `wrote ${n} ${result.digit}${n === 1 ? '' : 's'} in the archive`,
  ];
  screenSpace();
  ctx.font = `12px ${FONT}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  const wdt = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
  const hgt = lines.length * 16 + 10;
  const x = Math.min(sx + 14, W - wdt - 8);
  const y = Math.min(sy + 14, H - hgt - 8);
  ctx.fillStyle = 'rgba(0,0,0,0.85)';
  ctx.fillRect(x, y, wdt, hgt);
  lines.forEach((l, k) => {
    ctx.fillStyle = k ? 'rgba(255,255,255,0.6)' : '#fff';
    ctx.fillText(l, x + 8, y + 6 + k * 16);
  });
  worldSpace();
}

// Once read, you become a digit in the archive's own format, like theirs.
// Then every few seconds your digit writes itself again, stroke by stroke,
// at the speed you wrote it — the one hand here that is still moving.
function drawYou(t) {
  const { mine, you, unit, replay } = result;
  const image = (a) => {
    if (a <= 0) return;
    ctx.globalAlpha = a;
    ctx.drawImage(mine, you.x, you.y, unit, unit);
    ctx.globalAlpha = 1;
  };
  if (t < SWAP_MS) {
    image(ease(t / SWAP_MS));
    drawStrokes(1 - ease(t / SWAP_MS));
    return;
  }
  const period = REPLAY_WAIT + REPLAY_FADE + replay.duration + REPLAY_HOLD + SWAP_MS;
  let u = (t - SWAP_MS) % period;
  if (u < REPLAY_WAIT) return image(1);
  u -= REPLAY_WAIT;
  if (u < REPLAY_FADE) return image(1 - u / REPLAY_FADE);
  u -= REPLAY_FADE;
  if (u < replay.duration + REPLAY_HOLD) return drawStrokes(1, u);
  u -= replay.duration + REPLAY_HOLD;
  image(ease(u / SWAP_MS));
  drawStrokes(1 - ease(u / SWAP_MS));
}

// ---- Opening the archive ----

async function openArchive() {
  try {
    await tf.ready();
    const saved = await loadModel();
    if (!saved) throw new Error('No model yet — train one at /train.html');
    model = saved.model;
    index = await loadIndex();
    if (!index) throw new Error('The archive is not indexed yet — press Train at /train.html');
    mnist = await loadMnist(MNIST_TOTAL, (done, total) => {
      archiveText.textContent = `opening the archive ${Math.round((done / total) * 100)}%`;
    });
    archiveText.hidden = true; // loaded: the bottom of the page stays empty
    if (strokes.length && !drawing) read();
  } catch (err) {
    archiveText.textContent = err.message;
  }
}

requestAnimationFrame(frame);
openArchive();
