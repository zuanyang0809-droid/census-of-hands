// Mac side: mirrors the iPad page live and reads it. Each symbol gets a
// faint box and the model's guess beside it; the bottom of the screen shows
// the score — every row's notes, left to right.
//
// Stopped: each note sounds once when it is written (and again if it
// changes, e.g. an octave dot is added).
// Playing: all rows play together, one beat per symbol, and loop as one —
// the loop is as long as the longest row; shorter rows rest until it comes
// round again. Notes written while playing join on the loop's next pass.

import { INK, autoFit, drawGuides, drawStroke } from './ink.js';
import { rasterize } from './preprocess.js';
import { CLASSES, loadModel, predict } from './model.js';
import { Reader } from './reader.js';
import { Synth, noteFor, noteName } from './sound.js';
import { compileRow } from './sequence.js';

const MARGIN = 40;
const SCORE_SPACE = 120; // room under the page for the score
const UNSURE = 0.7; // below this confidence the guess is shown in orange
const SETTLE_MS = 450; // wait this long after a stroke before sounding, so a 4's first stroke is not played as a 1
const FLASH_MS = 500;
const LOOKAHEAD_S = 0.15; // notes are booked on the audio clock this far ahead, for steady timing
const TICK_MS = 25;
const PLAYHEAD = '#e0583a';

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const statusDot = document.getElementById('status');
const statusText = document.getElementById('status-text');
const modelText = document.getElementById('model-text');
const scoreEl = document.getElementById('score');
const soundBtn = document.getElementById('sound');
const playBtn = document.getElementById('play');
const tempoInput = document.getElementById('tempo');
const tempoText = document.getElementById('tempo-text');
const nowEl = document.getElementById('now');

const synth = new Synth();
const lastPlayed = new Map(); // symbol id -> what it last sounded as, e.g. "5+1"
const flashes = new Map(); // symbol id -> time it last sounded
let settleTimer = null;

// Playback
let rowSteps = []; // per row: beats from compileRow
let bpm = Number(tempoInput.value);
let playing = false;
let tickTimer = null;
let cyclePos = 0; // position of the next beat within the loop
let nextBeatTime = 0; // audio-clock time of the next beat
let beatQueue = []; // booked beats not yet reached: { pos, time, dur }
let currentBeat = null; // the beat now sounding

let size = { w: 1, h: 1 };
let page = { aspect: 0.75, rows: 4 };
let strokes = new Map(); // id -> points, in drawing order
let read = new Set(); // ids of finished strokes already given to the reader
let ws = null;
let pens = 0;
let penUrl = '';
let model = null;
let dirty = true;

const reader = new Reader((glyphStrokes) => {
  if (!model) return null;
  const probs = predict(model, rasterize(glyphStrokes));
  const best = probs.indexOf(Math.max(...probs));
  return { label: CLASSES[best], confidence: probs[best] };
});

const glyph = (label) => (label === 'other' ? '✕' : label === '-' ? '–' : label);
const withOctave = (text, octave) => text + (octave > 0 ? '̇' : '̣').repeat(Math.abs(octave));

function setStatus() {
  const open = ws?.readyState === WebSocket.OPEN;
  statusDot.className = `dot ${open ? (pens ? 'live' : 'waiting') : ''}`;
  if (!open) statusText.textContent = 'Not connected to server';
  else if (pens) statusText.textContent = 'iPad connected';
  else statusText.textContent = penUrl ? `Waiting for iPad — open ${penUrl}` : 'Waiting for iPad';
}

// Bigger writing is louder: symbol height as a share of the row height.
function velocityFor(box) {
  const size = (box.y1 - box.y0) / (page.aspect / page.rows);
  return Math.max(0.2, Math.min(1, 0.25 + (size - 0.2) * 1.1));
}

// Sound every note that is new or has changed since it last sounded.
function settle() {
  settleTimer = null;
  const now = performance.now();
  for (const { notes } of reader.score()) {
    for (const n of notes) {
      const key = `${n.symbol}${n.octave >= 0 ? '+' : ''}${n.octave}`;
      if (lastPlayed.get(n.id) === key) continue;
      lastPlayed.set(n.id, key);
      if (playing) continue; // the loop will play it on its next pass
      const midi = noteFor(n.symbol, n.octave);
      if (midi === null) continue; // 0, - and / are silent for now
      synth.play(midi, { velocity: velocityFor(n.box) });
      flashes.set(n.id, now);
      nowEl.textContent = `♪ ${n.symbol} · ${noteName(midi)}${synth.ready && !synth.muted ? '' : ' (sound off)'}`;
    }
  }
  dirty = true;
}

function scheduleSettle() {
  clearTimeout(settleTimer);
  settleTimer = setTimeout(settle, SETTLE_MS);
}

// After a full rebuild, treat what is already on the page as heard,
// so reconnecting does not play the whole page at once.
function markAllPlayed() {
  clearTimeout(settleTimer);
  lastPlayed.clear();
  for (const { notes } of reader.score()) {
    for (const n of notes) lastPlayed.set(n.id, `${n.symbol}${n.octave >= 0 ? '+' : ''}${n.octave}`);
  }
}

function rebuild(done) {
  reader.reset();
  reader.setPage(page);
  read = new Set();
  for (const [id, pts] of strokes) {
    if (!done.has(id)) continue;
    reader.addStroke(pts);
    read.add(id);
  }
}

// ---- Playback ----

const beatSeconds = () => 60 / bpm;

const loopLength = () => Math.max(0, ...rowSteps.map((s) => s.length));

function onBeat(pos) {
  const now = performance.now();
  const names = [];
  for (const steps of rowSteps) {
    const s = steps[pos]; // undefined once a shorter row has finished
    if (s?.kind !== 'note') continue;
    flashes.set(s.id, now);
    names.push(noteName(s.midi));
  }
  nowEl.textContent = names.length ? `♪ ${names.join(' + ')}` : '';
}

// Book every beat that starts before now + LOOKAHEAD_S.
function tick() {
  const audioNow = synth.ctx.currentTime;
  while (nextBeatTime < audioNow + LOOKAHEAD_S) {
    const dur = beatSeconds();
    const length = loopLength();
    if (cyclePos >= length) cyclePos = 0; // the page got shorter (or is empty)
    for (const steps of rowSteps) {
      const s = steps[cyclePos];
      if (s?.kind !== 'note') continue;
      synth.play(s.midi, {
        velocity: s.velocity,
        duration: Math.max(0.5, s.beats * dur * 1.05),
        when: Math.max(0, nextBeatTime - audioNow),
      });
    }
    beatQueue.push({ pos: cyclePos, time: nextBeatTime, dur });
    cyclePos = length ? (cyclePos + 1) % length : 0;
    nextBeatTime += dur;
  }
}

function sendState() {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'transport-state', playing, bpm, soundReady: synth.ready }));
  }
}

function updateButtons() {
  const on = synth.ready && !synth.muted;
  soundBtn.textContent = !synth.ready ? 'Turn on sound' : on ? `Sound on · ${synth.instrument}` : 'Sound off';
  soundBtn.classList.toggle('off', !on);
  playBtn.textContent = playing ? '■ Stop' : '▶ Play';
  playBtn.classList.toggle('playing', playing);
  tempoText.textContent = `${bpm} BPM`;
  sendState();
}

// `fromClick`: started by a click on this page, so sound may be switched on.
async function start(fromClick) {
  if (playing) return;
  if (!synth.ready) {
    if (!fromClick) {
      sendState(); // the iPad shows "turn on sound on the Mac first"
      return;
    }
    await synth.unlock();
  }
  rowSteps = reader.score().map((r) => compileRow(r.notes, velocityFor));
  cyclePos = 0;
  nextBeatTime = synth.ctx.currentTime + 0.08;
  beatQueue = [];
  currentBeat = null;
  playing = true;
  tick();
  tickTimer = setInterval(tick, TICK_MS);
  updateButtons();
}

function stop() {
  if (!playing) return;
  playing = false;
  clearInterval(tickTimer);
  beatQueue = [];
  currentBeat = null;
  nowEl.textContent = '';
  dirty = true;
  updateButtons();
}

const toggle = (fromClick) => (playing ? stop() : start(fromClick));

function handle(msg) {
  switch (msg.type) {
    case 'transport':
      if (msg.action === 'toggle') toggle(false);
      return;
    case 'peers':
      pens = msg.pens;
      setStatus();
      return;
    case 'sync':
      page = { aspect: msg.aspect, rows: msg.rows };
      strokes = new Map(msg.strokes.map((s) => [s.id, s.pts]));
      rebuild(new Set(msg.strokes.filter((s) => s.done !== false).map((s) => s.id)));
      markAllPlayed();
      renderScore();
      break;
    case 'start':
      strokes.set(msg.id, [msg.pt]);
      clearTimeout(settleTimer); // still writing: wait until the pen has been up a moment
      break;
    case 'points':
      strokes.get(msg.id)?.push(...msg.pts);
      break;
    case 'end':
      if (strokes.has(msg.id) && !read.has(msg.id)) {
        reader.addStroke(strokes.get(msg.id));
        read.add(msg.id);
        renderScore();
        scheduleSettle();
      }
      break;
    case 'clear':
      strokes.clear();
      read.clear();
      reader.reset();
      clearTimeout(settleTimer);
      lastPlayed.clear();
      flashes.clear();
      nowEl.textContent = '';
      renderScore();
      break;
    default:
      return;
  }
  dirty = true;
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws?role=display`);
  ws.onopen = () => {
    setStatus();
    sendState();
  };
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

function renderScore() {
  const score = reader.score();
  rowSteps = score.map((r) => compileRow(r.notes, velocityFor)); // a playing loop picks this up on its next beat
  scoreEl.innerHTML = '';
  for (const { row, notes } of score) {
    const line = document.createElement('div');
    line.className = 'row';
    const label = document.createElement('b');
    const beats = rowSteps[row].length;
    label.textContent = beats ? `row ${row + 1} · ${beats} beat${beats === 1 ? '' : 's'}` : `row ${row + 1}`;
    line.append(label);
    if (!notes.length) line.append(Object.assign(document.createElement('span'), { className: 'muted', textContent: '·' }));
    for (const n of notes) {
      const span = document.createElement('span');
      span.textContent = `${withOctave(glyph(n.symbol), n.octave)} `;
      if (n.confidence < UNSURE) span.className = 'unsure';
      line.append(span);
    }
    scoreEl.append(line);
  }
}

// The iPad page, fitted inside the window with its own aspect ratio.
function pageRect() {
  const availW = Math.max(0, size.w - MARGIN * 2);
  const availH = Math.max(0, size.h - MARGIN * 2 - SCORE_SPACE);
  const w = Math.min(availW, availH / page.aspect);
  const h = w * page.aspect;
  return { x: (size.w - w) / 2, y: MARGIN + (availH - h) / 2, w, h };
}

function drawGlyphs(r) {
  const rowPx = (r.h / page.rows);
  const font = Math.max(12, rowPx * 0.2);
  const octaves = new Map();
  for (const { notes } of reader.score()) for (const n of notes) octaves.set(n.id, n.octave);
  for (const g of reader.glyphs) {
    const x0 = r.x + g.box.x0 * r.w - 6;
    const y0 = r.y + g.box.y0 * r.w - 6;
    const x1 = r.x + g.box.x1 * r.w + 6;
    const y1 = r.y + g.box.y1 * r.w + 6;
    const res = g.result;
    const color = !res ? '#8a8a8e' : res.label === 'other' ? '#b0b0b4' : res.confidence < UNSURE ? '#d99a1e' : '#2f6fd6';
    const age = performance.now() - (flashes.get(g.id) ?? -Infinity);
    if (age < FLASH_MS) {
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.25 * (1 - age / FLASH_MS);
      ctx.beginPath();
      ctx.roundRect(x0, y0, x1 - x0, y1 - y0, 6);
      ctx.fill();
      dirty = true; // keep drawing until the flash has faded
    }
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x0, y0, x1 - x0, y1 - y0, 6);
    ctx.stroke();
    ctx.globalAlpha = 1;
    // Label under the box, centred: "7 98%". Under rather than beside, so
    // symbols written close together do not cover each other's labels.
    ctx.fillStyle = color;
    ctx.textBaseline = 'top';
    const name = res ? withOctave(glyph(res.label), octaves.get(g.id) ?? 0) : '…';
    const conf = res && res.label !== 'other' ? ` ${Math.round(res.confidence * 100)}%` : '';
    const nameFont = `600 ${font}px -apple-system, sans-serif`;
    const confFont = `${Math.round(font * 0.55)}px -apple-system, sans-serif`;
    ctx.font = nameFont;
    const nameW = ctx.measureText(name).width;
    ctx.font = confFont;
    const confW = ctx.measureText(conf).width;
    const left = (x0 + x1) / 2 - (nameW + confW) / 2;
    ctx.font = nameFont;
    ctx.fillText(name, left, y1 + 3);
    ctx.font = confFont;
    ctx.fillText(conf, left + nameW, y1 + 3 + font * 0.38);
  }
}

// One line per row, gliding from the sounding symbol towards the next.
// A row that has finished while a longer one plays on shows no line.
function drawPlayheads(r) {
  if (!playing || !currentBeat) return;
  const rowPx = r.h / page.rows;
  const frac = Math.max(0, Math.min(1, (synth.ctx.currentTime - currentBeat.time) / currentBeat.dur));
  ctx.strokeStyle = PLAYHEAD;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.8;
  rowSteps.forEach((steps, row) => {
    const i = currentBeat.pos;
    if (i >= steps.length) return;
    const here = steps[i].x;
    const next = i + 1 < steps.length ? steps[i + 1].x : here; // at the end of the row, wait, then jump back
    const x = r.x + (here + (next - here) * frac) * r.w;
    const y0 = r.y + row * rowPx + 8;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + rowPx - 16);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function frame() {
  requestAnimationFrame(frame); // first, so one bad draw cannot stop the loop
  if (playing) {
    const audioNow = synth.ctx.currentTime;
    while (beatQueue.length && beatQueue[0].time <= audioNow) {
      currentBeat = beatQueue.shift();
      onBeat(currentBeat.pos);
    }
    dirty = true;
  }
  const r = pageRect();
  if (dirty && r.w > 0) {
    dirty = false;
    ctx.clearRect(0, 0, size.w, size.h);
    ctx.fillStyle = '#fff';
    ctx.fillRect(r.x, r.y, r.w, r.h);
    drawGuides(ctx, r.x, r.y, r.w, r.h, page.rows);
    ctx.strokeStyle = ctx.fillStyle = INK;
    const map = (x, y) => [r.x + x * r.w, r.y + y * r.w];
    for (const pts of strokes.values()) drawStroke(ctx, pts, map, 3 * (r.w / 1000));
    drawGlyphs(r);
    drawPlayheads(r);
  }
}

// Also covers a tab opened in the background, which starts at 0×0.
autoFit(canvas, (s) => {
  size = s;
  dirty = true;
});

fetch('/api/info')
  .then((r) => r.json())
  .then(({ addresses, port }) => {
    if (addresses[0]) penUrl = `http://${addresses[0]}:${port}/pen.html`;
    setStatus();
  })
  .catch(() => {});

loadModel()
  .then((saved) => {
    if (!saved) {
      modelText.textContent = '· No model yet — train one at /train.html';
      return;
    }
    model = saved.model;
    modelText.textContent = `· Model from ${new Date(saved.meta.savedAt).toLocaleString()}`;
    reader.rerecognize();
    markAllPlayed();
    renderScore();
    dirty = true;
  })
  .catch((err) => {
    modelText.textContent = `· ${err.message}`;
  });

// Buttons drop focus after a click, so Space only ever means play/stop.
for (const el of [soundBtn, playBtn]) el.addEventListener('click', () => el.blur());
soundBtn.addEventListener('click', async () => {
  if (!synth.ready) {
    await synth.unlock();
    synth.play(noteFor('1', 0), { velocity: 0.5 }); // a short "hello" so you hear it is on
  } else {
    synth.muted = !synth.muted;
  }
  updateButtons();
});
playBtn.addEventListener('click', () => toggle(true));
tempoInput.addEventListener('input', () => {
  bpm = Number(tempoInput.value);
  updateButtons();
});
document.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || e.target.tagName === 'INPUT') return;
  e.preventDefault();
  toggle(true);
});
updateButtons();

connect();
setStatus();
renderScore();
requestAnimationFrame(frame);
