// iPad side: the writing surface. Draws locally for instant feedback and
// streams every stroke to the Mac display.
//
// Coordinates are sent normalised by the surface WIDTH (x and y both divided
// by width), plus the surface aspect ratio, so the display can redraw the
// page at any size without distortion.

import { INK, autoFit, debugOverlay, drawGuides, drawStroke, lockPage, pressureOf, round, trackStrokes } from './ink.js';

const ROWS = 4;

lockPage();

const canvas = document.getElementById('pad');
const ctx = canvas.getContext('2d');
const statusDot = document.getElementById('status');
const statusText = document.getElementById('status-text');
const playBtn = document.getElementById('play');
let soundHint = ''; // set when the Mac cannot play yet

let size = { w: 1, h: 1 };
let strokes = []; // { id, pts }
let current = null;
let nextId = 1;
let pending = []; // points not yet sent
let ws = null;
let displays = 0;
let dirty = true;

function send(msg) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function sync() {
  // `done` tells the display which strokes are finished and can be read.
  send({ type: 'sync', aspect: size.h / size.w, rows: ROWS, strokes: strokes.map((s) => ({ id: s.id, pts: s.pts, done: s !== current })) });
}

function setStatus() {
  const open = ws?.readyState === WebSocket.OPEN;
  statusDot.className = `dot ${open ? (displays ? 'live' : 'waiting') : ''}`;
  statusText.textContent = !open
    ? 'Not connected to Mac'
    : displays
      ? soundHint || 'Mac display connected'
      : 'Waiting for Mac display — on the Mac open localhost:3001/display.html';
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws?role=pen`);
  ws.onopen = () => {
    setStatus();
    sync();
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'peers') {
      displays = msg.displays;
      setStatus();
    } else if (msg.type === 'need-sync') {
      sync();
    } else if (msg.type === 'transport-state') {
      playBtn.hidden = false; // only the music version of the Mac page has playback
      playBtn.textContent = msg.playing ? '■ Stop' : '▶ Play';
      playBtn.classList.toggle('playing', msg.playing);
      soundHint = msg.soundReady ? '' : 'Turn on sound on the Mac first';
      setStatus();
    }
  };
  ws.onclose = () => {
    setStatus();
    setTimeout(connect, 1000);
  };
}

function toPoint(e) {
  const r = canvas.getBoundingClientRect();
  return [round((e.clientX - r.left) / r.width, 4), round((e.clientY - r.top) / r.width, 4), Math.round(e.timeStamp), round(pressureOf(e), 2)];
}

function flush() {
  if (current && pending.length) send({ type: 'points', id: current.id, pts: pending });
  pending = [];
}

trackStrokes(canvas, {
  toPoint,
  onStart(pt) {
    current = { id: nextId++, pts: [pt] };
    strokes.push(current);
    send({ type: 'start', id: current.id, pt });
    dirty = true;
  },
  onPoints(pts) {
    if (!current) return; // page was cleared mid-stroke
    current.pts.push(...pts);
    pending.push(...pts);
    dirty = true;
  },
  onEnd() {
    if (!current) return;
    flush();
    send({ type: 'end', id: current.id });
    current = null;
  },
});

playBtn.addEventListener('click', () => send({ type: 'transport', action: 'toggle' }));

document.getElementById('clear').addEventListener('click', () => {
  strokes = [];
  current = null;
  pending = [];
  send({ type: 'clear' });
  dirty = true;
});

function frame() {
  requestAnimationFrame(frame); // first, so one bad draw cannot stop the loop
  flush(); // batch points to one message per frame
  if (dirty) {
    dirty = false;
    ctx.clearRect(0, 0, size.w, size.h);
    drawGuides(ctx, 0, 0, size.w, size.h, ROWS);
    ctx.strokeStyle = ctx.fillStyle = INK;
    const map = (x, y) => [x * size.w, y * size.w];
    for (const s of strokes) drawStroke(ctx, s.pts, map, 3);
  }
}

autoFit(canvas, (s) => {
  size = s;
  dirty = true;
  sync();
});
debugOverlay(canvas, () => `ink size   ${Math.round(size.w)} x ${Math.round(size.h)}`);
connect();
setStatus();
requestAnimationFrame(frame);
