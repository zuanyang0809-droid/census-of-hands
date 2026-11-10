// Mac side: look through every saved sample, grouped by symbol, and delete bad ones.

import { LABELS, TARGET_PER_LABEL } from './labels.js';
import { INK, drawStroke } from './ink.js';

const TILE = 76;
const PAD = 10;

const groupsEl = document.getElementById('groups');
const summaryEl = document.getElementById('summary');

// Fit a sample to the tile by its own bounding box, keeping proportions,
// so a dash stays a flat line and a 1 stays tall.
function drawTile(canvas, strokes) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = TILE * dpr;
  canvas.height = TILE * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pts = strokes.flat();
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const bw = Math.max(...xs) - minX;
  const bh = Math.max(...ys) - minY;
  const scale = (TILE - PAD * 2) / Math.max(bw, bh, 1);
  const ox = (TILE - bw * scale) / 2;
  const oy = (TILE - bh * scale) / 2;

  ctx.strokeStyle = ctx.fillStyle = INK;
  for (const s of strokes) drawStroke(ctx, s, (x, y) => [ox + (x - minX) * scale, oy + (y - minY) * scale], 2);
}

async function remove(sample, tile) {
  if (!confirm(`Delete this "${sample.label}" sample? This removes the file.`)) return;
  const res = await fetch(`/api/samples/${encodeURIComponent(sample.id)}`, { method: 'DELETE' });
  if (!res.ok) {
    alert(`Could not delete: ${(await res.json()).error}`);
    return;
  }
  tile.remove();
  load();
}

async function load() {
  let samples;
  try {
    samples = await (await fetch('/api/samples')).json();
  } catch {
    summaryEl.textContent = 'Could not reach the server.';
    return;
  }
  const sessions = new Set(samples.map((s) => s.session));
  summaryEl.textContent = `${samples.length} of ${LABELS.length * TARGET_PER_LABEL} · ${sessions.size} session${sessions.size === 1 ? '' : 's'}`;

  groupsEl.innerHTML = '';
  for (const { label, hint } of LABELS) {
    const mine = samples.filter((s) => s.label === label);
    const h2 = document.createElement('h2');
    h2.innerHTML = `<span class="glyph"></span><span></span><span class="muted"></span>`;
    h2.children[0].textContent = label === 'other' ? '✕' : label;
    h2.children[1].textContent = `${mine.length}/${TARGET_PER_LABEL}`;
    h2.children[2].textContent = hint;
    const grid = document.createElement('div');
    grid.className = 'grid';
    if (!mine.length) grid.innerHTML = '<span class="empty muted">No samples yet</span>';
    for (const sample of mine) {
      const tile = document.createElement('div');
      tile.className = 'tile';
      tile.title = `${sample.session} · ${sample.pointerType} · ${sample.strokes.length} stroke(s)`;
      const canvas = document.createElement('canvas');
      const del = document.createElement('button');
      del.type = 'button';
      del.textContent = '×';
      del.addEventListener('click', () => remove(sample, tile));
      tile.append(canvas, del);
      grid.append(tile);
      drawTile(canvas, sample.strokes);
    }
    groupsEl.append(h2, grid);
  }
}

document.getElementById('reload').addEventListener('click', load);
load();
