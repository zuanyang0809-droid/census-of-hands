// The archive: the 60,000 handwritten digits of MNIST's training set, as
// rebuilt by QMNIST (Yadav & Bottou, 2019), which recovered who wrote each
// one. They come from 539 people: 289 adult employees (NIST series hsf0) and
// 250 high-school students (hsf4), about 116 digits each. Alongside it, an
// index of handwriting fingerprints, to find the people who write like you.

import { SIZE } from './preprocess.js';
import { DATA_BASE, EMBED_DIM } from './model.js';

export const PIX = SIZE * SIZE;
export const MNIST_TOTAL = 60000; // digits in QMNIST's training set
const CHUNK = 5000;

// The archive most recently loaded, so labels can name a digit's writer.
let current = null;

// { pixels: Uint8Array(rows×784), digits: Uint8Array(rows),
//   writers: Uint16Array(rows) (index into people), people: [{ id, series, count }], rows }
export async function loadMnist(rows = MNIST_TOTAL, onProgress = () => {}) {
  const img = new Image();
  img.src = `${DATA_BASE}qmnist/qmnist_images.png`;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = PIX;
  canvas.height = CHUNK;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const pixels = new Uint8Array(rows * PIX);
  for (let start = 0; start < rows; start += CHUNK) {
    const n = Math.min(CHUNK, rows - start);
    ctx.clearRect(0, 0, PIX, CHUNK);
    ctx.drawImage(img, 0, start, PIX, n, 0, 0, PIX, n);
    const data = ctx.getImageData(0, 0, PIX, n).data;
    for (let i = 0; i < n * PIX; i++) pixels[start * PIX + i] = data[i * 4];
    onProgress(start + n, rows);
    await new Promise((r) => {
      // MessageChannel, not setTimeout: not throttled in a background tab
      const ch = new MessageChannel();
      ch.port1.onmessage = r;
      ch.port2.postMessage(0);
    });
  }
  const labels = await (await fetch(`${DATA_BASE}qmnist/qmnist_labels.bin`)).arrayBuffer();
  const digits = new Uint8Array(labels, 0, rows).slice();
  const writers = new Uint16Array(labels.slice(MNIST_TOTAL, MNIST_TOTAL + rows * 2));
  const people = await (await fetch(`${DATA_BASE}qmnist/qmnist_writers.json`)).json();
  current = { pixels, digits, writers, people, rows };
  return current;
}

// The person who wrote archive row `row`.
export function writerOf(row, archive = current) {
  const p = archive.people[archive.writers[row]];
  return { index: archive.writers[row], id: p.id, student: p.series === 4 };
}

export function mnistImage(mnist, row) {
  const img = new Float32Array(PIX);
  const base = row * PIX;
  for (let i = 0; i < PIX; i++) img[i] = mnist.pixels[base + i] / 255;
  return img;
}

// Fingerprints are stored as signed bytes (×127) to keep the file small.
export function quantize(floats) {
  const out = new Int8Array(floats.length);
  for (let i = 0; i < floats.length; i++) out[i] = Math.round(Math.max(-1, Math.min(1, floats[i])) * 127);
  return out;
}

// Int8Array(rows×EMBED_DIM), or null if the archive has not been indexed yet.
export async function loadIndex() {
  const res = await fetch(`${DATA_BASE}model/mnist_fingerprints.bin?t=${Date.now()}`);
  if (!res.ok) return null;
  return new Int8Array(await res.arrayBuffer());
}

// Every archive row of `digit`, closest fingerprint to `query` (a unit-length
// Float32Array) first: [{ row, similarity }].
export function rankAll(index, mnist, query, digit) {
  const rows = Math.min(mnist.rows, index.length / EMBED_DIM);
  const out = [];
  for (let r = 0; r < rows; r++) {
    if (mnist.digits[r] !== digit) continue;
    let dot = 0;
    const base = r * EMBED_DIM;
    for (let j = 0; j < EMBED_DIM; j++) dot += query[j] * index[base + j];
    out.push({ row: r, similarity: Math.min(1, dot / 127) });
  }
  return out.sort((a, b) => b.similarity - a.similarity);
}

// The k closest archive rows of `digit`. With `people`, at most one digit
// per writer — each writer's closest — so k results are k different people.
export function nearest(index, mnist, query, digit, k = 12, { people = false } = {}) {
  const ranked = rankAll(index, mnist, query, digit);
  if (!people) return ranked.slice(0, k);
  const seen = new Set();
  const out = [];
  for (const s of ranked) {
    const w = mnist.writers[s.row];
    if (seen.has(w)) continue;
    seen.add(w);
    out.push(s);
    if (out.length === k) break;
  }
  return out;
}

// One archive digit as dark ink on a transparent 28×28 canvas, cached.
const inkCache = new Map();
export function inkCanvas(mnist, row, rgb = [27, 27, 27]) {
  const key = `${row}:${rgb.join(',')}`;
  let c = inkCache.get(key);
  if (c) return c;
  c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(SIZE, SIZE);
  const base = row * PIX;
  for (let i = 0; i < PIX; i++) {
    img.data[i * 4] = rgb[0];
    img.data[i * 4 + 1] = rgb[1];
    img.data[i * 4 + 2] = rgb[2];
    img.data[i * 4 + 3] = mnist.pixels[base + i];
  }
  ctx.putImageData(img, 0, 0);
  if (inkCache.size > 4000) inkCache.clear();
  inkCache.set(key, c);
  return c;
}

// "Writer 2,341 · student" — the only identity a person has in the archive:
// NIST's writer number, and which group they were collected from.
export function strangerName(row, archive = current) {
  if (!archive) return '';
  const w = writerOf(row, archive);
  return `Writer ${w.id.toLocaleString('en-US')} · ${w.student ? 'student' : 'employee'}`;
}
