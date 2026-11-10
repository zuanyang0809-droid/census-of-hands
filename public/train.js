// Mac side: build the training set, train, evaluate, save, index the
// archive, and try the model.
//
// Training set:
//   digits 0–9  the first 54,000 archive digits (+ your samples of 0–7, augmented)
//   other       your - and scribble samples, augmented
// Every image also says who wrote it: one of the archive's 539 writers, or
// you (all your samples), so the model learns to tell hands apart.
// A couple of your samples per class, and the last 6,000 archive digits, are
// held out to test on.
//
// After training, every one of the 60,000 archive digits is fingerprinted
// with the model's style layer and saved, so the live pages can find the
// people who write most like you. A test then checks the fingerprint: for a
// held-out digit, is its closest match written by the same person?

import { LABELS, TARGET_PER_LABEL } from './labels.js';
import { SIZE, drawPreview, rasterize } from './preprocess.js';
import { CLASSES, EMBED_DIM, buildModel, classOfLabel, fingerprintModel, fingerprints, loadModel, predict, predictMany, saveModel } from './model.js';
import { MNIST_TOTAL, PIX, inkCanvas, loadIndex, loadMnist, mnistImage, nearest, quantize, strangerName } from './archive.js';
import { INK, autoFit, drawStroke, trackStrokes } from './ink.js';

const MNIST_TRAIN_ROWS = 54000; // rows before this train, the rest test
const PER_CLASS = 1500; // images of "other", which only you provide
const MIN_AUGMENT = 30; // copies of each of your samples, at least
const HOLD_OUT = 2; // per class, when a class has 5 or more samples
const EPOCHS = 6;
const BATCH = 64;
const INDEX_BATCH = 1000;
const SAME_HAND_TESTS = 1000; // held-out digits used to test the fingerprint

const $ = (id) => document.getElementById(id);
const glyph = (label) => (label === 'other' ? '✕' : label);
const pct = (v) => `${(v * 100).toFixed(1)}%`;

let model = null;
let fpModel = null;
let mnist = null;
let index = null;
let busy = false;

function log(line) {
  $('log').textContent += `${line}\n`;
  $('log').scrollTop = $('log').scrollHeight;
}

// Let the page breathe between chunks of work. A MessageChannel message,
// unlike setTimeout, is not slowed to once a minute when the tab is in the
// background, so training keeps going if you switch away.
const yieldChannel = new MessageChannel();
const yieldQueue = [];
yieldChannel.port1.onmessage = () => yieldQueue.shift()?.();
const nextFrame = () =>
  new Promise((resolve) => {
    yieldQueue.push(resolve);
    yieldChannel.port2.postMessage(0);
  });

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const rand = (lo, hi) => lo + Math.random() * (hi - lo);
const noise = () => Math.random() + Math.random() + Math.random() - 1.5;

// Small random rotation, stretch, slant and wobble — new but plausible handwriting.
function augmentStrokes(strokes) {
  const pts = strokes.flat();
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1);
  const a = (rand(-10, 10) * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  const sx = rand(0.85, 1.15);
  const sy = rand(0.85, 1.15);
  const shear = rand(-0.15, 0.15);
  const wobble = size * 0.012;
  return strokes.map((s) =>
    s.map(([x, y]) => {
      const u = (x - cx + shear * (y - cy)) * sx;
      const v = (y - cy) * sy;
      return [u * cos - v * sin + noise() * wobble, u * sin + v * cos + noise() * wobble];
    }),
  );
}

async function fetchSamples() {
  return (await fetch('/api/samples')).json();
}

function renderCounts(samples) {
  const counts = Object.fromEntries(LABELS.map((l) => [l.label, 0]));
  for (const s of samples) counts[s.label] = (counts[s.label] ?? 0) + 1;
  const head = LABELS.map((l) => `<th>${glyph(l.label)}</th>`).join('');
  const cells = LABELS.map((l) => `<td class="${counts[l.label] < TARGET_PER_LABEL ? 'low' : ''}">${counts[l.label]}</td>`).join('');
  $('counts').innerHTML = `<table><tr>${head}</tr><tr>${cells}</tr></table>`;
  $('data-note').textContent = '8 and 9 come from the archive only. - and ✕ teach the model what “not a digit” looks like; / is left out because it looks like a slanted 1. All of your samples are labelled as one more writer: you.';
}

async function ensureMnist() {
  if (mnist) return mnist;
  log(`Loading the archive (${MNIST_TOTAL.toLocaleString()} digits, with their writers)…`);
  mnist = await loadMnist(MNIST_TOTAL);
  return mnist;
}

async function buildDataset(samples) {
  const images = [];
  const labels = [];
  const writers = [];
  const you = mnist.people.length; // you are the last writer
  const add = (img, cls, writer) => {
    images.push(img);
    labels.push(cls);
    writers.push(writer);
  };

  // Archive training part: every digit, as itself, with its writer.
  const fromMnist = new Array(CLASSES.length).fill(0);
  for (let r = 0; r < MNIST_TRAIN_ROWS; r++) {
    const cls = CLASSES.indexOf(String(mnist.digits[r]));
    fromMnist[cls]++;
    add(mnistImage(mnist, r), cls, mnist.writers[r]);
    if (r % 5000 === 4999) await nextFrame();
  }

  // Your samples: hold a few out, augment the rest.
  const byClass = CLASSES.map(() => []);
  for (const s of samples) {
    const cls = classOfLabel(s.label);
    if (cls !== null) byClass[CLASSES.indexOf(cls)].push(s);
  }
  const ownTest = [];
  const examples = [];
  for (let c = 0; c < CLASSES.length; c++) {
    const list = shuffle([...byClass[c]]);
    const hold = list.length >= 5 ? HOLD_OUT : 0;
    for (const s of list.slice(0, hold)) ownTest.push({ cls: c, image: rasterize(s.strokes) });
    const train = list.slice(hold);
    if (!train.length) {
      log(`  ${glyph(CLASSES[c])}: ${fromMnist[c]} MNIST${fromMnist[c] ? '' : ' — nothing to learn this from!'}`);
      continue;
    }
    const copies = Math.max(MIN_AUGMENT, Math.ceil((PER_CLASS - fromMnist[c]) / train.length));
    for (let k = 0; k < copies; k++) {
      for (const [i, s] of train.entries()) {
        const img = rasterize(augmentStrokes(s.strokes), { lineWidth: rand(2, 3.2) });
        add(img, c, you);
        if (k < 2 && i === 0) examples.push({ img, cls: c });
      }
      if (k % 10 === 0) await nextFrame();
    }
    log(`  ${glyph(CLASSES[c])}: ${fromMnist[c]} MNIST + ${train.length} yours × ${copies} = ${fromMnist[c] + train.length * copies}`);
  }

  const mnistTest = [];
  for (let r = MNIST_TRAIN_ROWS; r < mnist.rows; r++) {
    mnistTest.push({ image: mnistImage(mnist, r), cls: CLASSES.indexOf(String(mnist.digits[r])) });
  }
  return { images, labels, writers, ownTest, mnistTest, examples };
}

function showExamples(examples) {
  const strip = $('examples');
  strip.innerHTML = '';
  for (const { img, cls } of examples) {
    const fig = document.createElement('figure');
    const c = document.createElement('canvas');
    drawPreview(c, img, 2);
    fig.append(c, glyph(CLASSES[cls]));
    strip.append(fig);
  }
}

function evaluate(set) {
  if (!set.length) return [];
  const results = [];
  for (let i = 0; i < set.length; i += 500) {
    const probs = predictMany(model, set.slice(i, i + 500).map((t) => t.image));
    probs.forEach((p, j) => results.push({ ...set[i + j], guess: p.indexOf(Math.max(...p)) }));
  }
  return results;
}

function showResults(mnistResults, ownResults) {
  $('results').hidden = false;
  const acc = (rs) => (rs.length ? rs.filter((r) => r.guess === r.cls).length / rs.length : null);
  const mAcc = acc(mnistResults);
  const oAcc = acc(ownResults);
  $('mnist-acc').textContent = mAcc === null ? '—' : pct(mAcc);
  $('own-acc').textContent = oAcc === null ? '—' : `${pct(oAcc)} (${ownResults.filter((r) => r.guess === r.cls).length}/${ownResults.length})`;

  const grid = CLASSES.map(() => CLASSES.map(() => 0));
  for (const r of ownResults) grid[r.cls][r.guess]++;
  let html = `<table><tr><th></th>${CLASSES.map((c) => `<th>${glyph(c)}</th>`).join('')}</tr>`;
  CLASSES.forEach((c, i) => {
    html += `<tr><th>${glyph(c)}</th>`;
    grid[i].forEach((n, j) => {
      html += `<td class="${n === 0 ? '' : i === j ? 'hit' : 'miss'}">${n || ''}</td>`;
    });
    html += '</tr>';
  });
  $('confusion').innerHTML = `${html}</table>`;

  const misses = ownResults.filter((r) => r.guess !== r.cls);
  $('misses').innerHTML = misses.length ? '' : '<span class="muted">None.</span>';
  for (const r of misses) {
    const fig = document.createElement('figure');
    const c = document.createElement('canvas');
    drawPreview(c, r.image, 2);
    fig.append(c, `${glyph(CLASSES[r.cls])} → ${glyph(CLASSES[r.guess])}`);
    $('misses').append(fig);
  }
  return { mnistAccuracy: mAcc, ownAccuracy: oAcc, ownTested: ownResults.length };
}

// Fingerprints for every archive digit from one of the model's layers.
async function fingerprintAll(which) {
  const fp = fingerprintModel(model, which);
  const all = new Float32Array(mnist.rows * EMBED_DIM);
  const flat = new Float32Array(INDEX_BATCH * PIX);
  for (let start = 0; start < mnist.rows; start += INDEX_BATCH) {
    const n = Math.min(INDEX_BATCH, mnist.rows - start);
    for (let i = 0; i < n * PIX; i++) flat[i] = mnist.pixels[start * PIX + i] / 255;
    all.set(fingerprints(fp, flat.subarray(0, n * PIX), n), start * EMBED_DIM);
    if ((start / INDEX_BATCH) % 20 === 19) log(`  ${(start + n).toLocaleString()} done`);
    await nextFrame();
  }
  return all;
}

// For held-out digits, how often is the closest training digit of the same
// number written by the same person? Chance is about 1 in 539.
async function sameHandRate(all) {
  let hits = 0;
  const tests = Math.min(SAME_HAND_TESTS, mnist.rows - MNIST_TRAIN_ROWS);
  for (let t = 0; t < tests; t++) {
    const q = MNIST_TRAIN_ROWS + t;
    let best = -Infinity;
    let bestRow = -1;
    for (let r = 0; r < MNIST_TRAIN_ROWS; r++) {
      if (mnist.digits[r] !== mnist.digits[q]) continue;
      let dot = 0;
      for (let j = 0; j < EMBED_DIM; j++) dot += all[q * EMBED_DIM + j] * all[r * EMBED_DIM + j];
      if (dot > best) {
        best = dot;
        bestRow = r;
      }
    }
    if (mnist.writers[bestRow] === mnist.writers[q]) hits++;
    if (t % 100 === 99) await nextFrame();
  }
  return hits / tests;
}

// Fingerprint every archive digit with the style layer and save the index;
// then test it against the digit layer the earlier version used.
async function buildIndex() {
  log(`Indexing the archive: fingerprinting ${mnist.rows.toLocaleString()} handwritten digits…`);
  fpModel = fingerprintModel(model);
  const all = await fingerprintAll('style');
  index = quantize(all);
  const res = await fetch('/api/fingerprints', { method: 'POST', headers: { 'content-type': 'application/octet-stream' }, body: index });
  if (!res.ok) throw new Error(`Saving the index failed: ${(await res.json()).error}`);
  log(`Index saved (${(index.byteLength / 1e6).toFixed(1)} MB).`);

  log('Testing: does the closest match come from the same person?');
  const style = await sameHandRate(all);
  log(`  style fingerprint: ${pct(style)} of held-out digits`);
  const digitLayer = await sameHandRate(await fingerprintAll('digit'));
  log(`  digit layer (the old fingerprint): ${pct(digitLayer)}`);
  log(`  chance: ${pct(1 / mnist.people.length)}`);
  return { sameHandStyle: style, sameHandDigitLayer: digitLayer };
}

async function train() {
  if (busy) return;
  busy = true;
  $('train').disabled = true;
  $('log').textContent = '';
  try {
    const t0 = performance.now();
    const samples = await fetchSamples();
    renderCounts(samples);
    await tf.ready();
    log(`TensorFlow.js ${tf.version.tfjs}, backend ${tf.getBackend()}`);
    await ensureMnist();
    log('Building training set…');
    const data = await buildDataset(samples);
    showExamples(data.examples);
    log(`${data.images.length} training images, ${data.ownTest.length} of yours held out.`);

    const n = data.images.length;
    const flat = new Float32Array(n * PIX);
    data.images.forEach((img, i) => flat.set(img, i * PIX));
    const xs = tf.tensor4d(flat, [n, SIZE, SIZE, 1]);
    const ys = tf.oneHot(tf.tensor1d(data.labels, 'int32'), CLASSES.length);
    const yw = tf.tensor1d(data.writers, 'float32');
    const writers = mnist.people.length + 1;

    model?.dispose();
    model = buildModel(writers);
    log(`Training ${EPOCHS} rounds…`);
    // TensorFlow.js normally pauses for an animation frame between batches,
    // which never comes while the tab is hidden — training would freeze if
    // you switched tabs. Yield with a timer every few batches instead.
    await model.fit(xs, [ys, yw], {
      epochs: EPOCHS,
      batchSize: BATCH,
      shuffle: true,
      yieldEvery: 'never',
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const digitAcc = logs.digit_acc ?? logs.digit_accuracy;
          const writerAcc = logs.writer_acc ?? logs.writer_accuracy;
          log(`  round ${epoch + 1}/${EPOCHS}  loss ${logs.loss.toFixed(3)}  digit ${pct(digitAcc)}  writer ${pct(writerAcc)}`);
          await nextFrame();
        },
        onBatchEnd: async (batch) => {
          if (batch % 25 === 0) await nextFrame();
        },
      },
    });
    xs.dispose();
    ys.dispose();
    yw.dispose();

    const scores = showResults(evaluate(data.mnistTest), evaluate(data.ownTest));
    const counts = Object.fromEntries(LABELS.map((l) => [l.label, samples.filter((s) => s.label === l.label).length]));
    const meta = { ...scores, samples: counts, trainingImages: n, writers, people: mnist.people.length };
    await saveModel(model, meta);
    log('Model saved to the Mac (data/model).');
    const tests = await buildIndex();
    await saveModel(model, { ...meta, ...tests });
    log(`Done in ${Math.round((performance.now() - t0) / 1000)} s.`);
    $('model-status').textContent = `Model trained ${new Date().toLocaleTimeString()}`;
    refreshTry();
  } catch (err) {
    console.error(err);
    log(`Error: ${err.message}`);
  } finally {
    busy = false;
    $('train').disabled = false;
  }
}

// ---- Try it ----
const pad = $('pad');
const padCtx = pad.getContext('2d');
let padSize = { w: 1, h: 1 };
let padStrokes = [];
let padCurrent = null;

function drawPad() {
  padCtx.clearRect(0, 0, padSize.w, padSize.h);
  padCtx.strokeStyle = padCtx.fillStyle = INK;
  for (const s of padStrokes) drawStroke(padCtx, s, (x, y) => [x, y], 5);
}

function renderBars(probs) {
  const bars = $('bars');
  bars.innerHTML = '';
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  CLASSES.forEach((c, i) => {
    const p = probs ? probs[i] : 0;
    const name = document.createElement('span');
    name.textContent = glyph(c);
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = `<i style="width:${(p * 100).toFixed(1)}%"></i>`;
    const val = document.createElement('span');
    val.textContent = probs ? pct(p) : '';
    if (i === best) name.className = val.className = 'top';
    bars.append(name, bar, val);
  });
  $('guess').textContent = best >= 0 ? glyph(CLASSES[best]) : '';
  return best;
}

function renderStrangers(img, best) {
  const box = $('strangers');
  box.innerHTML = '';
  if (!model || !index || !mnist || best < 0 || CLASSES[best] === 'other') return;
  fpModel ??= fingerprintModel(model);
  const query = fingerprints(fpModel, img, 1);
  for (const { row, similarity } of nearest(index, mnist, query, Number(CLASSES[best]), 8)) {
    const fig = document.createElement('figure');
    const c = document.createElement('canvas');
    c.width = c.height = 56;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(inkCanvas(mnist, row), 0, 0, 56, 56);
    fig.append(c, `${strangerName(row)} · ${Math.round(similarity * 100)}%`);
    box.append(fig);
  }
}

function refreshTry() {
  if (!padStrokes.length) {
    renderBars(null);
    $('seen').getContext('2d').clearRect(0, 0, 84, 84);
    $('strangers').innerHTML = '';
    return;
  }
  const img = rasterize(padStrokes);
  drawPreview($('seen'), img, 3);
  const best = model ? renderBars(predict(model, img)) : renderBars(null);
  renderStrangers(img, best);
}

trackStrokes(pad, {
  toPoint: (e) => {
    const r = pad.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top, e.timeStamp, 0.5];
  },
  onStart: (pt) => {
    padCurrent = [pt];
    padStrokes.push(padCurrent);
    drawPad();
  },
  onPoints: (pts) => {
    if (!padCurrent) return;
    padCurrent.push(...pts);
    drawPad();
  },
  onEnd: () => {
    padCurrent = null;
    refreshTry();
  },
});
autoFit(pad, (s) => {
  padSize = s;
  drawPad();
});
$('clear').addEventListener('click', () => {
  padStrokes = [];
  drawPad();
  refreshTry();
});
$('train').addEventListener('click', train);

async function init() {
  renderBars(null);
  try {
    renderCounts(await fetchSamples());
  } catch {
    $('counts').textContent = 'Could not reach the server.';
  }
  try {
    const saved = await loadModel();
    if (!saved) {
      $('model-status').textContent = 'No model yet — press Train.';
      return;
    }
    model = saved.model;
    const m = saved.meta;
    $('model-status').textContent = `Saved model from ${new Date(m.savedAt).toLocaleString()} — MNIST ${m.mnistAccuracy == null ? '—' : pct(m.mnistAccuracy)}, yours ${m.ownAccuracy == null ? '—' : pct(m.ownAccuracy)}`;
    index = await loadIndex();
    if (index) {
      mnist = await loadMnist(MNIST_TOTAL);
      refreshTry();
    }
  } catch (err) {
    $('model-status').textContent = err.message;
  }
}

init();
