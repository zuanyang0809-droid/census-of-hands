// Census of Hands — the Mac side.
// Serves the pages and the archive (QMNIST digits, the trained model and its
// fingerprint index), relays live pen strokes from the iPad to the Mac
// display over a WebSocket, and stores handwriting samples in
// data/samples/<label>/.

import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';
import { LABELS } from './public/labels.js';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(ROOT, 'public');
const SAMPLES_DIR = path.join(ROOT, 'data', 'samples');
const MODEL_DIR = path.join(ROOT, 'data', 'model');
// Only these files are served from data/, so nothing else there is exposed.
// QMNIST: the MNIST training set again, with who wrote each digit.
const QMNIST_DIR = path.join(ROOT, 'data', 'qmnist');
const QMNIST_FILES = new Set(['qmnist_images.png', 'qmnist_labels.bin', 'qmnist_writers.json']);
const MODEL_FILES = new Set(['model.json', 'model.weights.bin', 'meta.json', 'mnist_fingerprints.bin']);
const MAX_INDEX_BYTES = 16 * 1024 * 1024;
const PORT = Number(process.env.PORT) || 3001;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.mp3': 'audio/mpeg',
  '.txt': 'text/plain; charset=utf-8',
};

const DIR_BY_LABEL = new Map(LABELS.map((l) => [l.label, l.dir]));
const KNOWN_DIRS = new Set(LABELS.map((l) => l.dir));

function lanAddresses() {
  const out = [];
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const a of addrs ?? []) if (a.family === 'IPv4' && !a.internal) out.push(a.address);
  }
  return out;
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'content-type': MIME['.json'], 'cache-control': 'no-store' });
  res.end(JSON.stringify(body));
}

async function readRaw(req, limit = MAX_BODY_BYTES) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw Object.assign(new Error('Body too large'), { status: 413 });
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

async function readBody(req) {
  return (await readRaw(req)).toString('utf8');
}

// The training page posts the archive's handwriting fingerprints as raw bytes.
async function saveFingerprints(req, res) {
  const body = await readRaw(req, MAX_INDEX_BYTES);
  if (!body.length) return sendJson(res, 400, { error: 'Empty index' });
  await fs.mkdir(MODEL_DIR, { recursive: true });
  await fs.writeFile(path.join(MODEL_DIR, 'mnist_fingerprints.bin'), body);
  sendJson(res, 201, { ok: true, bytes: body.length });
}

// A sample is a list of strokes; a stroke is a list of [x, y, t, pressure].
function isValidStrokes(strokes) {
  return (
    Array.isArray(strokes) &&
    strokes.length > 0 &&
    strokes.length <= 50 &&
    strokes.every(
      (s) =>
        Array.isArray(s) &&
        s.length > 0 &&
        s.every((p) => Array.isArray(p) && p.length === 4 && p.every(Number.isFinite)),
    )
  );
}

async function saveSample(req, res) {
  const data = JSON.parse(await readBody(req));
  const dir = DIR_BY_LABEL.get(data.label);
  if (!dir) return sendJson(res, 400, { error: 'Unknown label' });
  if (!isValidStrokes(data.strokes)) return sendJson(res, 400, { error: 'Bad strokes' });

  const sample = {
    label: data.label,
    session: String(data.session ?? '').slice(0, 64),
    pointerType: String(data.pointerType ?? '').slice(0, 16),
    box: { w: Number(data.box?.w) || 0, h: Number(data.box?.h) || 0 },
    createdAt: new Date().toISOString(),
    strokes: data.strokes,
  };
  const file = `${Date.now()}-${crypto.randomBytes(3).toString('hex')}.json`;
  await fs.mkdir(path.join(SAMPLES_DIR, dir), { recursive: true });
  await fs.writeFile(path.join(SAMPLES_DIR, dir, file), JSON.stringify(sample));
  sendJson(res, 201, { id: `${dir}/${file}` });
}

async function sampleFiles(dir) {
  try {
    return (await fs.readdir(path.join(SAMPLES_DIR, dir))).filter((f) => f.endsWith('.json')).sort();
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err;
  }
}

async function listSamples() {
  const out = [];
  for (const { dir } of LABELS) {
    for (const f of await sampleFiles(dir)) {
      try {
        const sample = JSON.parse(await fs.readFile(path.join(SAMPLES_DIR, dir, f), 'utf8'));
        out.push({ id: `${dir}/${f}`, ...sample });
      } catch (err) {
        console.warn(`Skipping unreadable sample ${dir}/${f}: ${err.message}`);
      }
    }
  }
  return out;
}

async function countSamples() {
  const counts = {};
  for (const { label, dir } of LABELS) counts[label] = (await sampleFiles(dir)).length;
  return counts;
}

// Sample ids look like "dash/1731200000000-a1b2c3.json"; anything else is refused.
function samplePathFromId(id) {
  const m = /^([a-z0-9]+)\/([\w-]+\.json)$/.exec(id ?? '');
  if (!m || !KNOWN_DIRS.has(m[1])) return null;
  return path.join(SAMPLES_DIR, m[1], m[2]);
}

// The training page posts the model as JSON (topology + base64 weights);
// it is written in the TensorFlow.js layers format so tf.loadLayersModel can read it back.
async function saveModel(req, res) {
  const data = JSON.parse(await readBody(req));
  if (!data.modelTopology || !Array.isArray(data.weightSpecs) || typeof data.weightData !== 'string') {
    return sendJson(res, 400, { error: 'Bad model' });
  }
  await fs.mkdir(MODEL_DIR, { recursive: true });
  const modelJson = {
    format: 'layers-model',
    generatedBy: data.generatedBy ?? 'TensorFlow.js',
    convertedBy: null,
    modelTopology: data.modelTopology,
    weightsManifest: [{ paths: ['model.weights.bin'], weights: data.weightSpecs }],
  };
  await fs.writeFile(path.join(MODEL_DIR, 'model.weights.bin'), Buffer.from(data.weightData, 'base64'));
  await fs.writeFile(path.join(MODEL_DIR, 'model.json'), JSON.stringify(modelJson));
  await fs.writeFile(path.join(MODEL_DIR, 'meta.json'), JSON.stringify({ ...data.meta, savedAt: new Date().toISOString() }, null, 2));
  sendJson(res, 201, { ok: true });
}

async function serveFile(res, file, contentType) {
  try {
    const body = await fs.readFile(file);
    res.writeHead(200, { 'content-type': contentType, 'cache-control': 'no-store' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
  }
}

async function serveStatic(res, pathname) {
  const rel = pathname === '/' ? 'index.html' : decodeURIComponent(pathname).replace(/^\/+/, '');
  const file = path.normalize(path.join(PUBLIC_DIR, rel));
  if (!file.startsWith(PUBLIC_DIR + path.sep)) {
    res.writeHead(403).end();
    return;
  }
  try {
    const body = await fs.readFile(file);
    res.writeHead(200, {
      'content-type': MIME[path.extname(file)] ?? 'application/octet-stream',
      'cache-control': 'no-store',
    });
    res.end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (url.pathname === '/api/samples' && req.method === 'POST') return await saveSample(req, res);
    if (url.pathname === '/api/samples' && req.method === 'GET') return sendJson(res, 200, await listSamples());
    if (url.pathname === '/api/stats' && req.method === 'GET') return sendJson(res, 200, await countSamples());
    if (url.pathname === '/api/model' && req.method === 'POST') return await saveModel(req, res);
    if (url.pathname === '/api/fingerprints' && req.method === 'POST') return await saveFingerprints(req, res);
    if (url.pathname.startsWith('/qmnist/') && req.method === 'GET') {
      const name = url.pathname.slice('/qmnist/'.length);
      if (!QMNIST_FILES.has(name)) return sendJson(res, 404, { error: 'Not found' });
      const type = name.endsWith('.png') ? MIME['.png'] : name.endsWith('.json') ? MIME['.json'] : 'application/octet-stream';
      return await serveFile(res, path.join(QMNIST_DIR, name), type);
    }
    if (url.pathname.startsWith('/model/') && req.method === 'GET') {
      const name = url.pathname.slice('/model/'.length);
      if (!MODEL_FILES.has(name)) return sendJson(res, 404, { error: 'Not found' });
      return await serveFile(res, path.join(MODEL_DIR, name), name.endsWith('.json') ? MIME['.json'] : 'application/octet-stream');
    }
    if (url.pathname === '/api/info' && req.method === 'GET') {
      return sendJson(res, 200, { port: PORT, addresses: lanAddresses() });
    }
    if (url.pathname.startsWith('/api/samples/') && req.method === 'DELETE') {
      const file = samplePathFromId(decodeURIComponent(url.pathname.slice('/api/samples/'.length)));
      if (!file) return sendJson(res, 400, { error: 'Bad sample id' });
      await fs.rm(file);
      return sendJson(res, 200, { ok: true });
    }
    if (req.method === 'GET') return await serveStatic(res, url.pathname);
    res.writeHead(405).end();
  } catch (err) {
    const status = err.status ?? (err instanceof SyntaxError ? 400 : err.code === 'ENOENT' ? 404 : 500);
    if (status === 500) console.error(err);
    sendJson(res, status, { error: err.message });
  }
});

// Live strokes: pens (iPad) send, displays (Mac) receive.
const wss = new WebSocketServer({ server, path: '/ws' });
wss.on('error', () => {}); // ws re-emits the HTTP server's errors; those are handled on `server`
const pens = new Set();
const displays = new Set();

function broadcast(clients, message) {
  for (const c of clients) if (c.readyState === c.OPEN) c.send(message);
}

function announcePeers() {
  const msg = JSON.stringify({ type: 'peers', pens: pens.size, displays: displays.size });
  broadcast(pens, msg);
  broadcast(displays, msg);
}

wss.on('connection', (ws, req) => {
  const role = new URL(req.url, 'http://localhost').searchParams.get('role') === 'pen' ? 'pen' : 'display';
  const group = role === 'pen' ? pens : displays;
  group.add(ws);
  // A display that joins late asks the pens to resend what is already on the page.
  if (role === 'display') broadcast(pens, JSON.stringify({ type: 'need-sync' }));
  announcePeers();

  // Pens send strokes and play/stop to displays; displays send playback state back.
  ws.on('message', (data) => {
    broadcast(role === 'pen' ? displays : pens, data.toString());
  });
  ws.on('close', () => {
    group.delete(ws);
    announcePeers();
  });
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`Port ${PORT} is already in use — Census of Hands is probably already running.`);
    console.log(`Just open http://localhost:${PORT}/hands.html. To restart it, stop the other one first (Ctrl+C in its terminal).`);
    process.exit(1);
  }
  throw err;
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Census of Hands running on http://localhost:${PORT}`);
  console.log(`  Mac, mouse:    http://localhost:${PORT}/hands.html`);
  for (const ip of lanAddresses()) {
    console.log(`  iPad pen:      http://${ip}:${PORT}/pen.html`);
    console.log(`  iPad collect:  http://${ip}:${PORT}/collect.html`);
  }
  console.log(`  Mac + iPad:    http://localhost:${PORT}/census.html`);
  console.log(`  Mac display:   http://localhost:${PORT}/display.html  (the earlier music version)`);
  console.log(`  Mac review:    http://localhost:${PORT}/review.html`);
  console.log(`  Mac train:     http://localhost:${PORT}/train.html`);
});
