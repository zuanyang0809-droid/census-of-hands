// The recogniser: a small convolutional network over 28×28 images.
// Shared by the training page and the live pages.
// Needs TensorFlow.js loaded first as the global `tf` (vendor/tf.min.js).
//
// It learns two things at once from the same drawing:
//   digit   which of the ten digits it is, or "other". The symbols you
//           collected that are not digits (-, scribbles) teach "other".
//   writer  which person wrote it: one of the archive's 539 writers, or you.
// To read digits well, a network learns to ignore how each person writes;
// to tell writers apart, it has to keep exactly that. So the writer branch's
// last hidden layer ("style", EMBED_DIM numbers) is the handwriting
// fingerprint used to find who writes most like you.

import { SIZE } from './preprocess.js';

export const CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'other'];
export const EMBED_DIM = 64;

// Which class a collected sample teaches: digits are themselves, the rest is
// "other" — except "/", which looks like a slanted 1 and taught the model to
// reject real 1s, so it is left out (null).
export function classOfLabel(label) {
  if (/^[0-9]$/.test(label)) return label;
  return label === '/' ? null : 'other';
}

// `writers`: how many people the writer branch chooses between (the
// archive's writers plus one for you).
export function buildModel(writers) {
  const input = tf.input({ shape: [SIZE, SIZE, 1] });
  let x = tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu' }).apply(input);
  x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);
  x = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' }).apply(x);
  x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);
  const features = tf.layers.flatten().apply(x);

  let d = tf.layers.dropout({ rate: 0.25 }).apply(features);
  d = tf.layers.dense({ units: 64, activation: 'relu' }).apply(d);
  const digit = tf.layers.dense({ units: CLASSES.length, activation: 'softmax', name: 'digit' }).apply(d);

  const style = tf.layers.dense({ units: EMBED_DIM, name: 'style' }).apply(features);
  let w = tf.layers.dropout({ rate: 0.2 }).apply(style);
  const writer = tf.layers.dense({ units: writers, activation: 'softmax', name: 'writer' }).apply(w);

  const model = tf.model({ inputs: input, outputs: [digit, writer] });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: ['categoricalCrossentropy', 'sparseCategoricalCrossentropy'],
    metrics: ['accuracy'],
  });
  return model;
}

// Digit probabilities only, whatever the model also predicts.
const digitOutput = (out) => (Array.isArray(out) ? out[0] : out);

// Probabilities for each class, in CLASSES order, for one or many 28×28 images.
export function predictMany(model, images) {
  return tf.tidy(() => {
    const flat = new Float32Array(images.length * SIZE * SIZE);
    images.forEach((img, i) => flat.set(img, i * SIZE * SIZE));
    const probs = digitOutput(model.predict(tf.tensor4d(flat, [images.length, SIZE, SIZE, 1]))).arraySync();
    return probs;
  });
}

export function predict(model, image) {
  return predictMany(model, [image])[0];
}

// A model that stops at the fingerprint layer (shares weights with `model`):
// "style" for handwriting, or "digit" for the layer the digit is read from
// (used only to compare the two).
export function fingerprintModel(model, which = 'style') {
  const layer = which === 'style' ? model.getLayer('style') : model.getLayer('digit').inboundNodes[0].inboundLayers[0];
  return tf.model({ inputs: model.inputs, outputs: layer.output });
}

// Fingerprints for `count` images in `flat` (Float32Array, count×784, 0..1),
// each scaled to length 1 so a dot product is the cosine similarity.
export function fingerprints(fpModel, flat, count) {
  const out = tf.tidy(() => fpModel.predict(tf.tensor4d(flat, [count, SIZE, SIZE, 1])).dataSync());
  for (let i = 0; i < count; i++) {
    let norm = 0;
    for (let j = 0; j < EMBED_DIM; j++) norm += out[i * EMBED_DIM + j] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < EMBED_DIM; j++) out[i * EMBED_DIM + j] /= norm;
  }
  return out;
}

// Returns { model, meta } or null when nothing has been trained yet.
export async function loadModel() {
  const res = await fetch('/model/meta.json');
  if (!res.ok) return null;
  const meta = await res.json();
  if (JSON.stringify(meta.classes) !== JSON.stringify(CLASSES) || !meta.writers) {
    throw new Error('The saved model is from an older version — train it again at /train.html.');
  }
  const model = await tf.loadLayersModel(`/model/model.json?t=${Date.now()}`);
  return { model, meta };
}

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function concatBuffers(data) {
  if (!Array.isArray(data)) return data;
  const total = data.reduce((n, b) => n + b.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const b of data) {
    out.set(new Uint8Array(b), offset);
    offset += b.byteLength;
  }
  return out.buffer;
}

// Save to the Mac (data/model/) through the server.
export async function saveModel(model, meta) {
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const res = await fetch('/api/model', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          modelTopology: artifacts.modelTopology,
          weightSpecs: artifacts.weightSpecs,
          weightData: toBase64(concatBuffers(artifacts.weightData)),
          generatedBy: `TensorFlow.js v${tf.version.tfjs}`,
          meta: { ...meta, classes: CLASSES },
        }),
      });
      if (!res.ok) throw new Error(`Saving failed: ${(await res.json()).error}`);
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }),
  );
}
