// Turn pen strokes into a 28×28 image the way MNIST digits were made:
// white ink on black, the symbol scaled (keeping its proportions) so its
// longer side is 20 px, then shifted so its centre of mass sits at (14, 14).
// Training and live recognition both go through this, so the model always
// sees the same kind of picture.

export const SIZE = 28;
const BOX = 20;
const DEFAULT_LINE = 2.6; // stroke width in 28-px space
const SUPER = 4; // draw at 4× and scale down, for MNIST-like soft edges
const MAX_SHIFT = 4;

const big = document.createElement('canvas');
big.width = big.height = SIZE * SUPER;
const bigCtx = big.getContext('2d');
const small = document.createElement('canvas');
small.width = small.height = SIZE;
const smallCtx = small.getContext('2d', { willReadFrequently: true });
smallCtx.imageSmoothingEnabled = true;
smallCtx.imageSmoothingQuality = 'high';

// strokes: [[ [x, y, ...], ... ], ...] in any units. Returns Float32Array(784), 0..1.
export function rasterize(strokes, { lineWidth = DEFAULT_LINE } = {}) {
  const pts = strokes.flat();
  const out = new Float32Array(SIZE * SIZE);
  if (!pts.length) return out;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const w = maxX - minX;
  const h = maxY - minY;
  const s = BOX / Math.max(w, h, 1e-6);
  const ox = (SIZE - w * s) / 2;
  const oy = (SIZE - h * s) / 2;

  const draw = (dx, dy) => {
    bigCtx.fillStyle = '#000';
    bigCtx.fillRect(0, 0, big.width, big.height);
    bigCtx.strokeStyle = bigCtx.fillStyle = '#fff';
    bigCtx.lineWidth = lineWidth * SUPER;
    bigCtx.lineCap = 'round';
    bigCtx.lineJoin = 'round';
    const X = (x) => ((x - minX) * s + ox + dx) * SUPER;
    const Y = (y) => ((y - minY) * s + oy + dy) * SUPER;
    for (const stroke of strokes) {
      if (!stroke.length) continue;
      bigCtx.beginPath();
      if (stroke.length === 1) {
        bigCtx.arc(X(stroke[0][0]), Y(stroke[0][1]), (lineWidth * SUPER) / 2, 0, Math.PI * 2);
        bigCtx.fill();
        continue;
      }
      bigCtx.moveTo(X(stroke[0][0]), Y(stroke[0][1]));
      for (let i = 1; i < stroke.length; i++) bigCtx.lineTo(X(stroke[i][0]), Y(stroke[i][1]));
      bigCtx.stroke();
    }
    smallCtx.drawImage(big, 0, 0, SIZE, SIZE);
    const data = smallCtx.getImageData(0, 0, SIZE, SIZE).data;
    for (let i = 0; i < out.length; i++) out[i] = data[i * 4] / 255;
  };

  // Where the 28×28 image sits in the strokes' own units, so a page can lay
  // the image exactly over what was drawn: image pixel 0 is at (x, y) and
  // the image is `size` across.
  const place = (dx, dy) => {
    out.place = { x: minX - (ox + dx) / s, y: minY - (oy + dy) / s, size: SIZE / s };
  };

  // Pass 1: centred by bounding box. Pass 2: shifted to centre of mass.
  draw(0, 0);
  place(0, 0);
  let mass = 0, mx = 0, my = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = out[y * SIZE + x];
      mass += v;
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  }
  if (mass > 0) {
    const clamp = (v) => Math.max(-MAX_SHIFT, Math.min(MAX_SHIFT, v));
    const dx = clamp(SIZE / 2 - mx / mass);
    const dy = clamp(SIZE / 2 - my / mass);
    if (Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05) {
      draw(dx, dy);
      place(dx, dy);
    }
  }
  return out;
}

// Paint a 28×28 image (Float32Array 0..1) into a canvas, scaled up, for previews.
export function drawPreview(canvas, pixels, scale = 3) {
  canvas.width = canvas.height = SIZE * scale;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.round(pixels[i] * 255);
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  small.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, SIZE * scale, SIZE * scale);
}
