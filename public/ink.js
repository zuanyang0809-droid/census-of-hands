// Shared input and drawing helpers for every page.
// A point is [x, y, t, pressure]; a stroke is an array of points.

export const INK = '#1c1c1e';
export const GUIDE = 'rgba(0, 0, 0, 0.08)';

export function pressureOf(e) {
  // Apple Pencil reports real pressure; mouse and finger get a neutral value.
  return e.pointerType === 'pen' && e.pressure > 0 ? e.pressure : 0.5;
}

// Stop iPad Safari from scrolling, zooming or bouncing while writing.
export function lockPage() {
  for (const type of ['gesturestart', 'gesturechange', 'dblclick']) {
    document.addEventListener(type, (e) => e.preventDefault());
  }
  document.addEventListener('touchmove', (e) => e.preventDefault(), { passive: false });
}

// One stroke at a time. Once an Apple Pencil has been seen, finger touches on
// the surface are ignored so a resting palm does not draw.
export function trackStrokes(el, { toPoint, onStart, onPoints, onEnd }) {
  let penSeen = false;
  let active = null;

  el.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'pen') penSeen = true;
    if (e.pointerType === 'touch' && penSeen) return;
    if (active !== null) return;
    e.preventDefault();
    active = e.pointerId;
    el.setPointerCapture(e.pointerId);
    onStart(toPoint(e), e);
  });

  el.addEventListener('pointermove', (e) => {
    if (e.pointerId !== active) return;
    const coalesced = e.getCoalescedEvents?.() ?? [];
    onPoints((coalesced.length ? coalesced : [e]).map(toPoint));
  });

  const finish = (e) => {
    if (e.pointerId !== active) return;
    active = null;
    onEnd(e);
  };
  el.addEventListener('pointerup', finish);
  el.addEventListener('pointercancel', finish);
}

// Size a canvas's backing store to its CSS box at device resolution.
// Returns the CSS size; the context is scaled so drawing uses CSS pixels.
export function fitCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: r.width, h: r.height };
}

// Keep a canvas's backing store matched to its on-screen size. If the CSS
// size changes and the backing store does not, the drawing is stretched and
// ink lands away from the pen — e.g. when iPad Safari's toolbar shows or hides.
export function autoFit(canvas, onResize) {
  let last = '';
  const check = () => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const key = `${w}x${h}@${window.devicePixelRatio}`;
    if (!w || !h || key === last) return;
    last = key;
    onResize(fitCanvas(canvas));
  };
  new ResizeObserver(check).observe(canvas);
  window.addEventListener('resize', check);
  window.visualViewport?.addEventListener('resize', check);
  check();
}

// Add ?debug to a page URL: a red crosshair follows the raw pointer position
// the browser reports, with the numbers behind it. If the crosshair sits under
// the pencil tip but the ink does not, the bug is in our maths; if the
// crosshair is off too, the offset comes from the browser or device.
export function debugOverlay(canvas, extra = () => '') {
  if (!new URLSearchParams(location.search).has('debug')) return;
  const cross = document.createElement('div');
  cross.style.cssText =
    'position:fixed;left:0;top:0;width:31px;height:31px;margin:-15px 0 0 -15px;pointer-events:none;z-index:10;' +
    'background:linear-gradient(red,red) center/1px 100% no-repeat,linear-gradient(red,red) center/100% 1px no-repeat;';
  const info = document.createElement('pre');
  info.style.cssText =
    'position:fixed;left:8px;bottom:8px;margin:0;padding:8px 10px;font:11px/1.4 ui-monospace,Menlo,monospace;' +
    'background:rgba(255,255,255,0.9);border:1px solid #ccc;border-radius:8px;pointer-events:none;z-index:10;';
  document.body.append(cross, info);

  const show = (e) => {
    cross.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    const r = canvas.getBoundingClientRect();
    const vv = window.visualViewport;
    const co = e.getCoalescedEvents?.() ?? [];
    const lastCo = co[co.length - 1];
    const f = (v) => (Math.round(v * 10) / 10).toString();
    info.textContent = [
      `${e.type} ${e.pointerType}  pressure ${f(e.pressure)}  buttons ${e.buttons}`,
      `client     ${f(e.clientX)}, ${f(e.clientY)}`,
      `in canvas  ${f(e.clientX - r.left)}, ${f(e.clientY - r.top)}`,
      `canvas css ${f(r.left)}, ${f(r.top)}  ${f(r.width)} x ${f(r.height)}`,
      `backing    ${canvas.width} x ${canvas.height}  (dpr ${window.devicePixelRatio}, ratio ${f(canvas.width / r.width)} x ${f(canvas.height / r.height)})`,
      `window     ${innerWidth} x ${innerHeight}`,
      vv ? `viewport   ${f(vv.width)} x ${f(vv.height)}  offset ${f(vv.offsetLeft)}, ${f(vv.offsetTop)}  scale ${f(vv.scale)}` : 'viewport   n/a',
      lastCo ? `coalesced  ${co.length}  last Δ ${f(lastCo.clientX - e.clientX)}, ${f(lastCo.clientY - e.clientY)}` : 'coalesced  none',
      extra(),
    ]
      .filter(Boolean)
      .join('\n');
  };
  for (const type of ['pointerdown', 'pointermove', 'pointerup']) canvas.addEventListener(type, show);
}

// `map(x, y)` turns stored coordinates into canvas coordinates.
export function drawStroke(ctx, pts, map, baseWidth) {
  if (!pts.length) return;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (pts.length === 1) {
    const [X, Y] = map(pts[0][0], pts[0][1]);
    ctx.beginPath();
    ctx.arc(X, Y, Math.max(0.5, (baseWidth * widthFor(pts[0][3])) / 2), 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  let prev = map(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    const next = map(pts[i][0], pts[i][1]);
    ctx.lineWidth = baseWidth * widthFor((pts[i - 1][3] + pts[i][3]) / 2);
    ctx.beginPath();
    ctx.moveTo(prev[0], prev[1]);
    ctx.lineTo(next[0], next[1]);
    ctx.stroke();
    prev = next;
  }
}

function widthFor(pressure) {
  return 0.5 + Math.min(1, Math.max(0, pressure));
}

// Faint lines that split the writing surface into melody rows.
export function drawGuides(ctx, x, y, w, h, rows) {
  ctx.strokeStyle = GUIDE;
  ctx.lineWidth = 1;
  for (let i = 1; i < rows; i++) {
    const yy = Math.round(y + (h * i) / rows) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, yy);
    ctx.lineTo(x + w, yy);
    ctx.stroke();
  }
}

export const round = (v, digits) => Number(v.toFixed(digits));
