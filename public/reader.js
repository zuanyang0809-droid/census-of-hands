// Reads the page: groups pen strokes into symbols, recognises each one, and
// turns the page into a score — one list of notes per row, left to right.
//
// Grouping rules (sizes are fractions of a row's height):
//  - A stroke smaller than DOT_MAX is a dot, not a symbol. A dot above a
//    symbol raises it an octave, below lowers it (numbered notation).
//  - A stroke that overlaps a symbol in the same row horizontally, or comes
//    within JOIN_GAP of it, belongs to that symbol (the two strokes of a 4).
//    Otherwise it starts a new symbol.
// Position decides, not timing, so you can go back and add to a symbol.
//
// Pure logic, no DOM: points are [x, y, ...] in page units where the page
// is 1 wide and `aspect` tall.

export const DOT_MAX = 0.08;
export const JOIN_GAP = 0.08;
export const DOT_REACH = 0.6; // how far sideways a dot can be from its symbol

function bounds(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1 };
}

const union = (a, b) => ({ x0: Math.min(a.x0, b.x0), y0: Math.min(a.y0, b.y0), x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1) });
const centerX = (b) => (b.x0 + b.x1) / 2;
const centerY = (b) => (b.y0 + b.y1) / 2;

export class Reader {
  // recognize(strokes) -> { label, confidence } or null (no model yet)
  constructor(recognize) {
    this.recognize = recognize;
    this.page = { aspect: 0.75, rows: 4 };
    this.reset();
  }

  reset() {
    this.glyphs = []; // { id, row, strokes, box, result }
    this.dots = []; // { row, box }
    this.nextId = 1;
  }

  setPage({ aspect, rows }) {
    this.page = { aspect, rows };
  }

  get rowHeight() {
    return this.page.aspect / this.page.rows;
  }

  rowOf(box) {
    const r = Math.floor(centerY(box) / this.rowHeight);
    return Math.max(0, Math.min(this.page.rows - 1, r));
  }

  // Returns the symbol the stroke went into, or null for a dot.
  addStroke(pts) {
    if (!pts?.length) return null;
    const box = bounds(pts);
    const row = this.rowOf(box);
    const h = this.rowHeight;
    if (Math.max(box.x1 - box.x0, box.y1 - box.y0) < DOT_MAX * h) {
      this.dots.push({ row, box });
      return null;
    }
    let target = null;
    let bestGap = JOIN_GAP * h;
    for (const g of this.glyphs) {
      if (g.row !== row) continue;
      const gap = Math.max(g.box.x0, box.x0) - Math.min(g.box.x1, box.x1); // < 0 means overlap
      if (gap < bestGap) {
        bestGap = gap;
        target = g;
      }
    }
    if (target) {
      target.strokes.push(pts);
      target.box = union(target.box, box);
    } else {
      target = { id: this.nextId++, row, strokes: [pts], box, result: null };
      this.glyphs.push(target);
    }
    target.result = this.recognize(target.strokes);
    return target;
  }

  // Run recognition again on everything (e.g. once the model has loaded).
  rerecognize() {
    for (const g of this.glyphs) g.result = this.recognize(g.strokes);
  }

  // [{ row, notes: [{ id, symbol, confidence, octave, box }] }], each row left to right.
  // Symbols the model calls "other" are left out of the score.
  score() {
    const rows = Array.from({ length: this.page.rows }, (_, row) => ({ row, notes: [] }));
    for (const g of this.glyphs) {
      if (!g.result || g.result.label === 'other') continue;
      rows[g.row].notes.push({ id: g.id, symbol: g.result.label, confidence: g.result.confidence, octave: 0, box: g.box });
    }
    for (const d of this.dots) {
      let nearest = null;
      let best = DOT_REACH * this.rowHeight;
      for (const n of rows[d.row].notes) {
        const dist = Math.abs(centerX(n.box) - centerX(d.box));
        if (dist < best) {
          best = dist;
          nearest = n;
        }
      }
      if (nearest) nearest.octave += centerY(d.box) < centerY(nearest.box) ? 1 : -1;
    }
    for (const r of rows) r.notes.sort((a, b) => a.box.x0 - b.box.x0);
    return rows;
  }
}
