// Turns one row of the score into beats for looping, following numbered
// notation: each digit is one beat; "-" holds the note before it one beat
// longer; "0" is a one-beat rest; "/" takes no time (no meaning yet).
//
// Each step is one beat: { kind: 'note' | 'hold' | 'rest', x, id, ... }.
// A note step also has midi, velocity and `beats` (1 + the holds after it).
// `x` is the centre of the symbol, used to move the playhead.

import { noteFor } from './sound.js';

export function compileRow(notes, velocityOf = () => 0.7) {
  const steps = [];
  for (const n of notes) {
    if (n.symbol === '/') continue;
    const x = (n.box.x0 + n.box.x1) / 2;
    if (n.symbol === '-') {
      steps.push({ kind: 'hold', x, id: n.id });
      const owner = soundingNote(steps);
      if (owner) owner.beats++;
      continue;
    }
    const midi = noteFor(n.symbol, n.octave);
    if (midi === null) {
      steps.push({ kind: 'rest', x, id: n.id });
      continue;
    }
    steps.push({ kind: 'note', x, id: n.id, midi, beats: 1, velocity: velocityOf(n.box) });
  }
  return steps;
}

// The note a "-" at the end of `steps` extends: walk back over holds; a rest
// stops the search, so "0 -" is simply a longer rest.
function soundingNote(steps) {
  for (let i = steps.length - 2; i >= 0; i--) {
    if (steps[i].kind === 'note') return steps[i];
    if (steps[i].kind === 'rest') return null;
  }
  return null;
}
