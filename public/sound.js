// Sound: a sampled grand piano on the Web Audio API, and the mapping from
// written symbols to pitches. Numbered notation: 1–7 are do–si of C major,
// a dot above raises an octave, a dot below lowers one, 0 is a rest.
//
// Piano: Salamander Grand Piano by Alexander Holm (CC-BY 3.0), 13 notes from
// C3 to C6 in public/samples/piano/. Notes in between are the nearest
// recording re-pitched by at most a tone and a half. If the recordings
// cannot be loaded, a simple synthesised tone is used instead.

export const SCALE = { 1: 60, 2: 62, 3: 64, 4: 65, 5: 67, 6: 69, 7: 71 }; // MIDI, C4 = 60
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const PIANO_DIR = 'samples/piano';
const PIANO_NOTES = ['C3', 'Ds3', 'Fs3', 'A3', 'C4', 'Ds4', 'Fs4', 'A4', 'C5', 'Ds5', 'Fs5', 'A5', 'C6'];
const RELEASE_S = 0.35; // how long a note takes to fade once its time is up

// MIDI note for a symbol, or null for anything that does not sound.
export function noteFor(symbol, octave = 0) {
  const base = SCALE[symbol];
  return base === undefined ? null : base + 12 * octave;
}

export function noteName(midi) {
  return `${NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

// "Ds4" -> 63
function midiOf(name) {
  const m = /^([A-G])(s?)(\d)$/.exec(name);
  return NAMES.indexOf(m[1]) + (m[2] ? 1 : 0) + 12 * (Number(m[3]) + 1);
}

export class Synth {
  constructor() {
    this.ctx = null;
    this.out = null;
    this.muted = false;
    this.samples = []; // { midi, buffer }, once loaded
    this.loading = null;
  }

  // Browsers only allow sound after a click on the page, so call this from one.
  async unlock() {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      const comp = this.ctx.createDynamicsCompressor();
      comp.threshold.value = -14;
      comp.ratio.value = 3;
      this.out = this.ctx.createGain();
      this.out.gain.value = 0.9;
      this.out.connect(comp).connect(this.ctx.destination);
      this.loading = this.loadPiano();
    }
    await this.ctx.resume();
    await this.loading;
  }

  async loadPiano() {
    try {
      this.samples = await Promise.all(
        PIANO_NOTES.map(async (name) => {
          const res = await fetch(`${PIANO_DIR}/${name}.mp3`);
          if (!res.ok) throw new Error(`${name}.mp3: ${res.status}`);
          return { midi: midiOf(name), buffer: await this.ctx.decodeAudioData(await res.arrayBuffer()) };
        }),
      );
    } catch (err) {
      console.warn('Piano samples not loaded, using the synthesised tone instead.', err);
      this.samples = [];
    }
  }

  get ready() {
    return this.ctx?.state === 'running';
  }

  get instrument() {
    return this.samples.length ? 'piano' : 'synth';
  }

  // Play `midi` `when` seconds from now. `duration` is how long it should
  // sound before fading; `velocity` 0..1 is how hard it is struck.
  play(midi, { velocity = 0.7, duration = 1.6, when = 0 } = {}) {
    if (!this.ready || this.muted) return false;
    const t = this.ctx.currentTime + when;
    const v = Math.max(0.05, Math.min(1, velocity));
    if (this.samples.length) this.playPiano(midi, v, duration, t);
    else this.playSynth(midi, v, duration, t);
    return true;
  }

  playPiano(midi, v, duration, t) {
    const ctx = this.ctx;
    let nearest = this.samples[0];
    for (const s of this.samples) if (Math.abs(s.midi - midi) < Math.abs(nearest.midi - midi)) nearest = s;

    const src = ctx.createBufferSource();
    src.buffer = nearest.buffer;
    src.playbackRate.value = 2 ** ((midi - nearest.midi) / 12);

    // Softer notes are quieter and a little darker, like a real piano.
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 2500 + 9000 * v;

    const env = ctx.createGain();
    const level = 0.25 + 0.75 * v ** 1.4;
    const end = t + duration;
    env.gain.setValueAtTime(level, t);
    env.gain.setValueAtTime(level, end);
    env.gain.exponentialRampToValueAtTime(0.0001, end + RELEASE_S);

    src.connect(filter).connect(env).connect(this.out);
    src.start(t);
    src.stop(end + RELEASE_S + 0.05);
  }

  // Fallback: triangle plus a quiet octave sine, through a low-pass filter.
  playSynth(midi, v, duration, t) {
    const ctx = this.ctx;
    const freq = 440 * 2 ** ((midi - 69) / 12);
    const env = ctx.createGain();
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(v, t + 0.012);
    env.gain.exponentialRampToValueAtTime(v * 0.45, t + 0.18);
    env.gain.exponentialRampToValueAtTime(0.0001, t + duration);
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 1800 + 2400 * v;
    filter.connect(env).connect(this.out);
    for (const { type, ratio, gain } of [
      { type: 'triangle', ratio: 1, gain: 0.8 },
      { type: 'sine', ratio: 2, gain: 0.18 },
    ]) {
      const osc = ctx.createOscillator();
      osc.type = type;
      osc.frequency.value = freq * ratio;
      const g = ctx.createGain();
      g.gain.value = gain;
      osc.connect(g).connect(filter);
      osc.start(t);
      osc.stop(t + duration + 0.05);
    }
  }
}
