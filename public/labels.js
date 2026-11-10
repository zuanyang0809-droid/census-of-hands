// The symbols the model will learn. Shared by the server (to validate and
// file samples) and the pages (to prompt and review).
// `dir` is the folder name under data/samples/ — symbols are not safe file names.

export const LABELS = [
  { label: '0', dir: '0', hint: 'rest' },
  { label: '1', dir: '1', hint: 'do' },
  { label: '2', dir: '2', hint: 're' },
  { label: '3', dir: '3', hint: 'mi' },
  { label: '4', dir: '4', hint: 'fa' },
  { label: '5', dir: '5', hint: 'sol' },
  { label: '6', dir: '6', hint: 'la' },
  { label: '7', dir: '7', hint: 'si' },
  { label: '-', dir: 'dash', hint: 'hold — a long dash' },
  { label: '/', dir: 'slash', hint: 'slash' },
  { label: 'other', dir: 'other', hint: 'anything else: scribble, circle, letter' },
];

// Digits are pre-trained on MNIST, so a few of your own samples per symbol is enough.
export const TARGET_PER_LABEL = 10;
