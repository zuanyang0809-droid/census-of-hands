# Census of Hands

Every machine that reads handwritten digits learned from MNIST: 60,000 digits
written around 1990 by 539 people, 289 adult employees and 250 American
high-school students, about 116 digits each.

Write a number. A model I trained reads it, turns your handwriting into a
fingerprint, and the digits in the archive written most like yours gather
around it. The closest sit right beside you. Further out, the digits are less
like yours, until everyone who ever wrote that number for the archive has a
place. Click any digit to see everything else its writer wrote.

The strangers are still. Only your digit moves: every few seconds it writes
itself again, stroke by stroke, the way you wrote it.

Made for the recognition assignment in Drawing++ (Zach Lieberman, Fall 2026):
*create a tool that detects and responds to specific drawings.*

**Try it:** https://zuanyang0809-droid.github.io/census-of-hands/ (mouse and trackpad)

## Run it

Needs [Node.js](https://nodejs.org) 18 or newer.

```bash
npm install
npm start
```

Then open **http://localhost:3001/hands.html** and draw with the mouse.

| Gesture | What it does |
| --- | --- |
| Drag | Write a digit (a new one replaces the last) |
| Click a digit | Light up everything its writer wrote; click again to let go |
| Two-finger scroll | Move around the archive |
| Pinch | Zoom in and out |
| Space | Fly back to your digit |
| Esc | Clear |

### With an iPad

The iPad is only the pen; the Mac runs everything. With both on the same
network, open `http://localhost:3001/census.html` on the Mac and the
`iPad pen` address printed by `npm start` on the iPad. There, the 12 people
who write your digit most like you (12 different people) gather around it.

## How it works

1. **Your digit becomes an MNIST digit.** Strokes are drawn into a 28 × 28
   image the way MNIST's were made: the digit scaled to fit a 20-pixel box,
   then shifted so its centre of mass sits in the middle
   (`public/preprocess.js`).
2. **One small network, two questions.** A convolutional network trained from
   scratch in the browser with TensorFlow.js (`public/model.js`,
   `public/train.js`) answers:
   - *which digit is it?* (0–9, or "not a digit")
   - *who wrote it?* (one of the archive's 539 writers, or you)

   To read digits well, a network learns to ignore how each person writes. To
   tell writers apart, it has to keep exactly that. So the 64 numbers just
   before the *who wrote it* answer are the handwriting fingerprint.
3. **The archive is indexed.** After training, all 60,000 archive digits are
   fingerprinted and saved (`data/model/mnist_fingerprints.bin`). Your digit is
   compared with every archive digit of the same number, and the grid is laid
   out by rank: the n-th closest cell holds the n-th closest digit
   (`public/hands.js`).

### How well it works

Trained on the first 54,000 archive digits plus my own handwriting samples
(164 of my 180; the slashes are left out because they look like a slanted 1),
with the last 6,000 archive digits and 18 of my samples held out:

| | |
| --- | --- |
| Reads the archive's held-out digits | 99.2% |
| Reads my own held-out samples | 18 / 18 |
| Closest match to a held-out digit is by the same person, using the style fingerprint | 9.7% |
| … using the digit layer (the earlier fingerprint) | 5.1% |
| … by chance | 0.2% |

Telling who wrote a single digit is hard. The fingerprint finds the same
person about 50 times more often than chance, but most of the time the digit
closest to yours is by someone else. Read the similarity as *how alike the
machine thinks two hands are*, not as a measure of handwriting itself.

## Train it again

Open `http://localhost:3001/collect.html` on the iPad to write more samples,
`/review.html` on the Mac to check them, then `/train.html` and press
**Train** (about 4 minutes, keep the tab in front). Training rebuilds the
model and the fingerprint index.

## Files

```
server.js            serves the pages and data, relays the iPad pen, stores samples
public/
  hands.html/.js     the piece, drawn with the mouse
  census.html/.js    the iPad version, shown on the Mac
  pen.html/.js       the iPad writing surface
  collect, review    writing and checking training samples
  train.html/.js     training, evaluation, indexing, and a try-it pad
  model.js           the network
  archive.js         loading the archive, nearest digits, writer names
  preprocess.js      strokes to a 28 × 28 MNIST-style image
  display, sequence, sound
                     the earlier music version (digits as notation, piano)
  vendor/tf.min.js   TensorFlow.js 4.22.0
data/
  qmnist/            the archive: digits, labels, writers
  model/             the trained model and its fingerprint index
  samples/           my 180 handwriting samples, one JSON file each
tools/               rebuilding data/qmnist from the original QMNIST files
```

## Credits

- **QMNIST**: Chhavi Yadav and Léon Bottou, *Cold Case: The Lost MNIST
  Digits* (2019). [github.com/facebookresearch/qmnist](https://github.com/facebookresearch/qmnist),
  BSD licence. It rebuilt MNIST's training set from the original NIST forms
  and recovered who wrote each digit.
- **MNIST**: Yann LeCun, Corinna Cortes and Christopher J. C. Burges, from NIST
  Special Databases 1 and 3.
- **TensorFlow.js** 4.22.0, Apache 2.0.
- **Salamander Grand Piano V3** by Alexander Holm, CC-BY 3.0 (the music
  version only).
