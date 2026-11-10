"""Turn QMNIST's training set into files the browser can load quickly.

qmnist_images.png   784 x N grayscale sprite, one digit per row (like the TF.js MNIST sprite)
qmnist_labels.bin   N bytes of digit, then N little-endian uint16 compact writer indices
qmnist_writers.json [{ "id": NIST writer id, "series": HSF series, "count": digits }] per compact index
"""
import array, gzip, json, statistics, struct, sys
from collections import Counter, defaultdict
from PIL import Image

raw, out = sys.argv[1], sys.argv[2]

with gzip.open(f"{raw}/qmnist-train-images-idx3-ubyte.gz") as f:
    magic, n, h, w = struct.unpack(">IIII", f.read(16))
    pixels = f.read()
with gzip.open(f"{raw}/qmnist-train-labels-idx2-int.gz") as f:
    magic, rows, cols = struct.unpack(">III", f.read(12))
    labels = array.array("i", f.read())
    if sys.byteorder == "little":
        labels.byteswap()
assert rows == n and h == w == 28 and len(pixels) == n * 784

digits = bytes(labels[r * cols] for r in range(n))
series = [labels[r * cols + 1] for r in range(n)]
writer_ids = [labels[r * cols + 2] for r in range(n)]

ids = sorted(set(writer_ids))
compact = {wid: i for i, wid in enumerate(ids)}
writers = array.array("H", (compact[x] for x in writer_ids))
if sys.byteorder != "little":
    writers.byteswap()

count = Counter(writer_ids)
series_of = defaultdict(set)
for wid, s in zip(writer_ids, series):
    series_of[wid].add(s)
info = [{"id": wid, "series": min(series_of[wid]), "count": count[wid]} for wid in ids]

Image.frombytes("L", (784, n), pixels).save(f"{out}/qmnist_images.png", optimize=True)
with open(f"{out}/qmnist_labels.bin", "wb") as f:
    f.write(digits)
    f.write(writers.tobytes())
with open(f"{out}/qmnist_writers.json", "w") as f:
    json.dump(info, f)

counts = list(count.values())
students = sum(1 for x in info if x["series"] == 4)
mixed = sum(1 for s in series_of.values() if len(s) > 1)
per_digit = [len({w for w, d in zip(writer_ids, digits) if d == k}) for k in range(10)]
print(f"{n} digits by {len(ids)} writers ({students} students, {mixed} with mixed series)")
print(f"digits per writer: {min(counts)}-{max(counts)}, median {statistics.median(counts)}")
print("series:", sorted(set(series)), "writers per digit:", per_digit)
print("sevens:", digits.count(7))
