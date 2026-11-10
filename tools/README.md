# tools

`convert_qmnist.py` rebuilds `data/qmnist/` from the original QMNIST files.
It needs Python 3 and Pillow.

```bash
mkdir -p data/qmnist-raw data/qmnist
curl -L -o data/qmnist-raw/qmnist-train-images-idx3-ubyte.gz https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz
curl -L -o data/qmnist-raw/qmnist-train-labels-idx2-int.gz https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz
python3 tools/convert_qmnist.py data/qmnist-raw data/qmnist
```
