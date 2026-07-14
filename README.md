# Image Quality Assessment Toolkit

Python utilities for evaluating image quality using **PSNR** and **SSIM**, with
automated batch comparison of compressed images against their reference
originals, and generated performance statistics / quality reports.

Built with OpenCV + NumPy only (no scikit-image dependency) — SSIM is
implemented from scratch using a Gaussian-weighted sliding window, matching
the standard Wang et al. (2004) formulation.

## Files

| File | Purpose |
|---|---|
| `metrics.py` | Core metric functions: `compute_psnr`, `compute_ssim`, `compute_mse`, `evaluate_pair` |
| `compare_single.py` | CLI to compare one reference/distorted image pair |
| `batch_compare.py` | CLI to batch-compare a folder of images and generate a CSV + text report + chart |
| `test_data/` | Synthetic sample images (generated for demo/testing) |

## Setup

```bash
pip3 install opencv-python numpy matplotlib
```

(matplotlib is optional — only needed for the chart in batch reports)

## Usage

### Compare a single pair of images

```bash
python3 compare_single.py --reference original.png --distorted compressed.jpg
```

Output:
```
MSE:  2.7697
PSNR: 43.71 dB
SSIM: 0.9967
```

### Batch-compare a folder

Reference and distorted images must share the same filename across the two
directories (e.g. `image1.png` in both `reference/` and `distorted/` — the
distorted one being the compressed/processed version).

```bash
python3 batch_compare.py \
    --reference-dir path/to/reference \
    --distorted-dir path/to/distorted \
    --output-dir path/to/report
```

This generates in `--output-dir`:
- `iqa_results.csv` — per-image MSE, PSNR, SSIM, file sizes, compression ratio
- `iqa_summary.txt` — mean/min/max/std statistics and a ranked per-image table
- `iqa_chart.png` — bar charts of PSNR and SSIM per image (skip with `--no-chart`)

### Use as a library

```python
import cv2
from metrics import compute_psnr, compute_ssim, evaluate_pair

ref = cv2.imread("original.png")
dist = cv2.imread("compressed.png")

psnr = compute_psnr(ref, dist)
ssim = compute_ssim(ref, dist)

# or get everything at once (also handles size mismatch by resizing)
metrics = evaluate_pair(ref, dist)   # {'mse':..., 'psnr':..., 'ssim':...}
```

## Metric notes

- **PSNR** (Peak Signal-to-Noise Ratio, dB): higher is better; `inf` means the
  images are pixel-identical. Sensitive to raw pixel-level error, not human
  perception.
- **SSIM** (Structural Similarity Index, range -1 to 1, typically 0 to 1 for
  natural images): higher is better; 1.0 = identical. Correlates better with
  perceived visual quality since it accounts for luminance, contrast, and
  structure.

## Demo

The included `test_data/` folder has 4 synthetic reference images and their
JPEG-compressed counterparts at decreasing quality (95, 70, 40, 10), useful
for a quick sanity check:

```bash
python3 batch_compare.py --reference-dir test_data/reference --distorted-dir test_data/distorted --output-dir test_output
```

Expect PSNR/SSIM to decrease monotonically as JPEG quality drops.
