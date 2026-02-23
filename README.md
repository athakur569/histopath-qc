# histopath-qc

Quality control for H&E histopathology patches: blur detection, entropy thresholding, tissue scoring, and batch QC pipeline.

## Filters
- **Blur** — Laplacian variance; patches below threshold are out-of-focus
- **Entropy** — Shannon entropy on grayscale; low entropy = marker/pen artifact or near-blank
- **Tissue** — HSV saturation mask; fraction below threshold = mostly background
- **Batch QC** — run all filters in one pass, emit a per-patch pass/fail manifest

## Installation
```bash
pip install histopath-qc
```

## Quick start
```python
from histopath_qc import PatchQC

qc = PatchQC(blur_threshold=100, entropy_threshold=4.5, tissue_threshold=0.70)
result = qc.score(patch_array)   # numpy RGB (H,W,3) uint8
print(result)  # QCResult(blur_ok=True, entropy_ok=True, tissue_ok=True, passed=True)

# Batch
report = qc.run_batch("patches/", output="qc_report.csv")
print(f"Passed: {report.n_passed}/{report.n_total}")
```

## Thresholds (defaults)
| Filter | Default | Typical range |
|---|---|---|
| Laplacian variance (blur) | 100 | 50–200 |
| Shannon entropy | 4.5 | 3.5–6.0 |
| Tissue fraction | 0.70 | 0.50–0.85 |

## License
MIT
