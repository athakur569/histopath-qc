"""Batch QC pipeline: run all filters over a directory of patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .blur_detector import BlurDetector
from .entropy_filter import EntropyFilter
from .tissue_detector import TissueDetector


@dataclass
class QCResult:
    patch_id: str
    blur_score: float
    entropy_score: float
    tissue_fraction: float
    blur_ok: bool
    entropy_ok: bool
    tissue_ok: bool
    passed: bool


@dataclass
class BatchQCReport:
    records: list[QCResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.records)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_passed

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_total, 1)

    def to_dataframe(self) -> pd.DataFrame:
        from dataclasses import asdict
        return pd.DataFrame([asdict(r) for r in self.records])

    def save(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def filter_summary(self) -> dict:
        df = self.to_dataframe()
        return {
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "failed_blur": int((~df["blur_ok"]).sum()),
            "failed_entropy": int((~df["entropy_ok"]).sum()),
            "failed_tissue": int((~df["tissue_ok"]).sum()),
            "pass_rate": round(self.pass_rate, 4),
        }


class PatchQC:
    """Compose blur, entropy, and tissue filters into a single QC pipeline.

    Args:
        blur_threshold:    Laplacian variance minimum (default 100).
        entropy_threshold: Shannon entropy minimum (default 4.5).
        tissue_threshold:  Tissue fraction minimum (default 0.70).
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        entropy_threshold: float = 4.5,
        tissue_threshold: float = 0.70,
    ) -> None:
        self.blur = BlurDetector(threshold=blur_threshold)
        self.entropy = EntropyFilter(threshold=entropy_threshold)
        self.tissue = TissueDetector(threshold=tissue_threshold)

    def score(self, image: np.ndarray, patch_id: str = "") -> QCResult:
        """Score a single patch and return a QCResult."""
        blur_s = self.blur.score(image)
        ent_s = self.entropy.score(image)
        tissue_s = self.tissue.tissue_fraction(image)

        blur_ok = blur_s >= self.blur.threshold
        entropy_ok = ent_s >= self.entropy.threshold
        tissue_ok = tissue_s >= self.tissue.threshold

        return QCResult(
            patch_id=patch_id,
            blur_score=round(blur_s, 4),
            entropy_score=round(ent_s, 4),
            tissue_fraction=round(tissue_s, 4),
            blur_ok=blur_ok,
            entropy_ok=entropy_ok,
            tissue_ok=tissue_ok,
            passed=blur_ok and entropy_ok and tissue_ok,
        )

    def run_batch(
        self,
        patch_dir: str | Path,
        output: str | Path | None = None,
        pattern: str = "*.png",
    ) -> BatchQCReport:
        """Run QC on all patches in a directory.

        Args:
            patch_dir: Directory containing patch image files.
            output:    Optional CSV path to save the report.
            pattern:   Glob pattern for image files.

        Returns:
            BatchQCReport with per-patch results and summary statistics.
        """
        patch_dir = Path(patch_dir)
        paths = sorted(patch_dir.glob(pattern))
        report = BatchQCReport()

        for path in tqdm(paths, desc="QC", unit="patch"):
            img = np.array(Image.open(path).convert("RGB"))
            result = self.score(img, patch_id=path.stem)
            report.records.append(result)

        if output:
            report.save(output)

        summary = report.filter_summary()
        print(
            f"QC complete: {summary['n_passed']}/{summary['n_total']} passed "
            f"({summary['pass_rate']*100:.1f}%) — "
            f"blur_fails={summary['failed_blur']}, "
            f"entropy_fails={summary['failed_entropy']}, "
            f"tissue_fails={summary['failed_tissue']}"
        )
        return report
