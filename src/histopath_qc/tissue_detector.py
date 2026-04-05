"""HSV-based tissue detector for patch-level scoring."""

from __future__ import annotations

import cv2
import numpy as np


class TissueDetector:
    """Score how much of a patch is tissue (vs. background/glass).

    Uses HSV saturation channel: H&E-stained tissue is more saturated
    than glass background (which appears near-white/gray).

    Args:
        threshold:        Tissue fraction below this value is rejected.
        sat_threshold:    HSV saturation (0–255) above which a pixel is tissue.
        kernel_size:      Morphological kernel for noise removal.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        sat_threshold: int = 20,
        kernel_size: int = 7,
    ) -> None:
        self.threshold = threshold
        self.sat_threshold = sat_threshold
        self.kernel_size = kernel_size

    def tissue_fraction(self, image: np.ndarray) -> float:
        """Return fraction of pixels classified as tissue (0–1)."""
        mask = self.get_mask(image)
        return float(mask.sum()) / float(mask.size * 255)

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Return binary tissue mask (255=tissue, 0=background)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        _, binary = cv2.threshold(sat, self.sat_threshold, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        return cv2.morphologyEx(
            cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k),
            cv2.MORPH_OPEN, k,
        )

    def passes(self, image: np.ndarray) -> bool:
        return self.tissue_fraction(image) >= self.threshold
