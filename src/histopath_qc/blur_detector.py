"""Blur detection via Laplacian variance."""

from __future__ import annotations

import cv2
import numpy as np


class BlurDetector:
    """Detect blurry patches using the variance of the Laplacian operator.

    A sharp patch has high edge energy (high Laplacian variance).
    A blurry patch has diffuse edges (low Laplacian variance).

    Args:
        threshold: Laplacian variance below this value is considered blurry.
                   Typical range 50–200; default 100 works well for 256×256 patches.
    """

    def __init__(self, threshold: float = 100.0) -> None:
        self.threshold = threshold

    def score(self, image: np.ndarray) -> float:
        """Return Laplacian variance of a patch (higher = sharper)."""
        gray = self._to_gray(image)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def is_sharp(self, image: np.ndarray) -> bool:
        """Return True if the patch passes the sharpness threshold."""
        return self.score(image) >= self.threshold

    def batch_score(self, images: list[np.ndarray]) -> np.ndarray:
        """Score a list of patches. Returns (n,) float array."""
        return np.array([self.score(img) for img in images])

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.astype(np.uint8)
