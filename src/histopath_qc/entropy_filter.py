"""Entropy-based filter to reject near-blank or artifact patches."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy


class EntropyFilter:
    """Reject patches with low Shannon entropy on the grayscale histogram.

    Low entropy ≈ near-blank, pen marks, or single-color artifacts.
    High entropy ≈ rich tissue texture.

    Args:
        threshold:  Entropy below this value is rejected. Default 4.5.
                    Range: blank/background ≈ 0–3, tissue ≈ 5–7.
        n_bins:     Histogram bins for entropy estimation. Default 256.
    """

    def __init__(self, threshold: float = 4.5, n_bins: int = 256) -> None:
        self.threshold = threshold
        self.n_bins = n_bins

    def score(self, image: np.ndarray) -> float:
        """Return Shannon entropy of the grayscale histogram (nats)."""
        gray = self._to_gray(image)
        hist, _ = np.histogram(gray, bins=self.n_bins, range=(0, 256))
        hist = hist + 1e-10  # avoid log(0)
        hist = hist / hist.sum()
        return float(scipy_entropy(hist))

    def passes(self, image: np.ndarray) -> bool:
        """Return True if patch entropy is above threshold."""
        return self.score(image) >= self.threshold

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
