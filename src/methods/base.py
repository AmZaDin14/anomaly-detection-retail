"""
Base interface and utilities for anomaly detection methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional


class BaseDetector(ABC):
    """Abstract base for all anomaly detectors."""

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Fit detector on clean (or unlabeled) training data."""
        ...

    @abstractmethod
    def score(self, series: pd.Series) -> pd.Series:
        """
        Return anomaly scores (higher = more anomalous).
        Same index as input series.
        """
        ...

    def predict(self, series: pd.Series, threshold: Optional[float] = None) -> pd.Series:
        """
        Return binary labels (True = anomaly).
        If threshold is None, use method-specific automatic thresholding.
        """
        scores = self.score(series)
        if threshold is None:
            threshold = self._auto_threshold(scores)
        return scores > threshold

    def _auto_threshold(self, scores: pd.Series) -> float:
        """Default: 95th percentile of scores."""
        return scores.quantile(0.95)


def sliding_window_features(
    series: pd.Series,
    window: int = 14,
    step: int = 1,
) -> tuple[pd.DataFrame, pd.Index]:
    """
    Convert a time series into sliding-window feature matrix.

    Each row = statistics over a window:
        - mean, std, min, max, median
        - last value, slope (linear regression)
        - relative change (last / mean)
        - day-of-week (categorical as sin/cos)

    Returns
    -------
    X : pd.DataFrame (n_windows, n_features)
    indices : pd.Index (n_windows,) — the last timestamp of each window
    """
    n = len(series)
    rows, indices = [], []

    for i in range(window, n, step):
        chunk = series.iloc[i - window : i]
        last_val = chunk.iloc[-1]
        mean_val = chunk.mean()
        std_val = chunk.std()

        # Slope: linear regression over chunk
        x = np.arange(len(chunk))
        y = chunk.values
        if std_val > 0 and len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        # Day-of-week features
        dow = chunk.index[-1].dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        rows.append({
            "mean": mean_val,
            "std": std_val,
            "min": chunk.min(),
            "max": chunk.max(),
            "median": chunk.median(),
            "last": last_val,
            "slope": slope,
            "rel_change": (last_val - mean_val) / (mean_val + 1e-8),
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
        })
        indices.append(chunk.index[-1])

    X = pd.DataFrame(rows, index=indices)
    X = X.fillna(0)
    return X, pd.Index(indices)
