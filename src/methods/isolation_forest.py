"""
Isolation Forest anomaly detector.

Uses sliding-window feature extraction + sklearn Isolation Forest.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest as SKIsolationForest

from .base import BaseDetector, sliding_window_features


class IsolationForestDetector(BaseDetector):
    """Isolation Forest on sliding-window retail features."""

    def __init__(
        self,
        window: int = 14,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.window = window
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._trained_index = None

    def fit(self, series: pd.Series) -> None:
        X, _ = sliding_window_features(series, window=self.window)
        self._model = SKIsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X.values)
        # Store full series mean/std for out-of-window scoring
        self._series_mean = series.mean()
        self._series_std = series.std()

    def score(self, series: pd.Series) -> pd.Series:
        if self._model is None:
            raise ValueError("Call fit() before score()")

        X, indices = sliding_window_features(series, window=self.window)
        # decision_function: lower = more anomalous; we negate so higher = anomalous
        raw_scores = -self._model.decision_function(X.values)

        # Align scores to full series index (fill NaN for first `window` points)
        scores = pd.Series(0.0, index=series.index, dtype=float)
        for idx, score in zip(indices, raw_scores):
            if idx in scores.index:
                scores[idx] = score

        return scores

    def _auto_threshold(self, scores: pd.Series) -> float:
        """Use the contamination-based threshold from training."""
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            return scores.quantile(0.95)
        return non_zero.quantile(1.0 - self.contamination)
