"""
Local Outlier Factor anomaly detector.

Uses sliding-window feature extraction + sklearn LOF.
LOF is trained on the given series and scores each window's anomaly level.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as SKLOF

from .base import BaseDetector, sliding_window_features


class LOFDetector(BaseDetector):
    """Local Outlier Factor on sliding-window retail features."""

    def __init__(
        self,
        window: int = 14,
        n_neighbors: int = 20,
        contamination: float = 0.05,
    ):
        self.window = window
        self.n_neighbors = min(n_neighbors, 30)
        self.contamination = contamination
        self._model = None

    def fit(self, series: pd.Series) -> None:
        X, _ = sliding_window_features(series, window=self.window)
        self._model = SKLOF(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1,
        )
        self._model.fit(X.values)
        self._series_mean = series.mean()
        self._series_std = series.std()

    def score(self, series: pd.Series) -> pd.Series:
        if self._model is None:
            raise ValueError("Call fit() before score()")

        X, indices = sliding_window_features(series, window=self.window)
        # Negative LOF = outlier (sklearn convention); we negate so higher = anomalous
        raw_scores = -self._model.decision_function(X.values)

        scores = pd.Series(0.0, index=series.index, dtype=float)
        for idx, score in zip(indices, raw_scores):
            if idx in scores.index:
                scores[idx] = score

        return scores

    def _auto_threshold(self, scores: pd.Series) -> float:
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            return scores.quantile(0.95)
        return non_zero.quantile(1.0 - self.contamination)
