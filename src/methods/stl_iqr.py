"""
STL + IQR statistical anomaly detector.

Seasonal-Trend decomposition using LOESS (STL), then flag points where
the residual component exceeds IQR-based thresholds.
"""

from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from .base import BaseDetector


class STLIQRDetector(BaseDetector):
    """Detect anomalies via STL decomposition + IQR on resid."""

    def __init__(
        self,
        period: Optional[int] = None,
        iqr_multiplier: float = 1.5,
        robust: bool = True,
    ):
        self.period = period
        self.iqr_multiplier = iqr_multiplier
        self.robust = robust
        self._result = None
        self._residual = None

    def fit(self, series: pd.Series) -> None:
        period = self.period or self._detect_period(series)
        model = STL(series.values, period=period, robust=self.robust)
        self._result = model.fit()
        self._residual = pd.Series(
            self._result.resid, index=series.index, name="residual"
        )

    def score(self, series: pd.Series) -> pd.Series:
        if self._result is None:
            raise ValueError("Call fit() before score()")
        if len(series) != len(self._residual):
            self.fit(series)

        resid = self._residual.reindex(series.index)
        med = resid.median()
        q1 = resid.quantile(0.25)
        q3 = resid.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            iqr = resid.std() if resid.std() > 0 else 1.0
        scores = (resid - med).abs() / iqr
        return scores.fillna(0)

    def _auto_threshold(self, scores: pd.Series) -> float:
        return self.iqr_multiplier

    @staticmethod
    def _detect_period(series: pd.Series) -> int:
        if len(series) >= 365 * 2:
            return 365
        elif len(series) >= 90:
            return 7
        else:
            return max(2, len(series) // 10)
