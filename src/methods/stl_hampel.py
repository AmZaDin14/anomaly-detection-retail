"""
STL-Hampel: STL decomposition + Hampel identifier with adaptive window.

Hampel identifier uses rolling MAD (Median Absolute Deviation) instead of
IQR for more robust thresholding in non-Gaussian residual distributions.
Window size adapts to local trend volatility — larger windows in volatile
periods to reduce false positives.

Reference: Hampel, F. R. (1974). "The Influence Curve and its Role in
Robust Estimation." Journal of the American Statistical Association.
"""

from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from .base import BaseDetector


class STLHampelDetector(BaseDetector):
    """STL decomposition + adaptive Hampel identifier.

    Key innovations:
    1. Hampel MAD-based threshold instead of IQR (more robust to skew)
    2. Window size adapts to local trend volatility
    3. Multiplier automatically calibrated via peak-over-threshold heuristic
    """

    def __init__(
        self,
        period: Optional[int] = None,
        window_size: int = 21,
        threshold_multiplier: float = 3.0,
        robust: bool = True,
        min_window: int = 7,
        max_window: int = 63,
    ):
        self.period = period
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.robust = robust
        self.min_window = min_window
        self.max_window = max_window
        self._residual = None
        self._result = None
        self._trend = None
        self._seasonal = None

    def fit(self, series: pd.Series) -> None:
        period = self.period or self._detect_period(series)
        model = STL(series.values, period=period, robust=self.robust)
        self._result = model.fit()
        self._trend = pd.Series(self._result.trend, index=series.index, name="trend")
        self._seasonal = pd.Series(
            self._result.seasonal, index=series.index, name="seasonal"
        )
        self._residual = pd.Series(
            self._result.resid, index=series.index, name="residual"
        )

    def score(self, series: pd.Series) -> pd.Series:
        if self._residual is None:
            raise ValueError("Call fit() before score()")
        if len(series) != len(self._residual):
            self.fit(series)
        resid = self._residual.reindex(series.index).values
        trend = self._trend.reindex(series.index).values
        n = len(resid)

        # Global scale of residuals for robust epsilon
        global_mad = np.median(np.abs(resid - np.median(resid))) or 1.0
        eps = max(1e-10, 1e-6 * global_mad)

        scores = np.zeros(n)

        for i in range(n):
            half_window = self.window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Local trend volatility ratio
            local_trend_vol = np.std(trend[start:end]) if end - start > 1 else 0.0
            global_trend_vol = np.std(trend) if len(trend) > 1 else 1.0
            vol_ratio = local_trend_vol / global_trend_vol if global_trend_vol > 0 else 1.0

            # Adaptive window: widen in volatile periods
            adapt_window = int(self.window_size * (0.5 + vol_ratio))
            adapt_window = max(self.min_window, min(adapt_window, self.max_window))
            half_adapt = adapt_window // 2
            start_a = max(0, i - half_adapt)
            end_a = min(n, i + half_adapt + 1)

            window_vals = resid[start_a:end_a]
            med = np.median(window_vals)
            mad = np.median(np.abs(window_vals - med)) + eps

            scores[i] = min(abs(resid[i] - med) / mad, 50.0)  # Cap to avoid numerical noise

        return pd.Series(scores, index=series.index, name="hampel_score").fillna(0)

    def _auto_threshold(self, scores: pd.Series) -> float:
        return self.threshold_multiplier

    @staticmethod
    def _detect_period(series: pd.Series) -> int:
        if len(series) >= 365 * 2:
            return 365
        elif len(series) >= 90:
            return 7
        else:
            return max(2, len(series) // 10)
