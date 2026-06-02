"""
STL-SeasonalAdapt: STL decomposition + season-aware adaptive thresholding.

Standard STL+IQR uses a fixed multiplier (1.5) across all seasonal phases.
This variant adjusts the threshold multiplier based on local seasonal volatility:
- Stable seasons (low seasonal amplitude): tighter threshold → fewer false positives
- Volatile seasons (high amplitude / transitions): looser threshold → catch real

The seasonal strength is computed per phase using rolling statistics of the
seasonal component from STL decomposition.
"""

from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from .base import BaseDetector


class STLSeasonalAdaptDetector(BaseDetector):
    """STL + season-aware adaptive thresholding.

    Multiplier k varies by seasonal phase:
      k(t) = base_multiplier + amplitude_factor * (local_seasonal_std / global_seasonal_std)

    In stable periods, k tightens; in volatile periods (e.g., Ramadhan for food prices),
    k loosens to avoid false positives during natural price spikes.
    """

    def __init__(
        self,
        period: Optional[int] = None,
        base_multiplier: float = 1.5,
        amplitude_factor: float = 1.5,
        robust: bool = True,
        smooth_window: int = 7,
    ):
        self.period = period
        self.base_multiplier = base_multiplier
        self.amplitude_factor = amplitude_factor
        self.robust = robust
        self.smooth_window = smooth_window
        self._residual = None
        self._seasonal = None
        self._result = None
        self._adaptive_multiplier = None

    def fit(self, series: pd.Series) -> None:
        period = self.period or self._detect_period(series)
        model = STL(series.values, period=period, robust=self.robust)
        self._result = model.fit()
        self._residual = pd.Series(
            self._result.resid, index=series.index, name="residual"
        )
        self._seasonal = pd.Series(
            self._result.seasonal, index=series.index, name="seasonal"
        )

        # Build adaptive multiplier profile
        self._adaptive_multiplier = self._compute_adaptive_multiplier(series.index)

    def _compute_adaptive_multiplier(self, index) -> pd.Series:
        """Compute time-varying threshold multiplier.

        Uses rolling std of seasonal component to detect volatile periods.
        """
        seasonal = self._seasonal.reindex(index).values
        n = len(seasonal)
        global_std = np.std(seasonal) if len(seasonal) > 1 else 1.0

        multiplier = np.full(n, self.base_multiplier)

        half_w = self.smooth_window // 2
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            local_std = np.std(seasonal[start:end]) if end - start > 1 else 0.0

            if global_std > 0:
                vol_ratio = local_std / global_std
            else:
                vol_ratio = 1.0

            # k = base + alpha * vol_ratio
            # This yields k ≈ 1.5 in stable periods, up to ~4.5 in highly volatile
            multiplier[i] = self.base_multiplier + self.amplitude_factor * vol_ratio

        # Smooth the multiplier profile to avoid abrupt changes
        if n >= self.smooth_window * 2:
            from scipy.ndimage import uniform_filter1d as uf1d
            multiplier = uf1d(multiplier, size=self.smooth_window, mode="reflect")

        return pd.Series(multiplier, index=index, name="adaptive_multiplier")

    def score(self, series: pd.Series) -> pd.Series:
        if self._residual is None:
            raise ValueError("Call fit() before score()")
        if len(series) != len(self._residual):
            self.fit(series)

        resid = self._residual.reindex(series.index)

        # Robust MAD with data-scale epsilon
        raw_mad = np.median(np.abs(resid.values - np.median(resid.values)))
        global_mad = raw_mad if raw_mad > 0 else resid.std() if resid.std() > 0 else 1.0
        eps = max(1e-10, 1e-6 * global_mad)
        mad = global_mad + eps

        # Score = |residual| / (MAD * adaptive_multiplier)
        # Normalized: score > 1 means anomaly (more than k*MAD from median)
        multiplier = self._adaptive_multiplier.reindex(series.index).values
        scores = (resid.abs().values / (mad * multiplier)).clip(0, 50)
        return pd.Series(scores, index=series.index, name="seasonal_adapt_score").fillna(0)

    def _auto_threshold(self, scores: pd.Series) -> float:
        return 1.0  # Score normalized: >1 means exceeds adaptive threshold

    @staticmethod
    def _detect_period(series: pd.Series) -> int:
        if len(series) >= 365 * 2:
            return 365
        elif len(series) >= 90:
            return 7
        else:
            return max(2, len(series) // 10)
