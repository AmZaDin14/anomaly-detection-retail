"""
STL-PELT: STL decomposition + PELT change point detection on residuals.

Novel hybrid approach:
1. STL decomposes series into trend + seasonal + residual
2. PELT (Pruned Exact Linear Time) detects change points in the residual
3. Combined score = residual magnitude + change point proximity

This catches BOTH types of anomaly:
- Isolated spikes: large |residual|, no change point nearby
- Pattern breaks/persistent shifts: moderate |residual|, near a change point
- Worst-case: large |residual| AND change point nearby → highest confidence

Reference for PELT: Killick, Fearnhead, Eckley (2012) "Optimal detection of
changepoints with a linear computational cost." JASA.
"""

from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import ruptures as rpt

from .base import BaseDetector


class STLPELTDetector(BaseDetector):
    """STL decomposition + PELT change point detection.

    Combines residual magnitude (classic STL+IQR approach) with
    change point proximity (PELT on residuals) for a hybrid score.

    Parameters
    ----------
    period : int, optional
        Seasonality period for STL. Auto-detected if None.
    alpha : float, default=0.7
        Weight for residual magnitude term. (1-alpha) for CP proximity.
    cp_penalty : float, default=None
        PELT penalty. Auto-selected via BIC heuristic if None.
    cp_sigma : float, default=None
        Gaussian decay sigma for CP proximity. Auto-selected if None.
    robust : bool, default=True
        Robust STL fitting.
    """

    def __init__(
        self,
        period: Optional[int] = None,
        alpha: float = 0.7,
        cp_penalty: Optional[float] = None,
        cp_sigma: Optional[float] = None,
        robust: bool = True,
    ):
        self.period = period
        self.alpha = alpha
        self.cp_penalty = cp_penalty
        self.cp_sigma = cp_sigma
        self.robust = robust
        self._residual = None
        self._trend = None
        self._seasonal = None
        self._change_points = None
        self._proximity = None
        self._mad = None

    def fit(self, series: pd.Series) -> None:
        period = self.period or self._detect_period(series)
        model = STL(series.values, period=period, robust=self.robust)
        self._result = model.fit()
        self._trend = pd.Series(self._result.trend, index=series.index, name="trend")
        self._seasonal = pd.Series(
            self._result.seasonal, index=series.index, name="seasonal"
        )
        resid = pd.Series(self._result.resid, index=series.index, name="residual")
        self._residual = resid

        # Robust MAD of residuals
        raw_mad = np.median(np.abs(resid.values - np.median(resid.values)))
        self._mad = raw_mad if raw_mad > 0 else resid.std() if resid.std() > 0 else 1.0

        # PELT change point detection
        self._find_change_points(resid.values)

        # Build proximity score
        self._compute_proximity(series.index)

    def _find_change_points(self, values: np.ndarray) -> None:
        """Run PELT on residuals to find change points."""
        n = len(values)

        # Auto-select penalty via BIC heuristic
        if self.cp_penalty is None:
            sigma2 = np.var(values) if np.var(values) > 0 else 1.0
            self.cp_penalty = 2.0 * np.log(n) * sigma2

        # Run PELT with L2 cost (mean-shift detection) — jump=5 for speed
        algo = rpt.Pelt(model="l2", min_size=5, jump=5).fit(values)
        change_points = algo.predict(pen=self.cp_penalty)

        # PELT returns n as the last element (end of series); remove it
        self._change_points = sorted(set(cp for cp in change_points if cp < n))
        self._raw_values = values

    def _compute_proximity(self, index) -> None:
        """Compute change point proximity score for each point.

        proximity(t) = exp(-min_dist_to_cp² / (2σ²))
        """
        n = len(index)
        if not self._change_points or len(self._change_points) == 0:
            self._proximity = pd.Series(0.0, index=index)
            return

        # Average segment length for adaptive sigma
        if self.cp_sigma is None:
            segments = [self._change_points[0]]
            segments += [
                self._change_points[i] - self._change_points[i - 1]
                for i in range(1, len(self._change_points))
            ]
            mean_seg_len = np.mean(segments) if segments else n // 4
            sigma = max(5, mean_seg_len / 4)
        else:
            sigma = self.cp_sigma

        proximities = np.zeros(n)
        for i in range(n):
            min_dist = min(abs(i - cp) for cp in self._change_points)
            proximities[i] = np.exp(-min_dist * min_dist / (2 * sigma * sigma))

        self._proximity = pd.Series(proximities, index=index, name="proximity")

    def score(self, series: pd.Series) -> pd.Series:
        if self._residual is None:
            raise ValueError("Call fit() before score()")
        if len(series) != len(self._residual):
            self.fit(series)

        resid = self._residual.reindex(series.index).values
        proximity = self._proximity.reindex(series.index).values

        # Residual magnitude score (MAD-normalized)
        mag_scores = resid / self._mad
        mag_scores = np.abs(mag_scores) / (np.median(np.abs(mag_scores)) + 1e-10)

        # Change point proximity score (already 0-1 normalized)
        cp_scores = proximity  # 0 to 1

        # Map CP scores from [0,1] to similar scale as mag scores
        cp_scaled = cp_scores * (np.percentile(mag_scores, 90) + 1e-10)

        # Combined score
        combined = self.alpha * mag_scores + (1 - self.alpha) * cp_scaled

        return pd.Series(combined, index=series.index, name="stl_pelt_score").fillna(0)

    def _auto_threshold(self, scores: pd.Series) -> float:
        """Adaptive threshold based on distribution of combined scores."""
        # Use median + 2*IQR of combined scores
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = max(q3 - q1, 1e-10)
        return float(q3 + 1.5 * iqr)

    @staticmethod
    def _detect_period(series: pd.Series) -> int:
        if len(series) >= 365 * 2:
            return 365
        elif len(series) >= 90:
            return 7
        else:
            return max(2, len(series) // 10)
