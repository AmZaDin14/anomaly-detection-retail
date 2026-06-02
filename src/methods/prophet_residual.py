"""
Prophet-based anomaly detector with timeout protection.

Fits Prophet to forecast expected values, then flags anomalies where
the residual (actual - forecast) exceeds a threshold.
"""

import signal
import pandas as pd
import numpy as np
from contextlib import contextmanager

from .base import BaseDetector


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Raise TimeoutError if code runs longer than `seconds`."""

    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ProphetResidualDetector(BaseDetector):
    """
    Detect anomalies via Prophet forecast residuals.

    Prophet models trend + seasonality. Points far from the forecast
    are flagged. Includes a timeout to handle problematic series.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        residual_threshold: float = 2.5,
        fit_timeout: int = 60,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.residual_threshold = residual_threshold
        self.fit_timeout = fit_timeout
        self._model = None
        self._forecast = None

    def fit(self, series: pd.Series) -> None:
        from prophet import Prophet

        df = pd.DataFrame({"ds": series.index, "y": series.values})

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=False,
        )

        try:
            with time_limit(self.fit_timeout):
                self._model.fit(df)
                forecast = self._model.predict(df)
                self._forecast = forecast.set_index("ds")
        except TimeoutError:
            raise TimeoutError(
                f"Prophet fitting timed out after {self.fit_timeout}s "
                f"on series with {len(series)} points"
            )

    def score(self, series: pd.Series) -> pd.Series:
        if self._forecast is None:
            raise ValueError("Call fit() before score()")

        fc = self._forecast.reindex(series.index, method="nearest", tolerance="1D")
        if fc.isna().any().any():
            fc = fc.fillna(method="ffill").fillna(method="bfill")

        predicted = fc["yhat"].values
        yhat_upper = fc["yhat_upper"].values
        yhat_lower = fc["yhat_lower"].values

        actual = series.values
        uncertainty = (yhat_upper - yhat_lower).clip(1.0)
        residuals = np.abs(actual - predicted)
        normalized_residuals = residuals / uncertainty

        scores = pd.Series(normalized_residuals, index=series.index)
        return scores.fillna(0)

    def _auto_threshold(self, scores: pd.Series) -> float:
        return self.residual_threshold
