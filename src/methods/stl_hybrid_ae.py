"""
STL-HybridAE: STL residuals + lightweight autoencoder.

Instead of thresholding STL residuals directly with a fixed statistic (IQR, MAD),
train a tiny autoencoder on sliding windows of the residual component.

The AE learns the "normal" residual pattern in a compressed latent space.
Anomalies violate this pattern → high reconstruction error → anomaly flag.

Key: only ~100 parameters, trains in <0.1s on CPU — far lighter than the
full-data autoencoder (which operates on raw series with sliding window features).
"""

from typing import Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.seasonal import STL

from .base import BaseDetector


class _ResidualAE(nn.Module):
    """Minimal autoencoder for STL residual windows."""

    def __init__(self, input_dim: int, encoding_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class STLHybridAEDetector(BaseDetector):
    """STL residuals + lightweight autoencoder hybrid.

    Pipeline:
    1. STL decomposition → residual component
    2. Sliding windows of residual → tiny AE (input_dim → 4 → input_dim)
    3. Reconstruction error (MSE) per window → anomaly score per point

    Novelty: STL removes seasonal/trend structure first, so the AE only needs
    to model short-range residual patterns. This avoids the deep architecture
    needed for raw data autoencoders.
    """

    def __init__(
        self,
        period: Optional[int] = None,
        window: int = 14,
        encoding_dim: int = 4,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        robust: bool = True,
        device: Optional[str] = None,
    ):
        self.period = period
        self.window = window
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.robust = robust
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._residual = None
        self._result = None

    def fit(self, series: pd.Series) -> None:
        # 1. STL decomposition
        period = self.period or self._detect_period(series)
        model = STL(series.values, period=period, robust=self.robust)
        self._result = model.fit()
        self._residual = pd.Series(
            self._result.resid, index=series.index, name="residual"
        )

        # 2. Build sliding windows of residuals
        X, _ = self._make_windows(self._residual)

        if len(X) < 5:
            # Too few windows — fall back to IQR on residuals
            self._model = None
            return

        # 3. Train tiny AE
        self._model = _ResidualAE(self.window, self.encoding_dim).to(self.device)
        dataset = TensorDataset(
            torch.tensor(X.values, dtype=torch.float32).to(self.device)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstructed = self._model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

        self._model.eval()

    def score(self, series: pd.Series) -> pd.Series:
        if self._residual is None:
            raise ValueError("Call fit() before score()")
        if len(series) != len(self._residual):
            self.fit(series)

        resid = self._residual.reindex(series.index, fill_value=0)

        if self._model is None:
            # Fallback: simple MAD-based scoring
            med = np.median(resid.values)
            mad = np.median(np.abs(resid.values - med)) + 1e-10
            scores = (resid - med).abs() / mad
            return scores.fillna(0)

        # Compute reconstruction error per window
        X, indices = self._make_windows(resid)
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            reconstructed = self._model(X_tensor)
            errors = (X_tensor - reconstructed).pow(2).mean(dim=1).cpu().numpy()

        # Map window-level errors back to point-level scores
        scores = pd.Series(0.0, index=resid.index, dtype=float)
        for idx, err in zip(indices, errors):
            # Assign error to the last point of the window
            if idx in scores.index:
                scores[idx] = err

        # Fill gaps with forward fill, then 0
        scores = scores.replace(0, np.nan).ffill().fillna(0)
        return scores

    def _make_windows(self, series: pd.Series):
        """Create sliding windows from a time series."""
        n = len(series)
        rows, indices = [], []
        for i in range(self.window, n):
            chunk = series.iloc[i - self.window : i]
            rows.append(chunk.values)
            indices.append(chunk.index[-1])
        X = pd.DataFrame(rows, index=indices)
        return X, indices

    def _auto_threshold(self, scores: pd.Series) -> float:
        non_zero = scores[scores > 0]
        if len(non_zero) < 3:
            return scores.quantile(0.95)
        return float(non_zero.mean() + 2 * non_zero.std())

    @staticmethod
    def _detect_period(series: pd.Series) -> int:
        if len(series) >= 365 * 2:
            return 365
        elif len(series) >= 90:
            return 7
        else:
            return max(2, len(series) // 10)
