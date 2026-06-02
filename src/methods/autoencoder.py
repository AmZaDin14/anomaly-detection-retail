"""
Autoencoder-based anomaly detector.

Trains a simple feedforward autoencoder on sliding-window features.
Anomaly score = reconstruction error (MSE). Higher = more anomalous.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector, sliding_window_features


class _Autoencoder(nn.Module):
    """Simple feedforward autoencoder."""

    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector(BaseDetector):
    """Autoencoder anomaly detector using reconstruction error."""

    def __init__(
        self,
        window: int = 14,
        encoding_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.window = window
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._input_dim = None

    def fit(self, series: pd.Series) -> None:
        X, _ = sliding_window_features(series, window=self.window)
        self._input_dim = X.shape[1]

        torch.manual_seed(self.random_state)
        self._model = _Autoencoder(self._input_dim, self.encoding_dim).to(self.device)

        dataset = TensorDataset(
            torch.tensor(X.values, dtype=torch.float32).to(self.device)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstructed = self._model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._model.eval()

    def score(self, series: pd.Series) -> pd.Series:
        if self._model is None:
            raise ValueError("Call fit() before score()")

        X, indices = sliding_window_features(series, window=self.window)
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            reconstructed = self._model(X_tensor)
            errors = (X_tensor - reconstructed).pow(2).mean(dim=1).cpu().numpy()

        scores = pd.Series(0.0, index=series.index, dtype=float)
        for idx, err in zip(indices, errors):
            if idx in scores.index:
                scores[idx] = err

        return scores

    def _auto_threshold(self, scores: pd.Series) -> float:
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            return scores.quantile(0.95)
        # Use mean + 2*std of non-zero reconstruction errors
        return float(non_zero.mean() + 2 * non_zero.std())


# Optional: silence the type checker at module level
from typing import Optional as _Optional
