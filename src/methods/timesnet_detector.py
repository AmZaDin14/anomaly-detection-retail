"""
Minimal TimesNet for anomaly detection.

Based on: Wu et al. (2023) "TimesNet: Temporal 2D-Variation Modeling
for General Time Series Analysis." ICLR 2023.

Core idea:
1. FFT to find dominant periods in the time series
2. Reshape 1D -> 2D by folding along each period
3. 2D convolutions capture inter-period (across columns) and
   intra-period (within columns) patterns
4. Reshape back -> reconstruction error as anomaly score

This is a stripped-down version: 1 TimesBlock, no Inception modules
(just 2D conv), focused on fast per-series training.
"""

from typing import Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector


class _TimesBlock(nn.Module):
    """Single TimesNet block with period-based 2D convolution."""

    def __init__(self, seq_len: int, top_k: int = 3, d_model: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model

        # 2D convolutions for each selected period
        # Shared across periods but applied separately
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        B, L = x.shape

        # 1. FFT to find dominant periods
        X_fft = torch.fft.rfft(x, dim=1)
        amplitude = X_fft.abs().mean(dim=0)  # (L//2 + 1,)
        amplitude[0] = 0  # Ignore DC component

        # Top-k frequencies
        _, top_indices = torch.topk(amplitude, min(self.top_k, len(amplitude)))
        periods = (L / (top_indices + 1e-8)).long().clamp(min=2, max=L // 2)

        # 2. For each period, reshape to 2D and apply conv
        outputs = []
        for period in periods:
            period = min(period.item(), L // 2)  # Safety check
            if period < 2:
                continue

            # Pad to make divisible
            n_folds = (L + period - 1) // period
            pad_len = n_folds * period - L
            x_pad = torch.nn.functional.pad(x, (0, pad_len))

            # Reshape to 2D: (B, n_folds, period)
            x_2d = x_pad.view(B, n_folds, period)

            # Add channel dim: (B, 1, n_folds, period)
            x_2d = x_2d.unsqueeze(1)

            # Apply 2D conv
            out_2d = self.conv(x_2d)  # (B, 1, n_folds, period)

            # Reshape back to 1D
            out_1d = out_2d.view(B, -1)[:, :L]  # Trim padding
            outputs.append(out_1d)

        if not outputs:
            return x

        # 3. Weighted aggregation (amplitude-based weights)
        weights = amplitude[top_indices][:len(outputs)]
        weights = torch.softmax(weights, dim=0)
        output = sum(w * o for w, o in zip(weights, outputs))
        return output


class _TimesNet(nn.Module):
    """Minimal TimesNet for reconstruction."""

    def __init__(self, seq_len: int, d_model: int = 32):
        super().__init__()
        # Project 1D input to d_model for the block
        self.input_proj = nn.Linear(1, d_model)
        self.block = _TimesBlock(seq_len, top_k=3, d_model=d_model)
        self.output_proj = nn.Linear(1, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        # Embed to d_model: (batch, seq_len, d_model)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        # Average over d_model for 1D block input: (batch, seq_len)
        x_1d = x.mean(dim=-1)

        # TimesBlock: (batch, seq_len)
        x_out = self.block(x_1d)

        # Project back: (batch, seq_len, 1) -> (batch, seq_len)
        x_out = x_out.unsqueeze(-1)
        x_out = self.output_proj(x_out).squeeze(-1)
        return x_out


class TimesNetDetector(BaseDetector):
    """TimesNet-based anomaly detector using reconstruction error.

    Trains TimesNet to reconstruct the input series (autoencoding task).
    Anomaly score = MSE between input and reconstruction.
    """

    def __init__(
        self,
        window: int = 64,
        d_model: int = 32,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.window = window
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def fit(self, series: pd.Series) -> None:
        # Create overlapping windows
        values = series.values.astype(np.float32)
        n = len(values)
        windows = []
        for i in range(0, n - self.window, self.window // 2):
            w = values[i:i + self.window]
            if len(w) == self.window:
                windows.append(w)

        if len(windows) < 3:
            window = min(self.window, n // 2)
            if window < 4:
                return
            windows = [values[:window]]
            for i in range(window, n, window):
                w = values[max(0, i - window):i]
                if len(w) == window:
                    windows.append(w)

        X = np.stack(windows)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self._model = _TimesNet(self.window, self.d_model).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        self._model.eval()
        self._values = series

    def score(self, series: pd.Series) -> pd.Series:
        if self._model is None:
            raise ValueError("Call fit() before score()")

        values = series.values.astype(np.float32)
        n = len(values)
        scores = np.zeros(n)

        # Score each point by the reconstruction error of the window containing it
        step = max(1, self.window // 4)
        score_counts = np.zeros(n)

        for i in range(0, n - self.window + 1, step):
            w = values[i:i + self.window]
            w_tensor = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                recon = self._model(w_tensor).cpu().numpy()[0]

            err = (w - recon) ** 2
            for j in range(self.window):
                idx = i + j
                scores[idx] += err[j]
                score_counts[idx] += 1

        # Average overlapping scores
        score_counts = np.maximum(score_counts, 1)
        scores = scores / score_counts

        return pd.Series(scores, index=series.index, name="timesnet_score").fillna(0)

    def _auto_threshold(self, scores: pd.Series) -> float:
        non_zero = scores[scores > 0]
        if len(non_zero) < 3:
            return scores.quantile(0.95)
        return float(non_zero.mean() + 2 * non_zero.std())
