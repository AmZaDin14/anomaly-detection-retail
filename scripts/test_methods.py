#!/usr/bin/env python
"""Quick smoke test for all detection methods."""
import logging
import time
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np

print("Loading methods...")
t0 = time.time()
from src.methods.isolation_forest import IsolationForestDetector
from src.methods.lof import LOFDetector
from src.methods.autoencoder import AutoencoderDetector
from src.methods.prophet_residual import ProphetResidualDetector
from src.methods.stl_iqr import STLIQRDetector
print(f"Loaded in {time.time()-t0:.1f}s")

rng = np.random.default_rng(42)
series = pd.Series(rng.normal(100, 20, 365), index=pd.date_range("2023-01-01", periods=365, freq="D"))

for name, cls, kwargs in [
    ("IForest", IsolationForestDetector, {"window": 14}),
    ("LOF", LOFDetector, {"window": 14}),
    ("AE", AutoencoderDetector, {"window": 14, "epochs": 10}),
    ("Prophet", ProphetResidualDetector, {}),
    ("STL+IQR", STLIQRDetector, {}),
]:
    try:
        t0 = time.time()
        det = cls(**kwargs)
        det.fit(series)
        scores = det.score(series)
        n_anom = det.predict(series).sum()
        print(f"  {name:<12} {time.time()-t0:.2f}s  scores={scores.mean():.3f}±{scores.std():.3f}  anomalies={n_anom}")
    except Exception as e:
        print(f"  {name:<12} FAILED: {e}")

print("Done!")
