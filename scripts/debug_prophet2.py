"""Test Prophet after other methods are loaded."""
import logging
logging.basicConfig(level=logging.INFO)

# Import other methods first (same order as test_methods.py)
from src.methods.isolation_forest import IsolationForestDetector
from src.methods.lof import LOFDetector
from src.methods.autoencoder import AutoencoderDetector
from src.methods.stl_iqr import STLIQRDetector

print("Other methods loaded successfully")

# Now import and test Prophet
from src.methods.prophet_residual import ProphetResidualDetector
print("Prophet imported successfully")

import pandas as pd
import numpy as np
rng = np.random.default_rng(42)
series = pd.Series(rng.normal(100, 20, 365), index=pd.date_range("2023-01-01", periods=365, freq="D"))

det = ProphetResidualDetector()
det.fit(series)
print("Prophet fitted OK")

scores = det.score(series)
print(f"Prophet scored OK: mean={scores.mean():.3f}")
