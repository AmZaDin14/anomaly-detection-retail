import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
from src.methods.prophet_residual import ProphetResidualDetector

rng = np.random.default_rng(42)
series = pd.Series(rng.normal(100, 20, 365), index=pd.date_range("2023-01-01", periods=365, freq="D"))

det = ProphetResidualDetector()
try:
    det.fit(series)
    print("Fitted OK")
except Exception as e:
    print(f"Fit FAILED: {e}")
    import traceback
    traceback.print_exc()
