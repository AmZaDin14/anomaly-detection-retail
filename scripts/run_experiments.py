#!/usr/bin/env python
"""
Phase 2+3: Run all anomaly detection methods on all datasets and evaluate.
"""

import logging
from pathlib import Path
import sys
import pickle
import time
import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"


class MethodTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise MethodTimeout("Method timed out")


def run_method(name, detector_class, detector_kwargs, datasets, timeout=120):
    """Run a detection method on all datasets with per-call timeout."""
    from src.evaluation.metrics import evaluate_detection

    results = []
    total = sum(len(ds["configs"]) for ds in datasets)
    log.info("Running %s on %d dataset configs...", name, total)
    i = 0

    signal.signal(signal.SIGALRM, _timeout_handler)

    for ds in datasets:
        clean = ds["clean"]
        sid = ds["name"]
        for cfg_entry in ds["configs"]:
            i += 1
            contaminated = cfg_entry["contaminated"]
            labels = cfg_entry["labels"]
            rate = cfg_entry["cfg"]["rate"]

            if i % 10 == 0 or i == 1:
                log.info("  [%s] %d/%d", name, i, total)

            try:
                detector = detector_class(**detector_kwargs)
                t0 = time.time()

                signal.alarm(timeout)
                detector.fit(contaminated)
                scores = detector.score(contaminated)
                predictions = detector.predict(contaminated)
                signal.alarm(0)

                fit_time = time.time() - t0
                metrics = evaluate_detection(labels, predictions, scores)
                results.append({
                    "dataset": ds.get("dataset_name", "?"),
                    "series_id": sid,
                    "method": name,
                    "anomaly_rate": rate,
                    "fit_time_s": round(fit_time, 4),
                    **metrics,
                })
            except MethodTimeout:
                log.warning("  [%s] TIMEOUT on %s rate=%.2f", name, sid, rate)
            except Exception as e:
                log.warning("  [%s] FAILED on %s rate=%.2f: %s", name, sid, rate, e)

    signal.alarm(0)
    return results


def main():
    log.info("=" * 60)
    log.info("PHASES 2+3: Running anomaly detection experiments")
    log.info("=" * 60)

    dataset_names = ["m5", "food_prices", "online_retail"]
    all_results = []

    from src.methods.isolation_forest import IsolationForestDetector
    from src.methods.lof import LOFDetector
    from src.methods.autoencoder import AutoencoderDetector
    from src.methods.prophet_residual import ProphetResidualDetector
    from src.methods.stl_iqr import STLIQRDetector

    methods = [
        ("IsolationForest", IsolationForestDetector, {"window": 14, "contamination": 0.05}, 60),
        ("LOF", LOFDetector, {"window": 14, "n_neighbors": 20, "contamination": 0.05}, 60),
        ("Autoencoder", AutoencoderDetector, {"window": 14, "epochs": 30}, 120),
        ("ProphetResidual", ProphetResidualDetector, {"residual_threshold": 2.5, "fit_timeout": 60}, 120),
        ("STL+IQR", STLIQRDetector, {"iqr_multiplier": 1.5}, 60),
    ]

    for ds_name in dataset_names:
        log.info("\n--- Loading %s ---", ds_name)
        datasets = load_prepared(ds_name)
        log.info("Loaded %d series for %s", len(datasets), ds_name)

        for ds in datasets:
            ds["dataset_name"] = ds_name

        for method_name, detector_cls, kwargs, timeout in methods:
            res = run_method(method_name, detector_cls, kwargs, datasets, timeout=timeout)
            all_results.extend(res)

    import pandas as pd
    df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "all_results.parquet"
    df.to_parquet(out_path)
    log.info("\nSaved %d result rows to %s", len(df), out_path)

    from src.evaluation.metrics import evaluate_all_series
    summary = evaluate_all_series(all_results)
    log.info("\n=== SUMMARY ===\n%s", summary.to_string())
    log.info("\n✅ Phases 2+3 complete.")


def load_prepared(name):
    path = RESULTS_DIR / f"{name}_prepared.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
