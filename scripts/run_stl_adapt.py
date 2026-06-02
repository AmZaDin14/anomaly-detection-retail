#!/usr/bin/env python
"""
Phase 1 validation: Run STL-Adapt methods on M5 dataset only.
Compares: STL+IQR (baseline), STL-Hampel, STL-SeasonalAdapt, STL-HybridAE
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
    from src.evaluation.metrics import evaluate_detection

    results = []
    total = sum(len(ds["configs"]) for ds in datasets)
    log.info("Running %s on %d dataset configs...", name, total)
    i = 0

    signal.signal(signal.SIGALRM, _timeout_handler)

    for ds in datasets:
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
                    "dataset": "m5",
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
    log.info("PHASE 1: STL-Adapt validation on M5")
    log.info("=" * 60)

    # Load M5 only
    log.info("Loading M5...")
    path = RESULTS_DIR / "m5_prepared.pkl"
    with open(path, "rb") as f:
        datasets = pickle.load(f)
    for ds in datasets:
        ds["dataset_name"] = "m5"
    log.info("Loaded %d series", len(datasets))

    from src.methods.stl_iqr import STLIQRDetector
    from src.methods.stl_hampel import STLHampelDetector
    from src.methods.stl_seasonal_adapt import STLSeasonalAdaptDetector
    from src.methods.stl_hybrid_ae import STLHybridAEDetector

    methods = [
        ("STL+IQR", STLIQRDetector, {"iqr_multiplier": 1.5}, 60),
        ("STL-Hampel", STLHampelDetector, {"threshold_multiplier": 3.0, "window_size": 21}, 60),
        ("STL-SeasonalAdapt", STLSeasonalAdaptDetector, {"base_multiplier": 1.5, "amplitude_factor": 1.5}, 60),
        ("STL-HybridAE", STLHybridAEDetector, {"epochs": 20, "window": 14}, 120),
    ]

    all_results = []
    for method_name, detector_cls, kwargs, timeout in methods:
        res = run_method(method_name, detector_cls, kwargs, datasets, timeout=timeout)
        all_results.extend(res)

    import pandas as pd
    df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "stl_adapt_m5_results.parquet"
    df.to_parquet(out_path)
    log.info("Saved %d rows to %s", len(df), out_path)

    # Print summary
    summary = df.groupby(["method", "anomaly_rate"])[["precision", "recall", "f1", "auc_roc", "auc_pr"]].mean().round(4)
    log.info("\n=== SUMMARY (M5 only) ===\n%s", summary.to_string())

    # Also print computational cost
    times = df.groupby("method")["fit_time_s"].agg(["mean", "std"]).round(4)
    log.info("\n=== FIT TIME ===\n%s", times.to_string())

    log.info("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
