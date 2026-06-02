#!/usr/bin/env python
"""
Phase 3: Full experiment across all datasets.
Runs STL-PELT_0.7 and key baselines on synthetic + calendar data.
"""

import logging, pickle, time, signal, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"


class MethodTimeout(Exception):
    pass


def _handler(signum, frame):
    raise MethodTimeout("Timed out")


def run_method(name, cls, kwargs, datasets, timeout=120):
    from src.evaluation.metrics import evaluate_detection
    results = []
    total = sum(len(d["configs"]) for d in datasets)
    log.info("Running %s on %d configs...", name, total)
    i = 0
    for ds in datasets:
        sid = ds["name"]
        ds_name = ds.get("dataset_name", "unknown")
        for cfg in ds["configs"]:
            i += 1
            if i % 10 == 0 or i == 1:
                log.info("  [%s] %d/%d", name, i, total)
            contaminated, labels, rate = (
                cfg["contaminated"], cfg["labels"], cfg["cfg"]["rate"],
            )
            try:
                detector = cls(**kwargs)
                t0 = time.time()
                signal.signal(signal.SIGALRM, _handler)
                signal.alarm(timeout)
                detector.fit(contaminated)
                scores = detector.score(contaminated)
                predictions = detector.predict(contaminated)
                signal.alarm(0)
                fit_time = time.time() - t0
                metrics = evaluate_detection(labels, predictions, scores)
                results.append({
                    "dataset": ds_name, "series_id": sid,
                    "method": name, "anomaly_rate": rate,
                    "fit_time_s": round(fit_time, 4), **metrics,
                })
            except MethodTimeout:
                log.warning("  [%s] TIMEOUT %s rate=%.2f", name, sid, rate)
            except Exception as e:
                log.warning("  [%s] FAILED %s rate=%.2f: %s", name, sid, rate, e)
    signal.alarm(0)
    return results


def load(name):
    path = RESULTS_DIR / f"{name}_prepared.pkl"
    with open(path, "rb") as f:
        datasets = pickle.load(f)
    for ds in datasets:
        ds["dataset_name"] = name
    log.info("Loaded %s: %d series", name, len(datasets))
    return datasets


def main():
    log.info("=" * 60)
    log.info("PHASE 3: Full experiment")
    log.info("=" * 60)

    from src.methods.stl_iqr import STLIQRDetector
    from src.methods.stl_pelt import STLPELTDetector
    from src.methods.autoencoder import AutoencoderDetector

    # =============================================
    # 1. STL-PELT_0.7 on ALL datasets (M5 already done)
    # =============================================
    pelt_kwargs = {"alpha": 0.7}

    datasets_to_run = [
        ("food_prices", pelt_kwargs, 120),
        ("food_prices_calendar", pelt_kwargs, 120),
        ("online_retail", pelt_kwargs, 120),
        ("m5", pelt_kwargs, 120),  # re-run for consistency
    ]

    all_results = []

    for ds_name, kwargs, timeout in datasets_to_run:
        try:
            datasets = load(ds_name)
        except FileNotFoundError:
            log.warning("Skipping %s (not found)", ds_name)
            continue
        res = run_method("STL-PELT_0.7", STLPELTDetector, kwargs, datasets, timeout)
        all_results.extend(res)

    # =============================================
    # 2. STL+IQR on calendar data (needed as baseline)
    # =============================================
    try:
        cal_data = load("food_prices_calendar")
        res = run_method("STL+IQR", STLIQRDetector, {"iqr_multiplier": 1.5}, cal_data, 60)
        all_results.extend(res)
    except FileNotFoundError:
        pass

    # =============================================
    # 3. Autoencoder on calendar data
    # =============================================
    try:
        cal_data = load("food_prices_calendar")
        res = run_method("Autoencoder", AutoencoderDetector, {"window": 14, "epochs": 30}, cal_data, 120)
        all_results.extend(res)
    except FileNotFoundError:
        pass

    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "phase3_results.parquet"
    df.to_parquet(out_path)
    log.info("Saved %d rows to %s", len(df), out_path)

    # Print summary
    summary = df.groupby(["dataset", "method", "anomaly_rate"])[
        ["precision", "recall", "f1"]
    ].mean().round(4)
    log.info("\n=== PHASE 3 RESULTS ===\n%s", summary.to_string())

    log.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
