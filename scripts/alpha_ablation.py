#!/usr/bin/env python3
"""Alpha ablation study for STL-PELT on calendar data.

Tests how the alpha parameter (balance between residual magnitude and
change point proximity) affects detection performance.
"""

import pickle, sys, time, signal, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA = Path(__file__).resolve().parent.parent / "data" / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class MethodTimeout(Exception):
    pass


def _handler(signum, frame):
    raise MethodTimeout("Timed out")


def run_alpha(name, cls, kwargs, datasets, timeout=120):
    from src.evaluation.metrics import evaluate_detection
    results = []
    total = sum(len(d["configs"]) for d in datasets)
    log.info("Running %s on %d configs...", name, total)
    i = 0
    for ds in datasets:
        sid = ds["name"]
        for cfg in ds["configs"]:
            i += 1
            if i % 10 == 0 or i == 1:
                log.info("  [%s] %d/%d", name, i, total)
            contaminated, labels, rate = cfg["contaminated"], cfg["labels"], cfg["cfg"]["rate"]
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
                    "dataset": "food_prices_calendar", "series_id": sid,
                    "method": name, "anomaly_rate": rate,
                    "alpha": kwargs.get("alpha", None),
                    "fit_time_s": round(fit_time, 4), **metrics,
                })
            except Exception as e:
                log.warning("  [%s] FAILED %s rate=%.2f: %s", name, sid, rate, e)
    signal.alarm(0)
    return results


def main():
    from src.methods.stl_pelt import STLPELTDetector

    with open(DATA / "food_prices_calendar_prepared.pkl", "rb") as f:
        datasets = pickle.load(f)

    # Alpha sweep: 0.0 (pure CP proximity) to 1.0 (pure residual magnitude)
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    all_results = []
    for alpha in alphas:
        name = f"STL-PELT_a={alpha}"
        res = run_alpha(name, STLPELTDetector, {"alpha": alpha}, datasets, timeout=120)
        all_results.extend(res)

    # Save
    df = pd.DataFrame(all_results)
    out_path = DATA / "alpha_ablation_results.parquet"
    df.to_parquet(out_path)
    log.info("Saved %d rows to %s", len(df), out_path)

    # Print summary
    print("\n" + "=" * 80)
    print("ALPHA ABLATION RESULTS (Food Prices Calendar)")
    print("=" * 80)

    for rate in [0.01, 0.05, 0.10]:
        print(f"\n--- Anomaly Rate: {rate*100:.0f}% ---")
        sub = df[df["anomaly_rate"] == rate]
        pivot = sub.groupby(["alpha", "method"])[
            ["precision", "recall", "f1"]
        ].mean().round(4)
        for alpha in alphas:
            row = pivot.loc[alpha]
            print(f"  alpha={alpha:.1f}: P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")

    # Find best alpha per rate
    print("\n" + "=" * 80)
    print("OPTIMAL ALPHA PER ANOMALY RATE")
    print("=" * 80)
    for rate in [0.01, 0.05, 0.10]:
        sub = df[df["anomaly_rate"] == rate]
        best = sub.loc[sub.groupby("series_id")["f1"].transform("mean").groupby(sub["alpha"]).mean().idxmax()]
        best_alpha = sub.groupby("alpha")["f1"].mean().idxmax()
        best_f1 = sub.groupby("alpha")["f1"].mean().max()
        base_f1 = sub[sub["alpha"] == 0.7]["f1"].mean()
        print(f"  @{rate*100:.0f}%: best alpha={best_alpha:.1f} (F1={best_f1:.4f}, vs alpha=0.7: F1={base_f1:.4f})")

    # Overall best
    overall = df.groupby("alpha")["f1"].mean()
    best_alpha = overall.idxmax()
    best_f1 = overall.max()
    print(f"\n  OVERALL: best alpha={best_alpha:.1f} (F1={best_f1:.4f})")

    log.info("Alpha ablation complete.")


if __name__ == "__main__":
    main()
