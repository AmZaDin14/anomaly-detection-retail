#!/usr/bin/env python
"""STL-PELT validation on M5. Compares alpha variants vs STL+IQR."""

import logging, pickle, time, signal, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"

class MethodTimeout(Exception): pass

def _handler(signum, frame): raise MethodTimeout("Timed out")

def run(name, cls, kwargs, datasets, timeout=60):
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
                results.append({"dataset":"m5","series_id":sid,"method":name,
                    "anomaly_rate":rate,"fit_time_s":round(fit_time,4),**metrics})
            except MethodTimeout:
                log.warning("  [%s] TIMEOUT %s rate=%.2f", name, sid, rate)
            except Exception as e:
                log.warning("  [%s] FAILED %s rate=%.2f: %s", name, sid, rate, e)
    signal.alarm(0)
    return results

def main():
    log.info("STL-PELT validation on M5")
    with open(RESULTS_DIR / "m5_prepared.pkl", "rb") as f:
        datasets = pickle.load(f)

    from src.methods.stl_iqr import STLIQRDetector
    from src.methods.stl_pelt import STLPELTDetector

    methods = [
        ("STL+IQR", STLIQRDetector, {"iqr_multiplier": 1.5}, 60),
        ("STL-PELT_0.5", STLPELTDetector, {"alpha": 0.5}, 120),
        ("STL-PELT_0.7", STLPELTDetector, {"alpha": 0.7}, 120),
        ("STL-PELT_0.9", STLPELTDetector, {"alpha": 0.9}, 120),
    ]

    all_results = []
    for name, cls, kwargs, timeout in methods:
        res = run(name, cls, kwargs, datasets, timeout)
        all_results.extend(res)

    import pandas as pd
    df = pd.DataFrame(all_results)
    out = RESULTS_DIR / "stl_pelt_m5_results.parquet"
    df.to_parquet(out)
    log.info("Saved %d rows to %s", len(df), out)

    summary = df.groupby(["method","anomaly_rate"])[["precision","recall","f1","auc_roc","auc_pr"]].mean().round(4)
    log.info("\n%s", summary.to_string())

    bl = df[df["method"]=="STL+IQR"].groupby("anomaly_rate")["f1"].mean()
    for name in ["STL-PELT_0.5","STL-PELT_0.7","STL-PELT_0.9"]:
        m = df[df["method"]==name].groupby("anomaly_rate")["f1"].mean()
        diffs = ((m - bl) / bl * 100).round(1)
        log.info("%s vs baseline:", name)
        for r in [0.01,0.05,0.10]:
            log.info("  rate=%.2f: %.4f vs %.4f (%+.1f%%)", r, m.get(r,0), bl.get(r,0), diffs.get(r,0))

    log.info("Done.")

if __name__ == "__main__":
    main()
