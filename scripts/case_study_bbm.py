#!/usr/bin/env python
"""
Case study: BBM price hike September 2022.

On Sept 3, 2022, the Indonesian government raised fuel prices:
- Pertalite: +30%
- Solar: +40%
This caused widespread food price increases.

We test: do our detectors flag anomalies around this event?
"""
import pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


def load_pihps():
    with open(DATA_DIR / "food_prices_prepared.pkl", "rb") as f:
        return pickle.load(f)


def run_detector(name, cls, kwargs, series):
    """Run a single detector on a clean series (no injected anomalies)."""
    import time, signal
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(Exception("Timeout")))
    detector = cls(**kwargs)
    signal.alarm(60)
    detector.fit(series)
    scores = detector.score(series)
    predictions = detector.predict(series)
    signal.alarm(0)
    return scores, predictions


def main():
    from src.methods.stl_iqr import STLIQRDetector
    from src.methods.stl_pelt import STLPELTDetector
    from src.methods.autoencoder import AutoencoderDetector

    data = load_pihps()

    # Event: BBM hike Sept 3, 2022
    event_date = date(2022, 9, 3)
    window = timedelta(days=14)  # ±14 days
    event_start = event_date - window
    event_end = event_date + window

    # Commodities affected by BBM (transport cost pass-through)
    target_series = ["beras", "minyak_goreng", "cabai_rawit"]

    detectors = [
        ("STL+IQR", STLIQRDetector, {"iqr_multiplier": 1.5}),
        ("STL-PELT_0.7", STLPELTDetector, {"alpha": 0.7}),
        ("Autoencoder", AutoencoderDetector, {"window": 14, "epochs": 30}),
    ]

    results = []

    for series_name in target_series:
        # Find the matching series
        matches = [d for d in data if d["name"] == series_name]
        if not matches:
            print(f"Series {series_name} not found")
            continue
        series = matches[0]["clean"]

        # Subset around event
        event_mask = (series.index.date >= event_start) & (series.index.date <= event_end)
        event_series = series[event_mask]
        event_len = len(event_series)

        print(f"\n{'='*60}")
        print(f"Case: {series_name} around BBM hike (Sept 2022)")
        print(f"  Series length: {len(series)}, Event window: {event_len} days")
        print(f"  Window: {event_start} to {event_end}")

        for det_name, det_cls, det_kwargs in detectors:
            scores, preds = run_detector(det_name, det_cls, det_kwargs, series)

            # Anomalies in event window
            event_scores = scores[event_mask]
            event_preds = preds[event_mask]
            n_event_anomalies = event_preds.sum()

            # Anomalies in full series
            n_total_anomalies = preds.sum()

            # Time to first detection (days from event)
            if n_event_anomalies > 0:
                first_anomaly = event_preds[event_preds].index[0].date()
                days_to_detect = (first_anomaly - event_date).days
            else:
                first_anomaly = None
                days_to_detect = None

            # Anomaly rate in window vs baseline
            window_rate = n_event_anomalies / event_len if event_len > 0 else 0
            baseline_rate = n_total_anomalies / len(series)

            # Precision within window: did detections cluster around event?
            # Count anomalies within ±3 days of event
            tight_mask = (series.index.date >= event_date - timedelta(days=3)) & \
                         (series.index.date <= event_date + timedelta(days=3))
            tight_anomalies = preds[tight_mask].sum()

            results.append({
                "commodity": series_name,
                "method": det_name,
                "total_anomalies": int(n_total_anomalies),
                "event_window_anomalies": int(n_event_anomalies),
                "tight_window_anomalies": int(tight_anomalies),
                "days_to_first_detection": days_to_detect,
                "window_anomaly_rate": round(window_rate, 4),
                "baseline_anomaly_rate": round(baseline_rate, 4),
            })

            print(f"\n  {det_name}:")
            print(f"    Anomalies in full series: {n_total_anomalies} ({baseline_rate*100:.1f}%)")
            print(f"    Anomalies in ±14d window: {n_event_anomalies} ({window_rate*100:.1f}%)")
            print(f"    Anomalies within ±3d:     {tight_anomalies}")
            print(f"    Days to first detection:  {days_to_detect}")

    # Summary table
    print(f"\n{'='*60}")
    print("CASE STUDY SUMMARY: BBM Hike Sept 3, 2022")
    print(f"{'='*60}")
    print(f"{'Commodity':20s} {'Method':20s} {'Window Anom':>10s} {'±3d Anom':>10s} {'Days to 1st':>10s}")
    print("-" * 70)
    for r in results:
        d = str(r["days_to_first_detection"]) if r["days_to_first_detection"] is not None else "N/A"
        print(f"{r['commodity']:20s} {r['method']:20s} {r['event_window_anomalies']:>10d} {r['tight_window_anomalies']:>10d} {d:>10s}")

    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    for commodity in target_series:
        comm_results = [r for r in results if r['commodity'] == commodity]
        detected = [r for r in comm_results if r['days_to_first_detection'] is not None]
        print(f"\n  {commodity}: {len(detected)}/{len(comm_results)} methods detected BBM event")
        for r in comm_results:
            status = "✅ DETECTED" if r['days_to_first_detection'] is not None else "❌ MISSED"
            print(f"    {r['method']:20s} {status} (window anomalies: {r['event_window_anomalies']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
