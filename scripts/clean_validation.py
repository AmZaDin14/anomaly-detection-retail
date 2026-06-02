#!/usr/bin/env python3
"""
Clean PIHPS validation: run detectors on REAL data near known economic events.

No anomalies are injected. We check if detectors flag real price disturbances
that coincide with known events (Idul Fitri, BBM hike, El Nino, etc.).

Metrics:
- Detection rate: % of event-commodity pairs where detector flags anomalies
- Detection density: % of points flagged within event window
- Background rate: % of points flagged in quiet period (no known events)
- Lift: density / background rate
- Detection delay: days from event start to first anomaly
"""

import pickle, sys, time, signal
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


# =====================================================
# KNOWN ECONOMIC EVENTS (independent of injection)
# =====================================================
EVENTS = [
    # (name, start, end, commodities, description)
    ("Idul Fitri 2022", date(2022, 4, 22), date(2022, 5, 8),
     ["beras", "gula_pasir", "minyak_goreng", "telur_ayam", "daging_ayam", "daging_sapi"],
     "Holiday demand spike"),

    ("BBM Hike 2022", date(2022, 9, 3), date(2022, 9, 17),
     ["beras", "minyak_goreng", "cabai_rawit", "gula_pasir", "telur_ayam", "daging_ayam"],
     "Fuel price +30-40%"),

    ("Idul Fitri 2023", date(2023, 4, 14), date(2023, 4, 28),
     ["beras", "gula_pasir", "minyak_goreng", "telur_ayam", "daging_ayam", "daging_sapi"],
     "Holiday demand spike"),

    ("El Nino 2023", date(2023, 6, 1), date(2023, 9, 30),
     ["beras", "cabai_rawit", "bawang_merah", "bawang_putih"],
     "Drought → crop failure → price spike"),

    ("Minyak Goreng Crisis", date(2022, 3, 1), date(2022, 5, 31),
     ["minyak_goreng"],
     "Global CPO price + export ban"),

    ("Idul Adha 2023", date(2023, 6, 22), date(2023, 7, 6),
     ["daging_sapi", "daging_ayam"],
     "Meat demand spike"),
]

# Quiet periods (no major events) for baseline false positive rate
QUIET_PERIODS = [
    (date(2019, 3, 1), date(2019, 4, 30)),
    (date(2021, 2, 1), date(2021, 3, 15)),
    (date(2023, 10, 1), date(2023, 11, 15)),
]


def load_pihps():
    with open(DATA_DIR / "food_prices_prepared.pkl", "rb") as f:
        return pickle.load(f)


def find_series(data, name):
    matches = [d for d in data if d["name"] == name]
    return matches[0]["clean"] if matches else None


def run_detector(cls, kwargs, series):
    """Run detector, return predictions Series (bool)."""
    detector = cls(**kwargs)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(120)
    try:
        detector.fit(series)
        preds = detector.predict(series)
        signal.alarm(0)
        return preds
    except Exception:
        signal.alarm(0)
        return pd.Series(False, index=series.index)


def _timeout_handler(signum, frame):
    raise TimeoutError("Detector timed out")


def main():
    from src.methods.stl_iqr import STLIQRDetector
    from src.methods.stl_pelt import STLPELTDetector
    from src.methods.autoencoder import AutoencoderDetector

    detectors = [
        ("STL+IQR", STLIQRDetector, {"iqr_multiplier": 1.5}),
        ("STL-PELT_0.7", STLPELTDetector, {"alpha": 0.7}),
        ("Autoencoder", AutoencoderDetector, {"window": 14, "epochs": 30}),
    ]

    data = load_pihps()

    all_results = []

    # =========================================
    # EVENT WINDOW ANALYSIS
    # =========================================
    for event_name, start, end, commodities, desc in EVENTS:
        window = (end - start).days

        for comm_name in commodities:
            series = find_series(data, comm_name)
            if series is None:
                continue

            event_mask = (series.index.date >= start) & (series.index.date <= end)
            window_len = event_mask.sum()
            if window_len == 0:
                continue

            for det_name, det_cls, det_kwargs in detectors:
                preds = run_detector(det_cls, det_kwargs, series)

                # Event window stats
                event_preds = preds[event_mask]
                n_flagged = event_preds.sum()
                density = n_flagged / window_len if window_len > 0 else 0

                # Detection delay (first anomaly after event start)
                delay = None
                after_start = preds[preds.index.date >= start]
                first_anom = after_start[after_start].index.min() if after_start.any() else None
                if first_anom is not None:
                    delay = (first_anom.date() - start).days

                all_results.append({
                    "event": event_name,
                    "commodity": comm_name,
                    "method": det_name,
                    "window_days": window,
                    "window_points": window_len,
                    "flagged": int(n_flagged),
                    "density": round(density, 4),
                    "detection_delay_days": delay,
                    "event_description": desc,
                })

    # =========================================
    # QUIET PERIOD BASELINE
    # =========================================
    quiet_results = []
    for q_start, q_end in QUIET_PERIODS:
        for comm_name in ["beras", "minyak_goreng", "cabai_rawit", "gula_pasir", "telur_ayam", "daging_ayam", "daging_sapi"]:
            series = find_series(data, comm_name)
            if series is None:
                continue
            q_mask = (series.index.date >= q_start) & (series.index.date <= q_end)
            q_len = q_mask.sum()
            if q_len < 10:
                continue

            for det_name, det_cls, det_kwargs in detectors:
                preds = run_detector(det_cls, det_kwargs, series)
                q_preds = preds[q_mask]
                n_q = q_preds.sum()
                q_density = n_q / q_len if q_len > 0 else 0
                quiet_results.append({
                    "method": det_name,
                    "commodity": comm_name,
                    "quiet_period": f"{q_start} to {q_end}",
                    "points": q_len,
                    "flagged": int(n_q),
                    "background_density": round(q_density, 4),
                })

    # =========================================
    # RESULTS
    # =========================================
    print("=" * 80)
    print("CLEAN PIHPS VALIDATION: REAL EVENT DETECTION")
    print("=" * 80)

    # Per-event summary
    for event_name in sorted(set(r["event"] for r in all_results)):
        event_results = [r for r in all_results if r["event"] == event_name]
        print(f"\n--- {event_name} ---")
        print(f"  {'Method':20s} {'Commodity':20s} {'Flagged':>8s} {'Density':>8s} {'Delay(d)':>8s}")
        print("  " + "-" * 64)
        for det_name in ["STL+IQR", "STL-PELT_0.7", "Autoencoder"]:
            det_results = [r for r in event_results if r["method"] == det_name]
            for r in det_results:
                delay = str(r["detection_delay_days"]) if r["detection_delay_days"] is not None else "N/A"
                print(f"  {det_name:20s} {r['commodity']:20s} {r['flagged']:>8d} {r['density']:>8.3f} {delay:>8s}")

    # Detection rate: how many event-commodity pairs were detected?
    print("\n" + "=" * 80)
    print("DETECTION RATE BY METHOD")
    print("=" * 80)
    for det_name in ["STL+IQR", "STL-PELT_0.7", "Autoencoder"]:
        det_results = [r for r in all_results if r["method"] == det_name]
        total_pairs = len(det_results)
        detected = sum(1 for r in det_results if r["flagged"] > 0)
        avg_density = np.mean([r["density"] for r in det_results])
        print(f"  {det_name:20s}: {detected}/{total_pairs} pairs detected ({detected/total_pairs*100:.0f}%), avg density={avg_density:.3f}")

    # Average detection delay
    print("\nAVERAGE DETECTION DELAY (days from event start)")
    for det_name in ["STL+IQR", "STL-PELT_0.7", "Autoencoder"]:
        det_results = [r for r in all_results if r["method"] == det_name and r["detection_delay_days"] is not None]
        if det_results:
            avg_delay = np.mean([r["detection_delay_days"] for r in det_results])
            print(f"  {det_name:20s}: {avg_delay:.1f} days (across {len(det_results)} detections)")

    # Background false positive rate
    print("\n" + "=" * 80)
    print("QUIET PERIOD BASELINE (background false positive density)")
    print("=" * 80)
    bg_df = pd.DataFrame(quiet_results)
    for det_name in ["STL+IQR", "STL-PELT_0.7", "Autoencoder"]:
        bg = bg_df[bg_df["method"] == det_name]
        print(f"  {det_name:20s}: mean={bg['background_density'].mean():.4f}, std={bg['background_density'].std():.4f}")

    # Lift: event density / background density
    print("\n" + "=" * 80)
    print("LIFT (event density / background density)")
    print("=" * 80)
    bg_means = bg_df.groupby("method")["background_density"].mean()
    for det_name in ["STL+IQR", "STL-PELT_0.7", "Autoencoder"]:
        det_results = [r for r in all_results if r["method"] == det_name]
        if det_name in bg_means.index and bg_means[det_name] > 0:
            avg_event_density = np.mean([r["density"] for r in det_results])
            lift = avg_event_density / bg_means[det_name]
            print(f"  {det_name:20s}: event density={avg_event_density:.4f}, background={bg_means[det_name]:.4f}, lift={lift:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
