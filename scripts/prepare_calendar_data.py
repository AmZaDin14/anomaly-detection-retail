#!/usr/bin/env python
"""
Prepare PIHPS data with calendar-based anomaly injection.

Loads the existing food_prices prepared data, then creates a new
version where anomalies are injected based on real Indonesian
economic calendar events (holidays, policy changes, disasters).
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.realistic_injection import inject_calendar_anomalies

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"


def extract_commodity(series_name: str) -> str:
    """Extract commodity type from series name."""
    # Normalize: strip suffixes like _sedang, _kb1, _k1, _merk1
    name = series_name.lower().replace("_", " ")
    # Remove quality/size suffixes
    for suffix in ["sedang", "premium", "lokal", "curah"]:
        name = name.replace(f" {suffix}", "")
    # Remove numeric suffixes (k1, k2, kb1, km1, ks1, merk1, merk2)
    import re
    name = re.sub(r"\s+k[bms]?\d+", "", name)
    name = re.sub(r"\s+merk\d+", "", name)
    name = re.sub(r"\s+\d+", "", name)
    return name.strip()


def main():
    print("Loading food_prices prepared data...")
    with open(RESULTS_DIR / "food_prices_prepared.pkl", "rb") as f:
        datasets = pickle.load(f)

    print(f"Loaded {len(datasets)} series")

    calendar_data = []
    for ds in datasets:
        series = ds["clean"]
        name = ds["name"]
        commodity = extract_commodity(name)

        # Create calendar-injected configs at 3 rates
        new_configs = []
        for rate in [0.01, 0.05, 0.10]:
            contaminated, labels = inject_calendar_anomalies(
                series, commodity, anomaly_rate=rate, seed=42
            )
            n_anomalies = labels.sum()
            cfg = {"rate": rate, "type": "calendar"}
            new_configs.append({
                "cfg": cfg,
                "contaminated": contaminated,
                "labels": labels,
                "n_anomalies": int(n_anomalies),
            })

        calendar_data.append({
            "name": name,
            "clean": series,
            "configs": new_configs,
        })

        if len(calendar_data) <= 3 or len(calendar_data) % 5 == 0:
            print(f"  [{len(calendar_data)}/{len(datasets)}] {name} -> {commodity}")

    # Save
    out_path = RESULTS_DIR / "food_prices_calendar.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(calendar_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(calendar_data)} series to {out_path}")

    # Verify
    total_anomalies = sum(
        sum(c["n_anomalies"] for c in ds["configs"])
        for ds in calendar_data
    )
    print(f"Total anomalies injected (all configs): {total_anomalies}")

    # Summary per rate
    for rate in [0.01, 0.05, 0.10]:
        anomalies = sum(
            c["n_anomalies"]
            for ds in calendar_data
            for c in ds["configs"]
            if c["cfg"]["rate"] == rate
        )
        n_points = sum(
            len(ds["clean"]) for ds in calendar_data
        )
        print(f"  rate={rate:.0%}: {anomalies} anomalies out of {n_points} points ({anomalies/n_points*100:.2f}%)")


if __name__ == "__main__":
    main()
