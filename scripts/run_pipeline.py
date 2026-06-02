#!/usr/bin/env python
"""
Phase 1: Download, preprocess, and prepare anomaly detection datasets.

Datasets:
  - m5:          50 Walmart products, 1,969 days each
  - food_prices: 17 Indonesian commodities, ~1,000 days each
  - online_retail: 50 products, ~300 days each
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"


def phase1_prepare_data():
    log.info("=" * 60)
    log.info("PHASE 1: Data Preparation")
    log.info("=" * 60)

    from src.data.preprocessing import preprocess_all
    preprocess_all()

    from src.data.download_datasets import load_processed_dataset
    from src.data.inject_anomalies import prepare_all_datasets

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_names = ["m5", "food_prices", "online_retail"]

    for ds_name in dataset_names:
        log.info("\n--- Injecting anomalies: %s ---", ds_name)
        series = load_processed_dataset(ds_name, n_series=30)
        log.info("Loaded %d series", len(series))

        if not series:
            log.warning("No series loaded for %s, skipping", ds_name)
            continue

        datasets = prepare_all_datasets(series)

        import pickle
        out_path = RESULTS_DIR / f"{ds_name}_prepared.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(datasets, f)
        log.info("Saved %s", out_path)

    log.info("\n✅ Phase 1 complete. Ready for Phase 2 (Methods).")


if __name__ == "__main__":
    phase1_prepare_data()
