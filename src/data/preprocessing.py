"""
Data preprocessing: clean, resample to daily, save as parquet.
"""

from pathlib import Path
import pandas as pd
import logging

from .download_datasets import (
    download_m5,
    download_indonesian_food_prices,
    download_online_retail,
    PROJECT_DIR,
)

log = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def preprocess_all():
    """Run preprocessing for all datasets and save parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "m5": download_m5,
        "food_prices": download_indonesian_food_prices,
        "online_retail": download_online_retail,
    }

    for name, downloader in datasets.items():
        log.info("Preprocessing %s...", name)
        try:
            df = downloader()
            out_path = PROCESSED_DIR / f"{name}_daily.parquet"
            df.to_parquet(out_path)
            n_series = df["id"].nunique()
            log.info(
                "✓ %s: %d rows, %d series, %s – %s",
                out_path.name, len(df), n_series,
                df["date"].min().date(), df["date"].max().date(),
            )
        except Exception as e:
            log.error("✗ Failed to preprocess %s: %s", name, e)


if __name__ == "__main__":
    preprocess_all()
