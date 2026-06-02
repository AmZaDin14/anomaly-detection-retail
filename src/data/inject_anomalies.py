"""
Controlled anomaly injection into retail time series data.

Three anomaly types (standard in the anomaly detection literature):
1. Point (global) — single extreme value
2. Contextual (local) — value extreme relative to its temporal context
3. Collective (pattern) — a subsequence that deviates from expected pattern

Injection is done at controlled rates (1%, 5%, 10%) with ground-truth labels.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "data" / "results"


def inject_point_anomalies(
    series: pd.Series,
    anomaly_rate: float = 0.05,
    magnitude: float = 3.0,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Inject point anomalies by replacing random values with outliers.

    Parameters
    ----------
    series : pd.Series
        Daily sales with datetime index
    anomaly_rate : float
        Fraction of points to turn into anomalies (0.01 = 1%)
    magnitude : float
        Multiplier for the anomaly value relative to local std
    random_state : int

    Returns
    -------
    contaminated : pd.Series
        Series with injected anomalies
    labels : pd.Series (bool)
        True where anomaly was injected, same index as series
    """
    rng = np.random.default_rng(random_state)
    n = len(series)
    n_anom = max(1, int(n * anomaly_rate))

    indices = rng.choice(n, size=n_anom, replace=False)
    labels = pd.Series(False, index=series.index)
    contaminated = series.copy().astype(float)

    # For each chosen point, replace with a value far from expected
    for idx in indices:
        # Use local statistics (rolling window of 14 days)
        window_start = max(0, idx - 14)
        window_end = min(n, idx + 14)
        local_vals = series.iloc[window_start:window_end].dropna()
        if len(local_vals) < 3:
            local_vals = series

        local_std = local_vals.std()
        local_mean = local_vals.mean()

        if local_std == 0:
            local_std = series.std() or 1.0

        # Inject upward or downward anomaly
        direction = 1 if rng.random() > 0.3 else -1  # 70% upward
        anomaly_value = local_mean + direction * magnitude * local_std

        contaminated.iloc[idx] = max(0, anomaly_value)  # sales can't be negative
        labels.iloc[idx] = True

    return contaminated, labels


def inject_contextual_anomalies(
    series: pd.Series,
    anomaly_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Inject contextual anomalies: values that are normal globally but
    unusual for their day-of-week or seasonal context.

    For example: a normal sales value on a Monday, but injected on a Sunday
    where sales are typically low.
    """
    rng = np.random.default_rng(random_state)
    n = len(series)
    n_anom = max(1, int(n * anomaly_rate))

    # Compute day-of-week profile as dict for robust indexing
    dow_avgs = series.groupby(series.index.dayofweek).mean().to_dict()

    indices = rng.choice(n, size=n_anom, replace=False)
    labels = pd.Series(False, index=series.index)
    contaminated = series.copy().astype(float)

    for idx in indices:
        ts = series.index[idx]
        actual_dow = int(ts.dayofweek)
        # Pick a different day-of-week to swap context
        other_dow = (actual_dow + int(rng.integers(1, 7))) % 7
        avg_actual = dow_avgs.get(actual_dow, series.mean())
        avg_other = dow_avgs.get(other_dow, series.mean())
        ratio = avg_other / avg_actual if avg_actual > 0 else 1.0
        contaminated.iloc[idx] = series.iloc[idx] * max(0.1, ratio)
        labels.iloc[idx] = True

    return contaminated, labels


def inject_collective_anomalies(
    series: pd.Series,
    anomaly_rate: float = 0.05,
    block_size: int = 7,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Inject collective (subsequence) anomalies by replacing a contiguous block
    with a shifted/compressed version of itself.

    For example: a week of flat, low sales where there should be seasonal variation.
    """
    rng = np.random.default_rng(random_state)
    n = len(series)
    labels = pd.Series(False, index=series.index)
    contaminated = series.copy().astype(float)

    # Number of blocks to inject
    n_blocks = max(1, int(n * anomaly_rate / block_size))

    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n - block_size))
        end = min(start + block_size, n)

        block = series.iloc[start:end].copy()
        if len(block) < 3:
            continue

        # Replace block with its mean (flat line) or a scaled version
        block_mean = block.mean()
        noise = rng.normal(0, block.std() * 0.1, len(block))
        contaminated.iloc[start:end] = max(0, block_mean + noise.mean())
        labels.iloc[start:end] = True

    return contaminated, labels


def inject_mixed_anomalies(
    series: pd.Series,
    total_rate: float = 0.05,
    point_frac: float = 0.4,
    contextual_frac: float = 0.3,
    collective_frac: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Inject a mixture of all three anomaly types at a total rate.
    """
    rng = np.random.default_rng(random_state)
    # Fraction of total points to allocate to each type
    total_points = len(series)
    n_point = int(total_points * total_rate * point_frac)
    n_contextual = int(total_points * total_rate * contextual_frac)
    n_collective = int(total_points * total_rate * collective_frac)

    labels = pd.Series(False, index=series.index)
    contaminated = series.copy().astype(float)

    methods = [
        (inject_point_anomalies, n_point / total_points),
        (inject_contextual_anomalies, n_contextual / total_points),
        (inject_collective_anomalies, n_collective / total_points),
    ]

    current_rs = random_state
    for method, rate in methods:
        if rate <= 0:
            continue
        contaminated, lbl = method(
            contaminated, anomaly_rate=rate, random_state=current_rs
        )
        labels = labels | lbl
        current_rs += 1

    return contaminated, labels


def prepare_anomaly_dataset(
    clean_series: pd.Series,
    name: str,
    anomaly_configs: Optional[list[dict]] = None,
    results_dir: Optional[Path] = None,
) -> dict:
    """
    Prepare a full anomaly detection dataset with multiple injection rates.

    Parameters
    ----------
    clean_series : pd.Series
        Clean daily sales with datetime index
    name : str
        Identifier for the series
    anomaly_configs : list[dict], optional
        Each dict: {'rate': 0.05, 'type': 'mixed'} or 'point'/'contextual'/'collective'
    results_dir : Path, optional

    Returns
    -------
    dict with keys: 'clean', 'contaminated', 'labels', 'config'
    """
    if anomaly_configs is None:
        anomaly_configs = [
            {"rate": 0.01, "type": "mixed"},
            {"rate": 0.05, "type": "mixed"},
            {"rate": 0.10, "type": "mixed"},
        ]

    result = {
        "name": name,
        "clean": clean_series,
        "configs": [],
    }

    for cfg in anomaly_configs:
        if cfg["type"] == "point":
            contaminated, labels = inject_point_anomalies(
                clean_series, anomaly_rate=cfg["rate"]
            )
        elif cfg["type"] == "contextual":
            contaminated, labels = inject_contextual_anomalies(
                clean_series, anomaly_rate=cfg["rate"]
            )
        elif cfg["type"] == "collective":
            contaminated, labels = inject_collective_anomalies(
                clean_series, anomaly_rate=cfg["rate"]
            )
        else:
            contaminated, labels = inject_mixed_anomalies(
                clean_series, total_rate=cfg["rate"]
            )

        result["configs"].append({
            "cfg": cfg,
            "contaminated": contaminated,
            "labels": labels,
            "n_anomalies": labels.sum(),
        })

    return result


def prepare_all_datasets(
    series_dict: dict[str, pd.Series],
    anomaly_configs: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Apply anomaly injection to all series in a dictionary.

    Parameters
    ----------
    series_dict : dict[str, pd.Series]
        Dictionary from load_processed_dataset
    anomaly_configs : list[dict], optional

    Returns
    -------
    list[dict]
        Each element is the output of prepare_anomaly_dataset
    """
    datasets = []
    for sid, series in series_dict.items():
        log.info("Preparing anomalies for series %s (%d points)", sid, len(series))
        ds = prepare_anomaly_dataset(series, sid, anomaly_configs)
        datasets.append(ds)
    return datasets


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(42)
    clean = pd.Series(
        rng.normal(100, 20, 365),
        index=pd.date_range("2023-01-01", periods=365, freq="D"),
    )
    result = prepare_anomaly_dataset(clean, "test")
    for cfg in result["configs"]:
        log.info(
            "Rate=%.2f type=%-12s anomalies=%d/%d (%.1f%%)",
            cfg["cfg"]["rate"], cfg["cfg"]["type"],
            cfg["labels"].sum(), len(clean),
            100 * cfg["labels"].sum() / len(clean),
        )
