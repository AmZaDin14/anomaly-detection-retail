"""
Realistic anomaly injection for PIHPS food price data.

Uses Indonesian economic calendar events to inject anomalies that
mimic real-world price disruptions. Each injected anomaly is backed
by a known event (holiday, policy change, natural disaster).

Strategy:
- For each event-commodity pair, inject a price disturbance at the
  event date with realistic shape (ramp-up, peak, decay)
- Different commodities react differently to the same event
- Reserve some events as validation (not disclosed to detectors)
"""

from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from .indonesian_calendar import EconomicEvent, ALL_EVENTS


def inject_calendar_anomalies(
    series: pd.Series,
    commodity_name: str,
    anomaly_rate: float = 0.05,
    seed: int = 42,
    holdout_events: list[str] = None,
) -> tuple[pd.Series, pd.Series]:
    """Inject anomalies based on real economic calendar events.

    Parameters
    ----------
    series : pd.Series
        Clean price series (daily, DatetimeIndex)
    commodity_name : str
        Name of the commodity (for matching events)
    anomaly_rate : float
        Target fraction of points to label as anomalies
    seed : int
        Random seed (for reproducibility)
    holdout_events : list[str], optional
        Event names to reserve as validation only (not injected)

    Returns
    -------
    contaminated : pd.Series
        Series with anomalies injected
    labels : pd.Series (bool)
        Ground truth: True where anomalies were injected
    """
    rng = np.random.RandomState(seed)
    holdout_events = holdout_events or []

    contaminated = series.copy()
    labels = pd.Series(False, index=series.index)

    # 1. Collect events that fall within the series date range
    applicable = []
    for event in ALL_EVENTS:
        if event.name in holdout_events:
            continue
        event_dates = event.date_range()
        overlap = [d for d in event_dates if d in series.index.date]
        if not overlap:
            continue

        # Check commodity match
        if event.commodities[0] != "*" and not any(
            c.lower() in commodity_name.lower() for c in event.commodities
        ):
            continue

        applicable.append((event, overlap))

    if not applicable:
        # Fall back to random injection
        return _random_injection(series, anomaly_rate, seed)

    # 2. Calculate how many anomaly points we need
    n_total = len(series)
    n_anomalies_target = max(1, int(n_total * anomaly_rate))

    # 3. For each event, inject anomalous effect
    anomaly_indices = []
    for event, overlap_dates in applicable:
        for d in overlap_dates:
            if d not in series.index:
                continue
            idx = series.index[series.index.date == d]
            if len(idx) == 0:
                continue
            idx = idx[0]

            # Skip if already anomalous
            if labels.loc[idx]:
                continue

            # Inject effect based on event type
            t = (d - event.start_date).days
            # Shape: ramp-up 3 days, peak on event day, decay 3-7 days
            if t < -event.window_days:
                continue
            elif t < 0:
                # Pre-event: gradual ramp-up
                effect = event.magnitude * (t + event.window_days) / event.window_days
            elif t == 0:
                # Peak on event day
                effect = event.magnitude
            else:
                # Post-event: exponential decay over remaining window
                decay_half = max(3, event.window_days // 2)
                effect = event.magnitude * np.exp(-t / decay_half)

            if event.effect_type == "dip":
                effect = -effect

            # Add noise so it's not perfectly recognizable
            noise = rng.normal(0, 0.02 * series.std())
            multiplier = 1.0 + effect + noise / (series.loc[idx] + 1e-8)

            contaminated.loc[idx] = series.loc[idx] * max(0.1, multiplier)
            labels.loc[idx] = True
            anomaly_indices.append(idx)

    # 4. If we don't have enough anomalies, add random ones
    n_current = labels.sum()
    if n_current < n_anomalies_target:
        n_needed = n_anomalies_target - n_current
        normal_mask = ~labels
        normal_indices = series.index[normal_mask].tolist()
        if normal_indices:
            added = rng.choice(normal_indices, min(n_needed, len(normal_indices)), replace=False)
            for idx in added:
                # Small random perturbation
                perturb = series.loc[idx] * rng.uniform(0.7, 1.5)
                contaminated.loc[idx] = perturb
                labels.loc[idx] = True

    return contaminated, labels


def _random_injection(
    series: pd.Series, anomaly_rate: float, seed: int
) -> tuple[pd.Series, pd.Series]:
    """Fallback: random point anomalies."""
    rng = np.random.RandomState(seed)
    contaminated = series.copy()
    labels = pd.Series(False, index=series.index)

    n = len(series)
    n_anomalies = max(1, int(n * anomaly_rate))
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)

    for i in anomaly_idx:
        idx = series.index[i]
        contaminated.loc[idx] = series.loc[idx] * rng.uniform(1.5, 3.0)
        labels.loc[idx] = True

    return contaminated, labels


def evaluate_calendar_coverage(
    start_year: int = 2020, end_year: int = 2024
) -> dict:
    """Evaluate how many days are covered by calendar events."""
    from .indonesian_calendar import get_event_calendar

    calendar = get_event_calendar(start_year, end_year)
    total_days = (date(end_year, 12, 31) - date(start_year, 1, 1)).days + 1
    event_days = len(calendar)

    return {
        "total_days": total_days,
        "event_days": event_days,
        "coverage_ratio": event_days / total_days,
        "total_events": len(ALL_EVENTS),
    }


if __name__ == "__main__":
    # Quick test
    idx = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    test_series = pd.Series(
        np.sin(np.arange(len(idx)) * 2 * np.pi / 365) * 1000 + 5000
        + np.random.randn(len(idx)) * 50,
        index=idx,
    )

    contaminated, labels = inject_calendar_anomalies(
        test_series, "beras", anomaly_rate=0.05
    )
    print(f"Series length: {len(contaminated)}")
    print(f"Anomalies injected: {labels.sum()} ({labels.mean()*100:.1f}%)")
    print(f"Calendar coverage: {evaluate_calendar_coverage()}")

    # Show first few anomaly dates
    anomaly_dates = labels[labels].index[:5]
    for d in anomaly_dates:
        print(f"  {d.date()}: clean={test_series.loc[d]:.0f}, contam={contaminated.loc[d]:.0f}")
