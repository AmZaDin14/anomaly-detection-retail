"""
Evaluation metrics for anomaly detection.

Computes: precision, recall, F1, AUC-ROC, AUC-PR
Supports per-series and aggregated results.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_detection(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    anomaly_scores: pd.Series,
) -> dict[str, float]:
    """
    Compute anomaly detection metrics.

    Parameters
    ----------
    true_labels : pd.Series (bool)
        Ground truth (True = anomaly)
    predicted_labels : pd.Series (bool)
        Binary predictions
    anomaly_scores : pd.Series (float)
        Raw anomaly scores (for AUC)
    """
    y_true = true_labels.values.astype(int)
    y_pred = predicted_labels.values.astype(int)
    y_score = anomaly_scores.values

    # Handle edge cases
    if y_true.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0, "auc_pr": 0.0}

    p = precision_score(y_true, y_pred, zero_division=0.0)
    r = recall_score(y_true, y_pred, zero_division=0.0)
    f = f1_score(y_true, y_pred, zero_division=0.0)

    # AUC requires both classes present; handle edge cases
    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_roc = 0.0

    try:
        auc_pr = average_precision_score(y_true, y_score)
    except ValueError:
        auc_pr = 0.0

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
    }


def evaluate_all_series(
    results: list[dict],
) -> pd.DataFrame:
    """
    Aggregate evaluation results across all series and configs.

    Parameters
    ----------
    results : list[dict]
        Each entry has:
            dataset, series_id, method, anomaly_config
            precision, recall, f1, auc_roc, auc_pr

    Returns
    -------
    pd.DataFrame with summary statistics per method × anomaly_rate
    """
    df = pd.DataFrame(results)
    summary = (
        df.groupby(["dataset", "method", "anomaly_rate"])
        [["precision", "recall", "f1", "auc_roc", "auc_pr"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    return summary
