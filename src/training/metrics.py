"""
Evaluation metrics for imbalanced binary classification.

Implements:
    - AUC-PR (Area Under Precision-Recall Curve) — the primary metric for
      imbalanced detection (Saito & Rehmsmeier, 2015).
    - F1 score on the minority (illicit) class at the best threshold.
    - Recall@k — recall when only the top-k highest-scoring predictions are
      reviewed (analyst-budget metric).
    - AUC-ROC — included for comparability with prior literature.
"""
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@torch.no_grad()
def get_probabilities(logits: torch.Tensor) -> np.ndarray:
    """Return probabilities for the positive (illicit) class."""
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return probs.cpu().numpy()


def compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_prob))


def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def compute_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """F1 at the threshold that maximises F1 on the given data.

    Returns
    -------
    best_f1 : float
    best_threshold : float
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 at every operating point
    f1_vals = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_vals))
    best_f1 = float(f1_vals[best_idx])
    # `thresholds` has length n-1 because precision/recall have a trailing point.
    best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
    return best_f1, best_thresh


def compute_f1_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> float:
    """F1 at a specified probability threshold (for evaluation on test using
    the threshold chosen on validation)."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_recall_at_k(
    y_true: np.ndarray, y_prob: np.ndarray, k: int
) -> float:
    """Recall when the top-k predictions are reviewed.

    Simulates the analyst-budget scenario in which only the highest-scoring
    transactions can be reviewed by human compliance staff.
    """
    if k >= len(y_prob):
        k = len(y_prob)
    # Top-k indices by probability (highest first)
    top_k_idx = np.argsort(-y_prob)[:k]
    true_positives_in_top_k = y_true[top_k_idx].sum()
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    return float(true_positives_in_top_k / total_positives)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    k_values: List[int] = (50, 100, 200),
) -> Dict[str, float]:
    """
    Compute the full metric panel for a single evaluation pass.

    If `threshold` is None, the threshold that maximises F1 on this set is
    found and reported (this is appropriate for validation but NOT for test —
    on test, pass the validation-selected threshold to avoid leakage).
    """
    metrics: Dict[str, float] = {
        "auc_pr": compute_auc_pr(y_true, y_prob),
        "auc_roc": compute_auc_roc(y_true, y_prob),
    }

    if threshold is None:
        best_f1, best_thresh = compute_best_f1(y_true, y_prob)
        metrics["f1_best"] = best_f1
        metrics["threshold_best"] = best_thresh
    else:
        metrics["f1_at_threshold"] = compute_f1_at_threshold(
            y_true, y_prob, threshold
        )
        metrics["threshold"] = float(threshold)

    for k in k_values:
        metrics[f"recall_at_{k}"] = compute_recall_at_k(y_true, y_prob, k)

    metrics["n_samples"] = int(len(y_true))
    metrics["n_positives"] = int(y_true.sum())
    return metrics
