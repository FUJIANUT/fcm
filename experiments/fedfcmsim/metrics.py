from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class ClusterMetrics:
    acc: float
    nmi: float
    ari: float


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    n = max(len(true_labels), len(pred_labels))
    contingency = np.zeros((n, n), dtype=int)
    true_index = {label: i for i, label in enumerate(true_labels)}
    pred_index = {label: i for i, label in enumerate(pred_labels)}
    for true, pred in zip(y_true, y_pred, strict=False):
        contingency[true_index[true], pred_index[pred]] += 1

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-contingency)
        return float(contingency[row_ind, col_ind].sum() / len(y_true))
    except Exception:
        used_cols: set[int] = set()
        total = 0
        for row in range(contingency.shape[0]):
            col_order = np.argsort(contingency[row])[::-1]
            for col in col_order:
                if int(col) not in used_cols:
                    used_cols.add(int(col))
                    total += int(contingency[row, col])
                    break
        return float(total / len(y_true))


def evaluate_labels(y_true: np.ndarray, y_pred: np.ndarray) -> ClusterMetrics:
    return ClusterMetrics(
        acc=clustering_accuracy(y_true, y_pred),
        nmi=float(normalized_mutual_info_score(y_true, y_pred)),
        ari=float(adjusted_rand_score(y_true, y_pred)),
    )


def summarize_client_metrics(metrics: list[ClusterMetrics]) -> dict[str, float]:
    acc = np.array([m.acc for m in metrics], dtype=float)
    nmi = np.array([m.nmi for m in metrics], dtype=float)
    ari = np.array([m.ari for m in metrics], dtype=float)
    return {
        "mean_client_acc": float(acc.mean()),
        "worst_client_acc": float(acc.min()),
        "std_client_acc": float(acc.std()),
        "mean_client_nmi": float(nmi.mean()),
        "worst_client_nmi": float(nmi.min()),
        "mean_client_ari": float(ari.mean()),
        "worst_client_ari": float(ari.min()),
    }
