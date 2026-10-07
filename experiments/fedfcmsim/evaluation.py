from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import davies_bouldin_score

from .fcm import fcm, initialize_centers_from_data, predict_membership, squared_distances
from .metrics import evaluate_labels, summarize_client_metrics
from .synthetic import ClientData

EPS = 1e-12


def labels_from_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmax(predict_membership(x, centers), axis=1)


def fuzzy_validity(x: np.ndarray, centers: np.ndarray, membership: np.ndarray, m: float = 2.0) -> dict[str, float]:
    n_samples = len(x)
    distances = squared_distances(x, centers)
    numerator = float(np.sum((membership**m) * distances))

    if len(centers) > 1:
        center_d2 = squared_distances(centers, centers)
        center_d2[center_d2 <= EPS] = np.inf
        min_center_d2 = float(np.min(center_d2))
    else:
        min_center_d2 = math.nan

    labels = np.argmax(membership, axis=1)
    if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(x):
        dbi = float(davies_bouldin_score(x, labels))
    else:
        dbi = math.nan

    partition_entropy = -float(np.sum(membership * np.log(np.maximum(membership, EPS))) / n_samples)
    return {
        "fpc": float(np.sum(membership * membership) / n_samples),
        "partition_entropy": partition_entropy,
        "xie_beni": numerator / (n_samples * min_center_d2) if min_center_d2 > EPS else math.nan,
        "davies_bouldin": dbi,
    }


def prototype_diagnostics(
    x: np.ndarray,
    centers: np.ndarray,
    membership: np.ndarray,
    m: float = 2.0,
    collapse_threshold_ratio: float = 0.03,
    underused_mass_threshold: float = 0.01,
) -> dict[str, float]:
    if len(centers) <= 1:
        return {
            "min_center_distance": math.nan,
            "collapse_threshold": math.nan,
            "collapsed_pairs": 0.0,
            "prototype_mass_entropy": math.nan,
            "effective_prototypes": math.nan,
            "underused_prototypes": math.nan,
        }

    center_dist = np.sqrt(np.maximum(squared_distances(centers, centers), 0.0))
    upper = center_dist[np.triu_indices(len(centers), k=1)]
    min_center_distance = float(np.min(upper))

    centered = x - np.mean(x, axis=0, keepdims=True)
    data_scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    collapse_threshold = max(1e-8, collapse_threshold_ratio * data_scale)
    collapsed_pairs = float(np.sum(upper < collapse_threshold))

    masses = np.sum(membership**m, axis=0)
    mass_total = float(np.sum(masses))
    if mass_total <= EPS:
        mass_entropy = math.nan
        effective = math.nan
        underused = math.nan
    else:
        shares = masses / mass_total
        raw_entropy = -float(np.sum(shares * np.log(np.maximum(shares, EPS))))
        mass_entropy = raw_entropy / math.log(len(centers)) if len(centers) > 1 else math.nan
        effective = float(np.exp(raw_entropy))
        underused = float(np.sum(shares < underused_mass_threshold))

    return {
        "min_center_distance": min_center_distance,
        "collapse_threshold": collapse_threshold,
        "collapsed_pairs": collapsed_pairs,
        "prototype_mass_entropy": mass_entropy,
        "effective_prototypes": effective,
        "underused_prototypes": underused,
    }


def summarize_validity(values: list[dict[str, float]]) -> dict[str, float]:
    keys = values[0].keys() if values else []
    out: dict[str, float] = {}
    lower_is_worse = {
        "fpc",
        "min_center_distance",
        "prototype_mass_entropy",
        "effective_prototypes",
    }
    higher_is_worse = {
        "xie_beni",
        "davies_bouldin",
        "partition_entropy",
        "collapsed_pairs",
        "underused_prototypes",
    }
    for key in keys:
        arr = np.array([entry[key] for entry in values], dtype=float)
        out[f"mean_client_{key}"] = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else math.nan
        if np.all(np.isnan(arr)):
            out[f"worst_client_{key}"] = math.nan
        elif key in higher_is_worse:
            out[f"worst_client_{key}"] = float(np.nanmax(arr))
        elif key in lower_is_worse:
            out[f"worst_client_{key}"] = float(np.nanmin(arr))
        else:
            out[f"worst_client_{key}"] = float(np.nanmax(arr))
    return out


def evaluate_global_method(clients: list[ClientData], centers: np.ndarray, m: float = 2.0) -> dict[str, float]:
    x_all = np.vstack([client.x for client in clients])
    y_all = np.concatenate([client.y for client in clients])
    pooled_membership = predict_membership(x_all, centers, m=m)
    pooled = evaluate_labels(y_all, np.argmax(pooled_membership, axis=1))

    client_metrics = []
    client_validity = []
    for client in clients:
        membership = predict_membership(client.x, centers, m=m)
        client_metrics.append(evaluate_labels(client.y, np.argmax(membership, axis=1)))
        validity = fuzzy_validity(client.x, centers, membership, m=m)
        validity.update(prototype_diagnostics(client.x, centers, membership, m=m))
        client_validity.append(validity)

    out = {
        "pooled_acc": pooled.acc,
        "pooled_nmi": pooled.nmi,
        "pooled_ari": pooled.ari,
    }
    pooled_validity = fuzzy_validity(x_all, centers, pooled_membership, m=m)
    pooled_validity.update(prototype_diagnostics(x_all, centers, pooled_membership, m=m))
    out.update(pooled_validity)
    out.update(summarize_client_metrics(client_metrics))
    out.update(summarize_validity(client_validity))
    return out


def evaluate_personalized_method(
    clients: list[ClientData],
    centers_by_client: list[np.ndarray],
    m: float = 2.0,
) -> dict[str, float]:
    y_all = []
    pred_all = []
    client_metrics = []
    client_validity = []
    for client, centers in zip(clients, centers_by_client, strict=True):
        membership = predict_membership(client.x, centers, m=m)
        pred = np.argmax(membership, axis=1)
        y_all.append(client.y)
        pred_all.append(pred)
        client_metrics.append(evaluate_labels(client.y, pred))
        validity = fuzzy_validity(client.x, centers, membership, m=m)
        validity.update(prototype_diagnostics(client.x, centers, membership, m=m))
        client_validity.append(validity)

    pooled = evaluate_labels(np.concatenate(y_all), np.concatenate(pred_all))
    out = {
        "pooled_acc": pooled.acc,
        "pooled_nmi": pooled.nmi,
        "pooled_ari": pooled.ari,
        "fpc": math.nan,
        "partition_entropy": math.nan,
        "xie_beni": math.nan,
        "davies_bouldin": math.nan,
        "min_center_distance": math.nan,
        "collapse_threshold": math.nan,
        "collapsed_pairs": math.nan,
        "prototype_mass_entropy": math.nan,
        "effective_prototypes": math.nan,
        "underused_prototypes": math.nan,
    }
    out.update(summarize_client_metrics(client_metrics))
    out.update(summarize_validity(client_validity))
    return out


def evaluate_local_fcm(clients: list[ClientData], n_clusters: int, seed: int, m: float) -> dict[str, float]:
    client_metrics = []
    client_validity = []
    for p, client in enumerate(clients):
        local_init = initialize_centers_from_data(client.x, n_clusters, random_state=seed + 1000 + p)
        local = fcm(client.x, n_clusters, init_centers=local_init, m=m, max_iter=150)
        pred = np.argmax(local.membership, axis=1)
        client_metrics.append(evaluate_labels(client.y, pred))
        validity = fuzzy_validity(client.x, local.centers, local.membership, m=m)
        validity.update(prototype_diagnostics(client.x, local.centers, local.membership, m=m))
        client_validity.append(validity)
    out = {
        "pooled_acc": math.nan,
        "pooled_nmi": math.nan,
        "pooled_ari": math.nan,
        "fpc": math.nan,
        "partition_entropy": math.nan,
        "xie_beni": math.nan,
        "davies_bouldin": math.nan,
        "min_center_distance": math.nan,
        "collapse_threshold": math.nan,
        "collapsed_pairs": math.nan,
        "prototype_mass_entropy": math.nan,
        "effective_prototypes": math.nan,
        "underused_prototypes": math.nan,
    }
    out.update(summarize_client_metrics(client_metrics))
    out.update(summarize_validity(client_validity))
    return out
