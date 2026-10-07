from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class FCMResult:
    centers: np.ndarray
    membership: np.ndarray
    objective_history: list[float]
    n_iter: int


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def update_membership_from_distances(distances: np.ndarray, m: float = 2.0) -> np.ndarray:
    if m <= 1.0:
        raise ValueError("The fuzzifier m must be greater than 1.")

    distances = np.asarray(distances, dtype=float)
    zero_mask = distances <= EPS
    safe_distances = np.maximum(distances, EPS)
    inv = safe_distances ** (-1.0 / (m - 1.0))
    membership = inv / np.sum(inv, axis=1, keepdims=True)

    rows_with_zero = np.where(np.any(zero_mask, axis=1))[0]
    for row in rows_with_zero:
        z = zero_mask[row]
        membership[row, :] = z / np.sum(z)
    return membership


def predict_membership(x: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    return update_membership_from_distances(squared_distances(x, centers), m=m)


def update_centers(
    x: np.ndarray,
    membership: np.ndarray,
    m: float = 2.0,
    previous_centers: np.ndarray | None = None,
) -> np.ndarray:
    um = membership**m
    denom = np.sum(um, axis=0)
    centers = np.empty((membership.shape[1], x.shape[1]), dtype=float)
    for j in range(membership.shape[1]):
        if denom[j] <= EPS:
            if previous_centers is None:
                centers[j] = x[np.random.randint(0, len(x))]
            else:
                centers[j] = previous_centers[j]
        else:
            centers[j] = (um[:, j][:, None] * x).sum(axis=0) / denom[j]
    return centers


def fcm_objective(x: np.ndarray, centers: np.ndarray, membership: np.ndarray, m: float = 2.0) -> float:
    return float(np.sum((membership**m) * squared_distances(x, centers)))


def initialize_centers_from_data(
    x: np.ndarray,
    n_clusters: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    if len(x) < n_clusters:
        raise ValueError("Need at least as many samples as clusters.")
    idx = rng.choice(len(x), size=n_clusters, replace=False)
    return x[idx].astype(float, copy=True)


def fcm(
    x: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-5,
    random_state: int | np.random.Generator | None = None,
    init_centers: np.ndarray | None = None,
) -> FCMResult:
    x = np.asarray(x, dtype=float)
    if init_centers is None:
        centers = initialize_centers_from_data(x, n_clusters, random_state=random_state)
    else:
        centers = np.asarray(init_centers, dtype=float).copy()

    objective_history: list[float] = []
    membership = predict_membership(x, centers, m=m)
    for it in range(max_iter):
        membership = predict_membership(x, centers, m=m)
        new_centers = update_centers(x, membership, m=m, previous_centers=centers)
        membership = predict_membership(x, new_centers, m=m)
        objective_history.append(fcm_objective(x, new_centers, membership, m=m))
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift < tol:
            return FCMResult(centers=centers, membership=membership, objective_history=objective_history, n_iter=it + 1)

    return FCMResult(centers=centers, membership=membership, objective_history=objective_history, n_iter=max_iter)


def run_local_fcm_steps(
    x: np.ndarray,
    init_centers: np.ndarray,
    steps: int,
    m: float = 2.0,
) -> FCMResult:
    centers = np.asarray(init_centers, dtype=float).copy()
    objective_history: list[float] = []
    membership = predict_membership(x, centers, m=m)
    for _ in range(steps):
        membership = predict_membership(x, centers, m=m)
        centers = update_centers(x, membership, m=m, previous_centers=centers)
        membership = predict_membership(x, centers, m=m)
        objective_history.append(fcm_objective(x, centers, membership, m=m))
    return FCMResult(centers=centers, membership=membership, objective_history=objective_history, n_iter=steps)
