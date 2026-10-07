from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientData:
    x: np.ndarray
    y: np.ndarray
    name: str


BASE_CENTERS = np.array(
    [
        [-5.0, -4.0],
        [-1.0, 5.0],
        [5.0, -3.5],
        [6.5, 5.0],
    ],
    dtype=float,
)


def _sample_client(
    rng: np.random.Generator,
    centers: np.ndarray,
    proportions: np.ndarray,
    n_samples: int,
    std: float,
    noise_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.choice(len(centers), size=n_samples, p=proportions)
    x = centers[labels] + rng.normal(0.0, std, size=(n_samples, centers.shape[1]))

    if noise_ratio > 0.0:
        n_noise = int(round(noise_ratio * n_samples))
        if n_noise > 0:
            lo = centers.min(axis=0) - 4.0
            hi = centers.max(axis=0) + 4.0
            noise_idx = rng.choice(n_samples, size=n_noise, replace=False)
            x[noise_idx] = rng.uniform(lo, hi, size=(n_noise, centers.shape[1]))
    return x, labels


def _quantity_counts(n_clients: int, base_samples: int) -> list[int]:
    multipliers = np.geomspace(0.25, 2.5, num=n_clients)
    multipliers = multipliers / multipliers.mean()
    return [max(40, int(round(base_samples * value))) for value in multipliers]


def _extreme_quantity_counts(n_clients: int, base_samples: int) -> list[int]:
    multipliers = np.geomspace(0.05, 8.0, num=n_clients)
    multipliers = multipliers / multipliers.mean()
    return [max(20, int(round(base_samples * value))) for value in multipliers]


def _dirichlet_alpha_from_scenario(scenario: str) -> float | None:
    prefix = "dirichlet_"
    if not scenario.startswith(prefix):
        return None
    return float(scenario[len(prefix) :])


def make_synthetic_clients(
    scenario: str,
    n_clients: int = 12,
    samples_per_client: int = 220,
    random_state: int | None = None,
) -> list[ClientData]:
    rng = np.random.default_rng(random_state)
    scenario = scenario.lower()
    dirichlet_alpha = _dirichlet_alpha_from_scenario(scenario)
    centers = BASE_CENTERS.copy()
    std = 0.75
    noise_ratio = 0.0

    if scenario in {"overlap_noise", "overlap_noise_hard"}:
        centers = np.array([[-3.2, -2.8], [-1.2, 2.8], [2.7, -2.2], [3.4, 2.7]], dtype=float)
        std = 1.55 if scenario == "overlap_noise_hard" else 1.25
        noise_ratio = 0.14 if scenario == "overlap_noise_hard" else 0.08

    if scenario == "cluster_skew_overlap":
        centers = np.array([[-2.6, -2.2], [-0.9, 2.3], [2.3, -1.8], [2.8, 2.3]], dtype=float)
        std = 1.45

    if scenario == "quantity_skew_extreme":
        counts = _extreme_quantity_counts(n_clients, samples_per_client)
    elif scenario == "quantity_skew":
        counts = _quantity_counts(n_clients, samples_per_client)
    else:
        counts = [samples_per_client] * n_clients

    clients: list[ClientData] = []
    for p in range(n_clients):
        if scenario == "iid":
            proportions = np.ones(len(centers)) / len(centers)
        elif dirichlet_alpha is not None:
            proportions = rng.dirichlet(np.full(len(centers), dirichlet_alpha))
        elif scenario == "cluster_skew":
            proportions = rng.dirichlet(np.full(len(centers), 0.08))
            keep = np.argsort(proportions)[-2:]
            masked = np.zeros_like(proportions)
            masked[keep] = proportions[keep]
            proportions = masked / masked.sum()
        elif scenario in {"cluster_skew_hard", "cluster_skew_overlap"}:
            dominant = p % len(centers)
            secondary = (dominant + rng.integers(1, len(centers))) % len(centers)
            proportions = np.full(len(centers), 0.0)
            proportions[dominant] = 0.92
            proportions[secondary] = 0.08
        elif scenario == "private_cluster":
            if p < max(1, n_clients // 4):
                proportions = np.array([0.25, 0.20, 0.10, 0.45])
            else:
                base = rng.dirichlet(np.array([1.2, 1.0, 1.0]))
                proportions = np.array([base[0], base[1], base[2], 0.0])
        elif scenario == "minority_private_cluster":
            if p in {0, max(1, n_clients // 2)}:
                proportions = np.array([0.06, 0.07, 0.07, 0.80])
            else:
                base = rng.dirichlet(np.array([0.9, 0.9, 0.9]))
                proportions = np.array([base[0], base[1], base[2], 0.0])
        elif scenario in {"quantity_skew", "quantity_skew_extreme"}:
            proportions = rng.dirichlet(np.full(len(centers), 0.25))
        elif scenario in {"overlap_noise", "overlap_noise_hard"}:
            proportions = rng.dirichlet(np.full(len(centers), 0.35))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        x, y = _sample_client(
            rng,
            centers=centers,
            proportions=proportions,
            n_samples=counts[p],
            std=std,
            noise_ratio=noise_ratio,
        )
        clients.append(ClientData(x=x, y=y, name=f"client_{p:02d}"))
    return clients
