from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PairedTestResult:
    metric: str
    target: str
    baseline: str
    n_pairs: int
    target_mean: float
    baseline_mean: float
    mean_diff: float
    median_diff: float
    p_value: float
    wins: int
    ties: int
    losses: int


def paired_wilcoxon(
    target_values: list[float],
    baseline_values: list[float],
    metric: str,
    target: str,
    baseline: str,
    larger_is_better: bool = True,
) -> PairedTestResult:
    target_arr = np.asarray(target_values, dtype=float)
    baseline_arr = np.asarray(baseline_values, dtype=float)
    valid = ~(np.isnan(target_arr) | np.isnan(baseline_arr))
    target_arr = target_arr[valid]
    baseline_arr = baseline_arr[valid]
    diff = target_arr - baseline_arr
    if not larger_is_better:
        diff = -diff

    p_value = math.nan
    if len(diff) >= 2 and np.any(np.abs(diff) > 1e-12):
        try:
            from scipy.stats import wilcoxon

            p_value = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            p_value = math.nan

    wins = int(np.sum(diff > 1e-12))
    ties = int(np.sum(np.abs(diff) <= 1e-12))
    losses = int(np.sum(diff < -1e-12))
    raw_diff = target_arr - baseline_arr
    return PairedTestResult(
        metric=metric,
        target=target,
        baseline=baseline,
        n_pairs=int(len(diff)),
        target_mean=float(np.mean(target_arr)) if len(target_arr) else math.nan,
        baseline_mean=float(np.mean(baseline_arr)) if len(baseline_arr) else math.nan,
        mean_diff=float(np.mean(raw_diff)) if len(raw_diff) else math.nan,
        median_diff=float(np.median(raw_diff)) if len(raw_diff) else math.nan,
        p_value=p_value,
        wins=wins,
        ties=ties,
        losses=losses,
    )
