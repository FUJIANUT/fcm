from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


DEFAULT_AGGREGATE_KEYS = [
    "pooled_acc",
    "pooled_nmi",
    "pooled_ari",
    "mean_client_acc",
    "worst_client_acc",
    "std_client_acc",
    "mean_client_nmi",
    "worst_client_nmi",
    "mean_client_ari",
    "worst_client_ari",
    "fpc",
    "partition_entropy",
    "xie_beni",
    "davies_bouldin",
    "min_center_distance",
    "collapse_threshold",
    "collapsed_pairs",
    "prototype_mass_entropy",
    "effective_prototypes",
    "underused_prototypes",
    "mean_client_fpc",
    "worst_client_fpc",
    "mean_client_partition_entropy",
    "worst_client_partition_entropy",
    "mean_client_xie_beni",
    "worst_client_xie_beni",
    "mean_client_davies_bouldin",
    "worst_client_davies_bouldin",
    "mean_client_min_center_distance",
    "worst_client_min_center_distance",
    "mean_client_collapse_threshold",
    "worst_client_collapse_threshold",
    "mean_client_collapsed_pairs",
    "worst_client_collapsed_pairs",
    "mean_client_prototype_mass_entropy",
    "worst_client_prototype_mass_entropy",
    "mean_client_effective_prototypes",
    "worst_client_effective_prototypes",
    "mean_client_underused_prototypes",
    "worst_client_underused_prototypes",
    "mean_relevance",
    "std_relevance",
    "min_observed_relevance",
    "max_observed_relevance",
    "low_relevance_share",
    "objective_last",
    "rounds",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...] = ("scenario", "method"),
    value_keys: list[str] | None = None,
) -> list[dict[str, object]]:
    if value_keys is None:
        value_keys = DEFAULT_AGGREGATE_KEYS

    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[group_key]) for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in sorted(groups.items()):
        out: dict[str, object] = {group_key: value for group_key, value in zip(group_keys, key, strict=True)}
        out["n_runs"] = len(group)
        for value_key in value_keys:
            values = np.array([float(row.get(value_key, math.nan)) for row in group], dtype=float)
            if np.all(np.isnan(values)):
                out[f"{value_key}_mean"] = math.nan
                out[f"{value_key}_std"] = math.nan
            else:
                out[f"{value_key}_mean"] = float(np.nanmean(values))
                out[f"{value_key}_std"] = float(np.nanstd(values))
        summary.append(out)
    return summary
