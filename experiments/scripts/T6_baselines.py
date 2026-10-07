"""T6: external baselines for federated fuzzy clustering.

Runs three external methods implemented in fedfcmsim/federated.py:

  stallmann_avg2   Stallmann & Wilbik FFCM avg2 (arXiv:2201.07316, Eq. 9):
                   clients run local FCM from V; server re-clusters the pooled
                   P*c local centres with k-means (k=c).
  fednova          FedNova-style normalized prototype updates (Wang et al.,
                   NeurIPS 2020), equal tau_p => FedAvg on local centres.
  scffcm           SC-FFCM (Zhang et al., IEEE TFS 33(9):3168-3181, 2025):
                   gradient FCM + SCAFFOLD Option II control variates,
                   equal-weight 1/P aggregation.

alongside four T3 design-space reference cells (identical code path -- the
cell runner is imported from T3_design_space.py):

  post_none_off       F-FCM baseline (post-adaptation mass, no gate)
  pre_none_off        aligned (pre-adaptation mass, no gate)
  post_footprint_on   published GF-PFedFCM
  pre_footprint_off   T3's recommended cheap cell

plus a `centralized_fcm` row (pooled-data FCM, same init) for context.

Protocol replicates T2/T3 exactly: SEED_OFFSET=1201, partition
random_state = SEED_OFFSET+seed, init random_state = SEED_OFFSET+seed+77,
support_skew partition on real data, m=2.0, local_steps=5, 50 rounds.
Synthetic scenarios use make_synthetic_clients (12 clients x 220, 4 clusters).

IMPORTANT: centralized FCM with m=2 collapses to a single prototype on
digits_pca16, digits_pca32, letter and mnist784_pca32 (high-dimensional
distance concentration -- NOT a federated effect). Those four configs are
invalid for comparing aggregation rules at m=2; the default dataset list is
the 9 valid ones (6 synthetic + wine, pendigits, satimage). The invalid ones
remain selectable via --datasets.

Frozen hyperparameters (calibrated on `iid` synthetic ONLY, 2 seeds, by
scripts/T6_calibrate.py; never tuned on support-skew scenarios):
  scffcm:  eta_l=0.2, eta_g=2.0   (paper argparse eta_g=0.5 diverged here)
  fednova: server_lr=1.0          (all grid values tied at ACC=1.0; canonical)

Usage:
  python3 scripts/T6_baselines.py --datasets cluster_skew_hard,wine --seeds 2
  python3 scripts/T6_baselines.py                      # 9 valid datasets, 10 seeds
  python3 scripts/T6_baselines.py --methods scffcm,fednova
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedfcmsim.datasets import make_feature_clients
from fedfcmsim.fcm import fcm, initialize_centers_from_data
from fedfcmsim.federated import fednova_fcm, scffcm, stallmann_avg2_fcm
from fedfcmsim.synthetic import BASE_CENTERS, make_synthetic_clients
from T3_design_space import client_accs, min_dist, run_cell

M = 2.0
ROUNDS = 50
LOCAL_STEPS = 5
SEED_OFFSET = 1201        # same as the published r1 benchmark configs / T2 / T3

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "t6_baselines"

# Frozen by scripts/T6_calibrate.py on the `iid` synthetic scenario (2 seeds).
SCFFCM_ETA_L = 0.2
SCFFCM_ETA_G = 2.0
FEDNOVA_SERVER_LR = 1.0

# ---------------------------------------------------------------------------
# Datasets. VALID = 6 synthetic scenarios + wine/pendigits/satimage.
# INVALID_AT_M2 = datasets where centralized FCM collapses to one prototype at
# m=2 (distance concentration): excluded from the default list only.
# ---------------------------------------------------------------------------
SYNTHETIC_SCENARIOS = [
    "cluster_skew_hard",
    "cluster_skew_overlap",
    "dirichlet_0.03",
    "dirichlet_0.1",
    "overlap_noise",
    "quantity_skew_extreme",
]

# (label, dataset, n_clients, labels_per_client, min_client_size) -- T2 verbatim
VALID_REAL_DATASETS = [
    ("wine",      "wine",      10, 2, 5),
    ("pendigits", "pendigits", 10, 3, 5),
    ("satimage",  "satimage",  10, 3, 5),
]
INVALID_REAL_DATASETS = [
    ("digits_pca16",         "digits_pca16",    10, 2, 5),
    ("digits_pca32",         "digits_pca32",    10, 2, 5),
    ("letter",               "letter",          10, 6, 50),
    ("mnist784_pca32 (20c)", "mnist784_pca32",  20, 3, 50),
    ("mnist784_pca32 (50c)", "mnist784_pca32",  50, 3, 20),
]

T3_CELLS = [
    ("post_none_off",     ("post", "none", "off")),
    ("pre_none_off",      ("pre",  "none", "off")),
    ("post_footprint_on", ("post", "footprint", "on")),
    ("pre_footprint_off", ("pre",  "footprint", "off")),
]
METHODS = [name for name, _ in T3_CELLS] + [
    "stallmann_avg2", "fednova", "scffcm", "centralized_fcm",
]


def build_clients(spec: tuple, seed: int):
    """Return (clients, n_clusters) for one (dataset-config, seed)."""
    kind = spec[0]
    if kind == "synthetic":
        _, label = spec
        clients = make_synthetic_clients(
            label, n_clients=12, samples_per_client=220,
            random_state=SEED_OFFSET + seed,
        )
        return clients, BASE_CENTERS.shape[0]
    _, label, dataset, n_clients, lpc, min_size = spec
    return make_feature_clients(
        dataset=dataset,
        n_clients=n_clients,
        partition="support_skew",
        labels_per_client=lpc,
        min_size=min_size,
        random_state=SEED_OFFSET + seed,
    )


def run_one_seed(spec: tuple, seed: int, methods: list[str]) -> list[dict]:
    """All requested methods for one (dataset-config, seed)."""
    global M
    import T3_design_space as _t3
    M = _t3.fuzzifier_for(spec[1])
    _t3.M = M  # run_cell reads T3's module-level fuzzifier
    clients, n_clusters = build_clients(spec, seed)
    pooled = np.vstack([c.x for c in clients])
    init = initialize_centers_from_data(pooled, n_clusters,
                                        random_state=SEED_OFFSET + seed + 77)
    n_clients = len(clients)
    cell_map = dict(T3_CELLS)

    rows = []
    for method in methods:
        if method in cell_map:
            V, eval_centers, _ = run_cell(clients, init, *cell_map[method])
        elif method == "stallmann_avg2":
            res = stallmann_avg2_fcm(clients, init, rounds=ROUNDS,
                                     local_steps=LOCAL_STEPS, m=M,
                                     seed=SEED_OFFSET + seed)
            V, eval_centers = res.centers, [res.centers] * n_clients
        elif method == "fednova":
            res = fednova_fcm(clients, init, rounds=ROUNDS,
                              local_steps=LOCAL_STEPS, m=M,
                              server_lr=FEDNOVA_SERVER_LR)
            V, eval_centers = res.centers, [res.centers] * n_clients
        elif method == "scffcm":
            res = scffcm(clients, init, rounds=ROUNDS, local_steps=LOCAL_STEPS,
                         m=M, eta_l=SCFFCM_ETA_L, eta_g=SCFFCM_ETA_G,
                         seed=SEED_OFFSET + seed)
            V, eval_centers = res.centers, [res.centers] * n_clients
        elif method == "centralized_fcm":
            res = fcm(pooled, n_clusters, init_centers=init, m=M)
            V, eval_centers = res.centers, [res.centers] * n_clients
        else:
            raise ValueError(f"unknown method {method}")

        mean_acc, worst_acc = client_accs(clients, eval_centers)
        rows.append({
            "method": method,
            "min_dist": min_dist(V),
            "mean_acc": mean_acc,
            "worst_acc": worst_acc,
        })
    return rows


def _worker(task):
    spec, seed, methods = task
    t0 = time.time()
    rows = run_one_seed(spec, seed, methods)
    return spec[1], seed, rows, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of dataset labels "
                         "(default: the 9 m=2-valid ones)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", default=None,
                    help=f"comma-separated subset of {METHODS}; default all")
    ap.add_argument("--tag", default="", help="extra suffix on output files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = OUT_DIR / f"per_seed_metrics{suffix}.csv"
    md_path = OUT_DIR / f"summary{suffix}.md"

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else list(METHODS)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; known: {METHODS}")

    valid_specs = [("synthetic", s) for s in SYNTHETIC_SCENARIOS]
    valid_specs += [("real",) + tuple(d) for d in VALID_REAL_DATASETS]
    all_specs = valid_specs + [("real",) + tuple(d) for d in INVALID_REAL_DATASETS]
    wanted = set(args.datasets.split(",")) if args.datasets else None
    configs = [s for s in all_specs if wanted is None or s[1] in wanted]
    if not configs:
        raise SystemExit(f"no datasets matched {wanted}; known: {[s[1] for s in all_specs]}")
    invalid_used = [s[1] for s in configs if s not in valid_specs]
    if invalid_used:
        print(f"WARNING: {invalid_used} collapse under centralized FCM at m=2 "
              f"(distance concentration); aggregation-rule comparison is invalid.",
              flush=True)

    tasks = [(spec, seed, methods) for spec in configs for seed in range(args.seeds)]

    all_rows: list[dict] = []
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            label, seed, rows, elapsed = fut.result()
            for r in rows:
                r.update({"dataset": label, "seed": seed})
            all_rows.extend(rows)
            done += 1
            print(f"[{done}/{len(tasks)}] {label} seed={seed} done in {elapsed:.1f}s "
                  f"({time.time()-t_start:.0f}s total)", flush=True)

    method_order = {name: i for i, name in enumerate(METHODS)}
    fieldnames = ["dataset", "seed", "method", "min_dist", "mean_acc", "worst_acc"]
    all_rows.sort(key=lambda r: (r["dataset"], r["seed"], method_order[r["method"]]))
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows([{k: r[k] for k in fieldnames} for r in all_rows])
    print(f"appended {len(all_rows)} rows to {csv_path}")

    by_ds: dict[str, list[dict]] = {}
    for r in all_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    lines = [f"# T6 external baselines — {args.seeds} seed(s), tag='{args.tag}'\n",
             f"frozen: scffcm eta_l={SCFFCM_ETA_L} eta_g={SCFFCM_ETA_G}; "
             f"fednova server_lr={FEDNOVA_SERVER_LR}\n"]
    for ds, rows in by_ds.items():
        lines.append(f"\n## {ds}\n")
        lines.append("| method | mean_acc | worst_acc | min_dist |")
        lines.append("|---|---|---|---|")
        for method in method_order:
            sub = [r for r in rows if r["method"] == method]
            if not sub:
                continue
            lines.append(
                f"| {method} | {np.mean([r['mean_acc'] for r in sub]):.4f} | "
                f"{np.mean([r['worst_acc'] for r in sub]):.4f} | "
                f"{np.mean([r['min_dist'] for r in sub]):.4f} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
