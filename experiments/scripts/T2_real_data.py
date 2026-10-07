"""T2: weight-measurement comparison (VERIFY4/VERIFY5 arms) on REAL data.

Four required arms, all sharing the same per-seed init (centers sampled from
pooled data via initialize_centers_from_data, exactly as runner.py does):

  (a) POST     weight = fuzzy mass from the POST-adaptation membership
               (repo F-FCM baseline; VERIFY4 weight_at="post", no gate)
  (b) PRE      weight = fuzzy mass computed at the BROADCAST prototypes
               ("aligned"; VERIFY4 weight_at="pre", no gate)
  (c) GF       GF-PFedFCM = personalized_federated_fcm with the footprint
               relevance gate (server aggregation weight = POST mass x gate,
               plus the published client-side delta personalization).
               Evaluated on the per-client personalized prototypes, exactly
               as the accepted paper's published ACC numbers were computed.
  (d) PEDRYCZ  pedrycz_gradient_fcm at alpha=0.01 (STABLE alpha; 0.1 diverges)

Supplementary arm (e) GATE_ONLY = VERIFY4's literal arm (c): POST mass x
footprint gate in the shared server update, NO personalization, evaluated on
the global prototypes. Included so the footprint gate can be compared to PRE
without the personalization confound.

Metrics per arm: mean client ACC, worst client ACC (Hungarian-matched
clustering_accuracy on nearest-prototype assignments, per client over the
labels present on that client), and min-distance between final GLOBAL
prototypes (for GF: the global track centers, not the personalized ones).

Protocol replicates run_semireal_support_benchmark.py exactly: partition
random_state = SEED_OFFSET + seed (SEED_OFFSET=1201), init centers
random_state = SEED_OFFSET + seed + 77, support_skew partition, m=2.0,
server_lr=0.75, local_steps=5 for (a)(b)(c)(e), local_steps=10 for (d),
50 rounds, 10 seeds.

Usage:
  python3 scripts/T2_real_data.py --datasets wine --seeds 10
  python3 scripts/T2_real_data.py                      # everything
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

from fedfcmsim.datasets import make_feature_clients
from fedfcmsim.fcm import initialize_centers_from_data, predict_membership, run_local_fcm_steps
from fedfcmsim.federated import (
    _footprint_relevance,
    granular_footprint,
    pedrycz_gradient_fcm,
    personalized_federated_fcm,
)
from fedfcmsim.metrics import clustering_accuracy

M = 2.0
ROUNDS = 50
LOCAL_STEPS = 5          # arms (a)(b)(c)(e)
PEDRYCZ_LOCAL_STEPS = 10 # arm (d)
PEDRYCZ_ALPHA = 0.01     # stable; 0.1 diverges (see results/t1b_pedrycz_gradient/alpha_sensitivity.md)
SERVER_LR = 0.75
DELTA_LR = 0.7
DELTA_REG = 0.05
MIN_RELEVANCE = 0.05
FOOTPRINT_GRID = 60
SEED_OFFSET = 1201       # same as the published r1 benchmark configs

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "t2_real_data"

# (label, dataset, n_clients, labels_per_client, min_client_size)
# values match the published r1 result configs exactly
DATASETS = [
    ("digits_pca16",         "digits_pca16",    10, 2, 5),
    ("digits_pca32",         "digits_pca32",    10, 2, 5),
    ("wine",                 "wine",            10, 2, 5),
    ("pendigits",            "pendigits",       10, 3, 5),
    ("satimage",             "satimage",        10, 3, 5),
    ("letter",               "letter",          10, 6, 50),
    ("mnist784_pca32 (20c)", "mnist784_pca32",  20, 3, 50),
    ("mnist784_pca32 (50c)", "mnist784_pca32",  50, 3, 20),
]

ARMS = ["POST", "PRE", "GF", "PEDRYCZ", "GATE_ONLY"]

# Published accepted-paper mean client ACC for the acceptance check
PUBLISHED = {
    "digits_pca16":         (0.561, 0.632),
    "digits_pca32":         (0.524, 0.646),
    "wine":                 (0.926, 0.931),
    "pendigits":            (0.557, 0.611),
    "satimage":             (0.719, 0.675),
    "letter":               (0.378, 0.360),
    "mnist784_pca32 (20c)": (0.353, 0.395),
    "mnist784_pca32 (50c)": (0.386, 0.409),
}


def min_dist(centers: np.ndarray) -> float:
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def client_accs(clients, centers_by_client) -> tuple[float, float]:
    accs = []
    for c, cen in zip(clients, centers_by_client):
        labels = np.argmax(predict_membership(c.x, cen, m=M), axis=1)
        accs.append(clustering_accuracy(c.y, labels))
    return float(np.mean(accs)), float(np.min(accs))


def global_eval(clients, centers) -> dict[str, float]:
    mean_acc, worst_acc = client_accs(clients, [centers] * len(clients))
    return {"min_dist": min_dist(centers), "mean_acc": mean_acc, "worst_acc": worst_acc}


def run_mass_weighted(clients, V0, L, weight_at, gate=False, rounds=ROUNDS, slr=SERVER_LR):
    """VERIFY4.py `run` verbatim: shared mass-weighted server update; the only
    choices are WHERE the fuzzy mass is measured (pre/post adaptation) and
    whether the footprint relevance gate multiplies it."""
    V = V0.copy()
    N = np.array([len(c.x) for c in clients], float)
    beta = N / N.sum()
    for _ in range(rounds):
        num = np.zeros_like(V)
        den = np.zeros(V.shape[0])
        for p, c in enumerate(clients):
            ubar = predict_membership(c.x, V, m=M)            # at BROADCAST prototypes
            loc = run_local_fcm_steps(c.x, V, steps=L, m=M)   # L local Picard steps
            mass = (ubar ** M).sum(0) if weight_at == "pre" else (loc.membership ** M).sum(0)
            a = np.ones(V.shape[0])
            if gate:
                fp, _ = granular_footprint(c.x, V, ubar, m=M, grid_size=FOOTPRINT_GRID)
                a = _footprint_relevance(fp, MIN_RELEVANCE)
            w = beta[p] * mass * a
            num += w[:, None] * loc.centers
            den += w
        T = V.copy()
        ok = den > 1e-12
        T[ok] = num[ok] / den[ok, None]
        V = (1 - slr) * V + slr * T
    return V


def run_one_seed(dataset: str, n_clients: int, labels_per_client: int,
                 min_size: int, seed: int) -> list[dict[str, object]]:
    """All arms for one (dataset-config, seed). Returns metric rows."""
    clients, n_clusters = make_feature_clients(
        dataset=dataset,
        n_clients=n_clients,
        partition="support_skew",
        labels_per_client=labels_per_client,
        min_size=min_size,
        random_state=SEED_OFFSET + seed,
    )
    pooled = np.vstack([c.x for c in clients])
    init = initialize_centers_from_data(pooled, n_clusters, random_state=SEED_OFFSET + seed + 77)

    rows = []

    # (a) POST: mass from post-adaptation membership (repo F-FCM baseline)
    V = run_mass_weighted(clients, init, LOCAL_STEPS, "post", gate=False)
    rows.append({"arm": "POST", **global_eval(clients, V)})

    # (b) PRE: mass at broadcast prototypes ("aligned")
    V = run_mass_weighted(clients, init, LOCAL_STEPS, "pre", gate=False)
    rows.append({"arm": "PRE", **global_eval(clients, V)})

    # (c) GF-PFedFCM: footprint gate + personalization (accepted paper)
    gf = personalized_federated_fcm(
        clients, init, rounds=ROUNDS, local_steps=LOCAL_STEPS, m=M,
        server_lr=SERVER_LR, delta_lr=DELTA_LR, delta_reg=DELTA_REG,
        min_relevance=MIN_RELEVANCE, use_footprint=True,
        footprint_grid_size=FOOTPRINT_GRID, track_objective=False,
    )
    gf_personal = [gf.centers + gf.deltas[p] for p in range(len(clients))]
    mean_acc, worst_acc = client_accs(clients, gf_personal)
    rows.append({"arm": "GF", "min_dist": min_dist(gf.centers),
                 "mean_acc": mean_acc, "worst_acc": worst_acc})

    # (d) PEDRYCZ: original gradient aggregation at stable alpha
    ped = pedrycz_gradient_fcm(
        clients, init, rounds=ROUNDS, local_steps=PEDRYCZ_LOCAL_STEPS,
        m=M, alpha=PEDRYCZ_ALPHA, track_objective=False,
    )
    rows.append({"arm": "PEDRYCZ", **global_eval(clients, ped.centers)})

    # (e) GATE_ONLY (supplementary): VERIFY4 arm (c) literally -- POST mass x
    # footprint gate, no personalization, evaluated on global prototypes
    V = run_mass_weighted(clients, init, LOCAL_STEPS, "post", gate=True)
    rows.append({"arm": "GATE_ONLY", **global_eval(clients, V)})

    return rows


def _worker(task):
    label, dataset, n_clients, lpc, min_size, seed = task
    t0 = time.time()
    rows = run_one_seed(dataset, n_clients, lpc, min_size, seed)
    for r in rows:
        r.update({"dataset": label, "seed": seed})
    return label, seed, rows, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of dataset labels (default: all)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="extra suffix on output files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = OUT_DIR / f"per_seed_metrics{suffix}.csv"

    wanted = set(args.datasets.split(",")) if args.datasets else None
    configs = [d for d in DATASETS if wanted is None or d[0] in wanted]
    if not configs:
        raise SystemExit(f"no datasets matched {wanted}; known: {[d[0] for d in DATASETS]}")

    tasks = [
        (label, dataset, n_clients, lpc, min_size, seed)
        for (label, dataset, n_clients, lpc, min_size) in configs
        for seed in range(args.seeds)
    ]

    all_rows: list[dict[str, object]] = []
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            label, seed, rows, elapsed = fut.result()
            all_rows.extend(rows)
            done += 1
            print(f"[{done}/{len(tasks)}] {label} seed={seed} done in {elapsed:.1f}s "
                  f"({time.time()-t_start:.0f}s total)", flush=True)

    fieldnames = ["dataset", "seed", "arm", "min_dist", "mean_acc", "worst_acc"]
    all_rows.sort(key=lambda r: (r["dataset"], r["seed"], ARMS.index(r["arm"])))
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows([{k: r[k] for k in fieldnames} for r in all_rows])
    print(f"appended {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
