"""T4A: diagnose the digits_pca32 personalization anomaly.

In the T3 12-cell design space, digits_pca32 is the ONLY dataset where
ungated personalization helps (+0.15..+0.20 mean ACC regardless of gate or
mass measurement point); everywhere else ungated personalization costs
-0.14..-0.47. digits_pca16 does NOT show this. This script tests four
hypotheses:

  H1  PCA whitening: load_feature_dataset applies PCA(whiten=True). At 32
      components on the 8x8 digits data (64 raw features) the trailing
      components are near-noise and whitening rescales them to unit
      variance. Sweep n_components x whiten and locate where the
      personalization gain appears.
  H2  Intrinsic dimension / distance concentration: per setting, report
      mean/std of pairwise distances and the effective rank of the client
      covariance.
  H3  Support-skew partition interaction: with 10 clients x 2 supported
      labels on 10 classes, report the per-client label histogram and the
      number of clients sharing each label.
  H4  Evaluation, not training: score the same personalized run on the
      GLOBAL prototypes vs the PERSONALIZED prototypes, separating
      "personalization learns something" from "personalized evaluation is
      easier".

  H5  (added) Cluster recovery difficulty: measure per-setting baseline
      quality -- if the GLOBAL prototypes are already near-perfect on
      pca16 but not pca32, personalization has headroom to help only where
      the global solution is broken.

Protocol identical to T3: SEED_OFFSET=1201, partition rs=offset+seed,
init rs=offset+seed+77, support_skew 10 clients x 2 labels, m=2.0,
server_lr=0.75, local_steps=5, 50 rounds.

Usage:
  python3 scripts/T4A_digits32_diagnosis.py --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR / "scripts"))

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

from fedfcmsim.datasets import support_skew_partition
from fedfcmsim.fcm import initialize_centers_from_data, predict_membership
from fedfcmsim.metrics import clustering_accuracy
from T3_design_space import (  # noqa: E402
    CELLS,
    M,
    SEED_OFFSET,
    cell_name,
    client_accs,
    min_dist,
    run_cell,
)

OUT_DIR = EXPERIMENTS_DIR / "results" / "t4a_digits32"

# H1 sweep grid
N_COMPONENTS = [8, 16, 24, 32, 48, 64]
WHITEN = [True, False]

N_CLIENTS = 10
LABELS_PER_CLIENT = 2
MIN_SIZE = 5


def load_digits_pca(n_components: int, whiten: bool) -> tuple[np.ndarray, np.ndarray]:
    """Same pipeline as load_feature_dataset('digits_pcaN') but with a
    selectable whiten flag (the loader hard-codes whiten=True)."""
    data = load_digits()
    x = StandardScaler().fit_transform(np.asarray(data.data, dtype=float))
    x = PCA(n_components=n_components, whiten=whiten, random_state=0).fit_transform(x)
    return x, np.asarray(data.target)


# ---------------------------------------------------------------------------
# H2 geometry metrics
# ---------------------------------------------------------------------------
def geometry_metrics(x: np.ndarray, clients) -> dict[str, float]:
    d = pdist(x)
    ratio = float(d.mean() / d.std())

    def _ranks(X: np.ndarray) -> tuple[float, float]:
        cov = np.cov(X, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.maximum(eig, 0.0)
        s = eig.sum()
        pr = float(s * s / np.sum(eig * eig))          # participation ratio
        p = eig / s
        p = p[p > 1e-15]
        erank = float(np.exp(-np.sum(p * np.log(p))))  # entropy effective rank
        return pr, erank

    pr_pool, er_pool = _ranks(x)
    prs, ers = [], []
    for c in clients:
        pr, er = _ranks(c.x)
        prs.append(pr)
        ers.append(er)
    return {
        "pdist_mean": float(d.mean()),
        "pdist_std": float(d.std()),
        "pdist_mean_over_std": ratio,
        "prank_pooled": pr_pool,
        "erank_pooled": er_pool,
        "prank_client_mean": float(np.mean(prs)),
        "erank_client_mean": float(np.mean(ers)),
    }


# ---------------------------------------------------------------------------
# One (n_components, whiten, seed) task: run all 12 cells + dual evaluation
# ---------------------------------------------------------------------------
def run_setting_seed(task):
    n_comp, whiten, seed = task
    x, y = load_digits_pca(n_comp, whiten)
    clients = support_skew_partition(
        x=x, y=y, n_clients=N_CLIENTS, labels_per_client=LABELS_PER_CLIENT,
        random_state=SEED_OFFSET + seed, min_size=MIN_SIZE)
    n_clusters = int(len(np.unique(y)))
    pooled = np.vstack([c.x for c in clients])
    init = initialize_centers_from_data(pooled, n_clusters,
                                        random_state=SEED_OFFSET + seed + 77)

    rows = []
    for mass_point, gate, pers in CELLS:
        V, eval_centers, _ = run_cell(clients, init, mass_point, gate, pers)
        mean_acc, worst_acc = client_accs(clients, eval_centers)
        row = {
            "n_comp": n_comp, "whiten": whiten, "seed": seed,
            "cell": cell_name(mass_point, gate, pers),
            "min_dist": min_dist(V),
            "mean_acc": mean_acc, "worst_acc": worst_acc,
            "mean_acc_global": "",   # H4: filled for personalized cells
            "worst_acc_global": "",
        }
        if pers == "on":
            g_mean, g_worst = client_accs(clients, [V] * len(clients))
            row["mean_acc_global"] = g_mean
            row["worst_acc_global"] = g_worst
        rows.append(row)
    return rows


def label_histogram(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """H3: per-client label histogram for the support_skew partition."""
    x, y = load_digits_pca(32, True)  # partition depends only on y + seed
    clients = support_skew_partition(
        x=x, y=y, n_clients=N_CLIENTS, labels_per_client=LABELS_PER_CLIENT,
        random_state=SEED_OFFSET + seed, min_size=MIN_SIZE)
    labels = np.unique(y)
    hist = np.zeros((len(clients), len(labels)), dtype=int)
    for p, c in enumerate(clients):
        for l in labels:
            hist[p, l] = int(np.sum(c.y == l))
    clients_per_label = np.sum(hist > 0, axis=0)
    return hist, clients_per_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- H3: partition structure (seed-independent label pattern) -----------
    hist, cpl = label_histogram(seed=0)
    print("H3 per-client label histogram (seed 0):")
    print(hist)
    print("clients per label:", cpl)

    # --- H2: geometry per setting (seed 0 clients, representative) ----------
    geo_rows = []
    for n_comp in N_COMPONENTS:
        for whiten in WHITEN:
            x, y = load_digits_pca(n_comp, whiten)
            clients = support_skew_partition(
                x=x, y=y, n_clients=N_CLIENTS,
                labels_per_client=LABELS_PER_CLIENT,
                random_state=SEED_OFFSET, min_size=MIN_SIZE)
            gm = geometry_metrics(x, clients)
            gm.update({"n_comp": n_comp, "whiten": whiten})
            geo_rows.append(gm)
            print(f"geo n={n_comp} whiten={whiten}: "
                  f"mean/std={gm['pdist_mean_over_std']:.3f} "
                  f"erank_pool={gm['erank_pooled']:.1f} "
                  f"erank_client={gm['erank_client_mean']:.1f}", flush=True)

    with (OUT_DIR / "geometry.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(geo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(geo_rows)

    # --- H1 + H4: full 12-cell sweep over the grid ---------------------------
    tasks = [(n, w, s) for n in N_COMPONENTS for w in WHITEN
             for s in range(args.seeds)]
    all_rows: list[dict] = []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_setting_seed, t): t for t in tasks}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done += 1
            print(f"[{done}/{len(tasks)}] done ({time.time()-t0:.0f}s)", flush=True)

    fieldnames = ["n_comp", "whiten", "seed", "cell", "min_dist",
                  "mean_acc", "worst_acc", "mean_acc_global", "worst_acc_global"]
    all_rows.sort(key=lambda r: (r["n_comp"], r["whiten"], r["seed"], r["cell"]))
    with (OUT_DIR / "per_seed_metrics.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {OUT_DIR / 'per_seed_metrics.csv'} ({len(all_rows)} rows)")

    # Personalization-gain summary: mean over seeds of (on - off) per cell pair.
    print("\nPersonalization gain (pers_on - pers_off), mean over seeds:")
    print(f"{'n_comp':>6} {'whiten':>6} | " +
          " ".join(f"{mp}_{g}" for mp in ("post", "pre")
                   for g in ("none", "footprint", "mass")))
    for n_comp in N_COMPONENTS:
        for whiten in WHITEN:
            gains = []
            for mp in ("post", "pre"):
                for g in ("none", "footprint", "mass"):
                    on = [r["mean_acc"] for r in all_rows
                          if r["n_comp"] == n_comp and r["whiten"] == whiten
                          and r["cell"] == f"{mp}_{g}_on"]
                    off = [r["mean_acc"] for r in all_rows
                           if r["n_comp"] == n_comp and r["whiten"] == whiten
                           and r["cell"] == f"{mp}_{g}_off"]
                    gains.append(np.mean(on) - np.mean(off))
            print(f"{n_comp:>6} {str(whiten):>6} | " +
                  " ".join(f"{v:+.3f}" for v in gains), flush=True)


if __name__ == "__main__":
    main()
