"""T3: the aggregation-weight DESIGN SPACE for federated fuzzy clustering.

Every method in this family computes the same server update

    v_j <- (1-slr)*v_j + slr * [ sum_p w_pj * c_pj / sum_p w_pj ],
    w_pj = beta_p * Mass_pj * Gate_pj

where c_pj is always the POST-adaptation local center. The methods differ
along three orthogonal axes:

  AXIS 1 (mass_point): where the fuzzy mass is measured
    "pre"  = memberships at the BROADCAST prototypes V (pre-adaptation)
    "post" = memberships after L local FCM steps (post-adaptation)
  AXIS 2 (gate): the gate multiplying the mass
    "none"      Gate_pj = 1
    "footprint" Gate_pj = granular-footprint relevance r_pj (G=60, r_min=0.05)
    "mass"      Gate_pj = clip(Mbar_pj / max_l Mbar_pl, 0.05, 1), the
                normalized PRE-adaptation fuzzy mass used directly as a gate
  AXIS 3 (personalize): client-side gated offsets delta_pj
    "off"  evaluate all clients on the shared global prototypes
    "on"   evaluate each client on v_j + delta_pj; delta update identical to
           personalized_federated_fcm (delta_lr=0.7, delta_reg=0.05, gated by
           the SAME axis-2 gate; ungated when gate="none")

2 x 3 x 2 = 12 cells. Four must reproduce T2 arms exactly:
    (post, none,      off) == T2 POST       (F-FCM baseline)
    (pre,  none,      off) == T2 PRE
    (post, footprint, off) == T2 GATE_ONLY
    (post, footprint, on)  == T2 GF         (accepted GF-PFedFCM)

To make those identities exact, the stopping rule follows the reference
implementation each cell generalizes: non-personalized cells run a fixed
ROUNDS rounds (like T2's run_mass_weighted / VERIFY4 `run`); personalized
cells early-stop at tol=1e-5 on the global-center shift (like
personalized_federated_fcm).

Protocol replicates T2_real_data.py exactly: SEED_OFFSET=1201, partition
random_state = SEED_OFFSET+seed, init random_state = SEED_OFFSET+seed+77,
support_skew partition, m=2.0, server_lr=0.75, local_steps=5, 50 rounds.
Synthetic scenarios use make_synthetic_clients (12 clients x 220 samples,
4 clusters) with the same seeding scheme.

Per (dataset, seed) the script also records the correlation between the
footprint gate and the mass gate, measured on the GATE_ONLY trajectory
(post mass + footprint gate), both gates re-evaluated each round at the
broadcast prototypes:
    pearson_end / spearman_end  = per-client correlation across the K
        prototypes, averaged over clients, at the FINAL round (rounds-end)
    pearson_all / spearman_all  = correlation pooled over all
        (round, client, prototype) entries

Usage:
  python3 scripts/T3_design_space.py --datasets wine --seeds 2
  python3 scripts/T3_design_space.py --datasets cluster_skew_hard --cells post_none_off
  python3 scripts/T3_design_space.py                       # everything
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedfcmsim.datasets import make_feature_clients
from fedfcmsim.fcm import EPS, initialize_centers_from_data, predict_membership, run_local_fcm_steps
from fedfcmsim.federated import _footprint_relevance, granular_footprint
from fedfcmsim.metrics import clustering_accuracy
from fedfcmsim.synthetic import BASE_CENTERS, make_synthetic_clients

M = 2.0

# Per-dataset fuzzifier m*: the largest m in {2.0, 1.5, 1.3, 1.2, 1.1} at which CENTRALIZED FCM keeps all
# c prototypes distinct on every seed (a label-free non-degeneracy rule; at m=2 FCM collapses to the
# centre of gravity on these high-dimensional whitened sets -- Winkler, Klawonn & Kruse, IJFSA 2011).
# Enabled via the environment variable FFCM_MSTAR=1 so that spawned process-pool workers inherit it.
import os as _os
MSTAR = {"pendigits": 1.5, "digits_pca16": 1.3, "letter": 1.3, "digits_pca32": 1.1,
         "mnist784_pca32 (20c)": 1.1, "mnist784_pca32 (50c)": 1.1}
USE_MSTAR = _os.environ.get("FFCM_MSTAR") == "1"


# FFCM_M forces one fuzzifier for every dataset (fuzziness-sensitivity runs); FFCM_NO_EARLYSTOP=1 makes
# personalized cells run the full round budget like the non-personalized ones (removes a stopping confound).
FORCE_M = _os.environ.get("FFCM_M")
NO_EARLYSTOP = _os.environ.get("FFCM_NO_EARLYSTOP") == "1"


def fuzzifier_for(label: str) -> float:
    if FORCE_M is not None:
        return float(FORCE_M)
    return MSTAR.get(label, 2.0) if USE_MSTAR else 2.0
# Communication rounds; overridable via env FFCM_ROUNDS (read at import, like FFCM_L). Default 50.
import os as _os_r
ROUNDS = int(_os_r.environ.get("FFCM_ROUNDS", "50"))
# Local FCM steps per round. Overridable via env FFCM_L (read at import time so that spawned workers and
# the def-time default of run_cell(local_steps=LOCAL_STEPS) both see it); default 5 = the paper protocol.
import os as _os_l
LOCAL_STEPS = int(_os_l.environ.get("FFCM_L", "5"))
# First seed index. Overridable via env FFCM_SEED_START for confirmatory runs on fresh seeds (default 0).
SEED_START = int(_os_l.environ.get("FFCM_SEED_START", "0"))
SERVER_LR = 0.75
DELTA_LR = 0.7
DELTA_REG = 0.05
MIN_RELEVANCE = 0.05
FOOTPRINT_GRID = 60
TOL = 1e-5                # personalized_federated_fcm early-stop tolerance
SEED_OFFSET = 1201        # same as the published r1 benchmark configs / T2

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "t3_design_space"

# ---------------------------------------------------------------------------
# Datasets: the six synthetic scenarios (12 clients x 220, 4 clusters) plus the
# eight real-data configs of T2_real_data.py (verbatim).
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
REAL_DATASETS = [
    ("digits_pca16",         "digits_pca16",    10, 2, 5),
    ("digits_pca32",         "digits_pca32",    10, 2, 5),
    ("wine",                 "wine",            10, 2, 5),
    ("pendigits",            "pendigits",       10, 3, 5),
    ("satimage",             "satimage",        10, 3, 5),
    ("letter",               "letter",          10, 6, 50),
    ("mnist784_pca32 (20c)", "mnist784_pca32",  20, 3, 50),
    ("mnist784_pca32 (50c)", "mnist784_pca32",  50, 3, 20),
]

MASS_POINTS = ["post", "pre"]
GATES = ["none", "footprint", "mass"]
PERSONALIZE = ["off", "on"]
CELLS = [(mp, g, pe) for mp in MASS_POINTS for g in GATES for pe in PERSONALIZE]

# Aliases for the four cells with known T2 equivalents (for --cells selection)
CELL_ALIASES = {
    "POST": ("post", "none", "off"),
    "PRE": ("pre", "none", "off"),
    "GATE_ONLY": ("post", "footprint", "off"),
    "GF": ("post", "footprint", "on"),
}


def cell_name(mass_point: str, gate: str, personalize: str) -> str:
    return f"{mass_point}_{gate}_{personalize}"


# ---------------------------------------------------------------------------
# Gates (axis 2). Signature: gate(x, V, ubar) -> (K,) relevance in [r_min, 1],
# where ubar = predict_membership(x, V) is the PRE-adaptation membership at the
# broadcast prototypes.
# ---------------------------------------------------------------------------
def gate_none(x: np.ndarray, V: np.ndarray, ubar: np.ndarray) -> np.ndarray:
    return np.ones(V.shape[0])


def gate_footprint(x: np.ndarray, V: np.ndarray, ubar: np.ndarray) -> np.ndarray:
    fp, _ = granular_footprint(x, V, ubar, m=M, grid_size=FOOTPRINT_GRID)
    return _footprint_relevance(fp, MIN_RELEVANCE)


def gate_mass(x: np.ndarray, V: np.ndarray, ubar: np.ndarray) -> np.ndarray:
    """Normalized PRE-adaptation fuzzy mass used directly as a gate:
    clip(Mbar_pj / max_l Mbar_pl, r_min, 1). No radius grid, no
    coverage/specificity search."""
    mbar = np.sum(ubar**M, axis=0)
    mx = float(np.max(mbar))
    if mx <= EPS:
        return np.full(V.shape[0], MIN_RELEVANCE)
    return np.clip(mbar / mx, MIN_RELEVANCE, 1.0)


GATE_FNS = {"none": gate_none, "footprint": gate_footprint, "mass": gate_mass}


# ---------------------------------------------------------------------------
# One cell of the design space.
# ---------------------------------------------------------------------------
def run_cell(clients, init, mass_point: str, gate: str, personalize: str,
             rounds: int = ROUNDS, local_steps: int = LOCAL_STEPS,
             slr: float = SERVER_LR, record_gates: bool = False):
    """Shared mass-weighted server update with the three axis choices.

    Returns (V, eval_centers, extras):
      V            final global prototypes
      eval_centers per-client prototype list for evaluation (V + delta_p when
                   personalize="on", else V for every client)
      extras       when record_gates=True, "fp_gates"/"mass_gates" hold the
                   per-round (n_clients, K) footprint/mass gate matrices
    """
    V = np.asarray(init, dtype=float).copy()
    n_clients = len(clients)
    n_clusters = V.shape[0]
    N = np.array([len(c.x) for c in clients], float)
    beta = N / N.sum()
    deltas = np.zeros((n_clients, n_clusters, V.shape[1]))
    gate_fn = GATE_FNS[gate]
    pers = personalize == "on"
    fp_hist, mass_hist = [], []

    for _ in range(rounds):
        previous = V.copy()
        num = np.zeros_like(V)
        den = np.zeros(n_clusters)
        fp_round = np.zeros((n_clients, n_clusters))
        mass_round = np.zeros((n_clients, n_clusters))

        for p, c in enumerate(clients):
            ubar = predict_membership(c.x, V, m=M)            # at BROADCAST prototypes
            start = V + deltas[p] if pers else V
            loc = run_local_fcm_steps(c.x, start, steps=local_steps, m=M)
            mass = np.sum(ubar**M, axis=0) if mass_point == "pre" \
                else np.sum(loc.membership**M, axis=0)
            g = gate_fn(c.x, V, ubar)
            w = beta[p] * mass * g
            num += w[:, None] * loc.centers
            den += w

            if pers:
                desired = loc.centers - V
                deltas[p] = (1.0 - DELTA_LR) * deltas[p] + DELTA_LR * (g[:, None] * desired)
                deltas[p] *= 1.0 - DELTA_REG

            if record_gates:
                fp_round[p] = g if gate == "footprint" else gate_footprint(c.x, V, ubar)
                mass_round[p] = g if gate == "mass" else gate_mass(c.x, V, ubar)

        T = V.copy()
        ok = den > EPS
        T[ok] = num[ok] / den[ok, None]
        V = (1.0 - slr) * V + slr * T

        if record_gates:
            fp_hist.append(fp_round)
            mass_hist.append(mass_round)
        if pers and not NO_EARLYSTOP and np.linalg.norm(V - previous) < TOL:
            break

    eval_centers = [V + deltas[p] for p in range(n_clients)] if pers else [V] * n_clients
    return V, eval_centers, {"fp_gates": fp_hist, "mass_gates": mass_hist}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def min_dist(centers: np.ndarray) -> float:
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def fcm_objective_mean(clients, centers_by_client) -> float:
    """Label-free fuzzy criterion: the FCM objective sum_k sum_j u_kj^m ||x_k - v_j||^2 of each client's evaluation
    prototypes on that client's data (memberships from those prototypes), summed over clients and divided by the
    total number of points. Lower is better; comparable across cells of one configuration at fixed m."""
    tot, n = 0.0, 0
    for c, cen in zip(clients, centers_by_client):
        u = predict_membership(c.x, cen, m=M)
        d2 = ((c.x[:, None, :] - np.asarray(cen)[None, :, :]) ** 2).sum(-1)
        tot += float(((u ** M) * d2).sum()); n += len(c.x)
    return tot / max(n, 1)


def client_accs(clients, centers_by_client) -> tuple[float, float]:
    accs = []
    for c, cen in zip(clients, centers_by_client):
        labels = np.argmax(predict_membership(c.x, cen, m=M), axis=1)
        accs.append(clustering_accuracy(c.y, labels))
    return float(np.mean(accs)), float(np.min(accs))


def _corr_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(Pearson, Spearman) between flattened gate vectors; NaN if degenerate."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
        return float("nan"), float("nan")
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b)[0])


def gate_correlations(fp_hist, mass_hist) -> dict[str, float]:
    """Footprint-vs-mass gate correlation on one trajectory.

    *_end : per-client correlation across the K prototypes, averaged over
            clients, at the final round.
    *_all : correlation pooled over all (round, client, prototype) entries.
    """
    fp_end, mass_end = fp_hist[-1], mass_hist[-1]
    pear, spear = [], []
    for p in range(fp_end.shape[0]):
        pr, sp = _corr_pair(fp_end[p], mass_end[p])
        pear.append(pr)
        spear.append(sp)
    fp_all = np.concatenate([h.ravel() for h in fp_hist])
    mass_all = np.concatenate([h.ravel() for h in mass_hist])
    pr_all, sp_all = _corr_pair(fp_all, mass_all)
    return {
        "pearson_end": float(np.nanmean(pear)),
        "spearman_end": float(np.nanmean(spear)),
        "pearson_all": pr_all,
        "spearman_all": sp_all,
    }


# ---------------------------------------------------------------------------
# Dataset construction (T2 protocol: SEED_OFFSET=1201, partition rs = offset+seed,
# init rs = offset+seed+77).
# ---------------------------------------------------------------------------
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


def run_one_seed(spec: tuple, seed: int, cells: list[tuple]) -> tuple[list[dict], dict]:
    """All requested cells for one (dataset-config, seed)."""
    global M
    M = fuzzifier_for(spec[1])
    clients, n_clusters = build_clients(spec, seed)
    pooled = np.vstack([c.x for c in clients])
    init = initialize_centers_from_data(pooled, n_clusters,
                                        random_state=SEED_OFFSET + seed + 77)

    rows = []
    probe = ("post", "footprint", "off")
    probe_res = None
    for mass_point, gate, pers in cells:
        rec = (mass_point, gate, pers) == probe
        V, eval_centers, extras = run_cell(
            clients, init, mass_point, gate, pers, record_gates=rec)
        if rec:
            probe_res = extras
        mean_acc, worst_acc = client_accs(clients, eval_centers)
        rows.append({
            "cell": cell_name(mass_point, gate, pers),
            "mass_point": mass_point,
            "gate": gate,
            "personalize": pers,
            "min_dist": min_dist(V),
            "mean_acc": mean_acc,
            "worst_acc": worst_acc,
            "fcm_obj": fcm_objective_mean(clients, eval_centers),
        })

    if probe_res is None:  # probe cell not selected: run it for the correlations
        _, _, probe_res = run_cell(clients, init, *probe, record_gates=True)
    corr = gate_correlations(probe_res["fp_gates"], probe_res["mass_gates"])
    return rows, corr


def _worker(task):
    spec, seed, cells = task
    t0 = time.time()
    rows, corr = run_one_seed(spec, seed, cells)
    return spec[1], seed, rows, corr, time.time() - t0


def _parse_cells(text: str | None) -> list[tuple]:
    if not text:
        return list(CELLS)
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok.upper() in CELL_ALIASES:
            cell = CELL_ALIASES[tok.upper()]
        else:
            parts = tok.lower().split("_")
            if len(parts) == 3 and parts[0] in MASS_POINTS and \
                    parts[1] in GATES and parts[2] in PERSONALIZE:
                cell = tuple(parts)
            else:
                raise SystemExit(
                    f"unknown cell '{tok}'; use mass_gate_pers like "
                    f"'post_none_off' or aliases {sorted(CELL_ALIASES)}")
        if cell not in out:
            out.append(cell)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of dataset labels (default: all)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cells", default=None,
                    help="comma-separated cells (e.g. post_none_off,GF); default all 12")
    ap.add_argument("--tag", default="", help="extra suffix on output files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = OUT_DIR / f"per_seed_metrics{suffix}.csv"
    md_path = OUT_DIR / f"summary{suffix}.md"

    cells = _parse_cells(args.cells)

    specs = [("synthetic", s) for s in SYNTHETIC_SCENARIOS]
    specs += [("real",) + tuple(d) for d in REAL_DATASETS]
    wanted = set(args.datasets.split(",")) if args.datasets else None
    configs = [s for s in specs if wanted is None or s[1] in wanted]
    if not configs:
        raise SystemExit(f"no datasets matched {wanted}; known: {[s[1] for s in specs]}")

    tasks = [(spec, seed, cells) for spec in configs for seed in range(SEED_START, SEED_START + args.seeds)]

    all_rows: list[dict] = []
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            label, seed, rows, corr, elapsed = fut.result()
            for r in rows:
                r.update({"dataset": label, "seed": seed, **corr})
            all_rows.extend(rows)
            done += 1
            print(f"[{done}/{len(tasks)}] {label} seed={seed} done in {elapsed:.1f}s "
                  f"({time.time()-t_start:.0f}s total)", flush=True)

    cell_order = {cell_name(*c): i for i, c in enumerate(CELLS)}
    fieldnames = ["dataset", "seed", "cell", "mass_point", "gate", "personalize",
                  "min_dist", "mean_acc", "worst_acc",
                  "pearson_end", "spearman_end", "pearson_all", "spearman_all", "fcm_obj"]
    # NOTE: fcm_obj was added 2026-09-24 (T9). Always use a NEW --tag for new runs: appending to a CSV
    # created before that date would put 14 fields under a 13-field header.
    all_rows.sort(key=lambda r: (r["dataset"], r["seed"], cell_order[r["cell"]]))
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows([{k: r[k] for k in fieldnames} for r in all_rows])
    print(f"appended {len(all_rows)} rows to {csv_path}")

    # Markdown summary: cells as rows, mean over the seeds in THIS run.
    by_ds: dict[str, list[dict]] = {}
    for r in all_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    lines = [f"# T3 design space — {args.seeds} seed(s), tag='{args.tag}'\n"]
    for ds, rows in by_ds.items():
        lines.append(f"\n## {ds}\n")
        lines.append("| cell | mass | gate | pers | mean_acc | worst_acc | min_dist |")
        lines.append("|---|---|---|---|---|---|---|")
        for cell in cell_order:
            sub = [r for r in rows if r["cell"] == cell]
            if not sub:
                continue
            lines.append(
                f"| {cell} | {sub[0]['mass_point']} | {sub[0]['gate']} | "
                f"{sub[0]['personalize']} | {np.mean([r['mean_acc'] for r in sub]):.4f} | "
                f"{np.mean([r['worst_acc'] for r in sub]):.4f} | "
                f"{np.mean([r['min_dist'] for r in sub]):.4f} |")
        lines.append(
            f"\nGate corr (footprint vs mass): pearson_end="
            f"{np.nanmean([r['pearson_end'] for r in rows]):.4f}, spearman_end="
            f"{np.nanmean([r['spearman_end'] for r in rows]):.4f}, pearson_all="
            f"{np.nanmean([r['pearson_all'] for r in rows]):.4f}, spearman_all="
            f"{np.nanmean([r['spearman_all'] for r in rows]):.4f}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
