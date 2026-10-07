"""T10: every experiment of experiments/T10_PREREGISTRATION.md, as subcommands of one module.

Subcommands (each writes ONE new CSV  results/t10/<tag>.csv  plus  <tag>.meta.json; an existing file is never
appended to and is only overwritten with --force):

  cells              T3 design-space cells (the 12 of T3 + fpmask/massmask gates) with the extra metrics
                     gacc / xb / hnorm; --participation < 1 (E-G) uses a sampled copy of T3.run_cell,
                     --split 0.2 (E-C) fits on 80% of every client and evaluates on the held-out 20%.
  scffcm             SC-FFCM (fedfcmsim.federated.scffcm) with steps from a calibration CSV or --eta-l/--eta-g.
  scffcm-calibrate   label-free step calibration on seed 100 (T9b rule) on a named or explicit grid.
  cfcm               centralized FCM exactly as T6_baselines.py's centralized_fcm row (max_iter 150, same init).
  pedrycz-calibrate  label-free alpha calibration (T9b rule) for fedfcmsim.federated.pedrycz_gradient_fcm.
  pedrycz            Pedrycz gradient federated FCM with the calibrated alpha.
  audit              E-F: m-grid degeneracy audit, centralized FCM from paper / k-means++ best-of-10 / oracle
                     inits, plus the federated lossless PRE one-step exchange.
  compare            bit-for-bit comparison of a T10 CSV with the prior CSVs (exit status 1 on any strict mismatch,
                     on zero overlap, on an uncovered in-scope prior row, or on fewer overlapping runs than
                     --min-runs; acc-only comparisons across stopping rules are advisory and only reported).
  argcheck           used by T10_run_all.sh before skipping a step: does <tag>.meta.json record the same command?
  analyze            the pre-registered statistics (§1 H1-H9 + secondary, §3 exact Wilcoxon / Holm / t-intervals /
                     sensitivity Holm / re-analysis of the earlier blocks, E-A interactions, E-C, decision analysis).

Environment. T3_design_space.py reads FFCM_MSTAR / FFCM_L / FFCM_ROUNDS / FFCM_SEED_START / FFCM_M /
FFCM_NO_EARLYSTOP AT IMPORT (run_cell's defaults are bound then). This module never imports T3 at module level:
every subcommand first writes the protocol values into os.environ (FFCM_MSTAR=1, FFCM_L, FFCM_ROUNDS,
FFCM_SEED_START; FFCM_M and FFCM_NO_EARLYSTOP removed) and only then imports T3, and every pool worker
receives the same values through its initializer before it imports T3. get_T() asserts that the imported
module's constants equal the requested ones, so a stale import cannot go unnoticed.

Seeds 20-29 are the pre-registered fresh block: any command touching them needs --allow-fresh.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import platform
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
OUT_DIR = RESULTS / "t10"
PAPER2 = EXP.parent / "paper2"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

# ---------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------
CONFIGS = ["cluster_skew_hard", "cluster_skew_overlap", "dirichlet_0.03", "dirichlet_0.1", "overlap_noise",
           "quantity_skew_extreme", "wine", "satimage", "pendigits", "digits_pca16", "digits_pca32", "letter",
           "mnist784_pca32 (20c)", "mnist784_pca32 (50c)"]
NON_MNIST = [c for c in CONFIGS if not c.startswith("mnist")]
# the 8 real configurations of paper2/compute_degeneracy.py, in its order
REAL8 = ["wine", "satimage", "pendigits", "digits_pca16", "letter", "digits_pca32",
         "mnist784_pca32 (20c)", "mnist784_pca32 (50c)"]
DATASET_ALIASES = {"all": CONFIGS, "nonmnist": NON_MNIST, "real8": REAL8,
                   "mnist20": ["mnist784_pca32 (20c)"], "mnist50": ["mnist784_pca32 (50c)"]}
# rough relative cost, only used to submit the expensive tasks first (load balancing; no effect on results)
COST = {"mnist784_pca32 (50c)": 100, "mnist784_pca32 (20c)": 90, "letter": 60, "satimage": 20, "pendigits": 20,
        "digits_pca32": 8, "digits_pca16": 8}

MASS_POINTS = ["post", "pre"]
GATES = ["none", "footprint", "mass", "fpmask", "massmask"]
PERSONALIZE = ["off", "on"]
T3_CELLS = [(mp, g, pe) for mp in MASS_POINTS for g in ("none", "footprint", "mass") for pe in PERSONALIZE]
MASK_CELLS = [(mp, g, "off") for mp in MASS_POINTS for g in ("fpmask", "massmask")]
CELLS16 = T3_CELLS + MASK_CELLS
CELL_ALIASES = {"POST": ("post", "none", "off"), "PRE": ("pre", "none", "off"),
                "GATE_ONLY": ("post", "footprint", "off"), "GF": ("post", "footprint", "on")}
CELL_GROUPS = {"all12": T3_CELLS, "all16": CELLS16, "masks": MASK_CELLS}

# E-D extended SC-FFCM grid and the T9b grid (to reproduce results/t9b_scffcm_calibration.csv)
SCFFCM_GRIDS = {
    "t9b": ((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0), (0.5, 1.0, 2.0, 4.0, 8.0)),
    "ed": ((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0), (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)),
}
PEDRYCZ_GRIDS = {"ee": ((0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),)}   # one axis (alpha)
CAL_SEED = 100
FRESH_SEEDS = range(20, 30)
M_GRID = (2.0, 1.5, 1.3, 1.2, 1.1)

METRIC_KEYS = ["min_dist", "mean_acc", "worst_acc", "fcm_obj", "gacc", "xb", "hnorm"]
# E-C held-out metrics. PRIMARY: test_fcm_obj, test_gacc_trmap (pooled label map fitted on the TRAIN part),
# test_mean_acc_trmap / test_worst_acc_trmap (each client's label map fitted on its own TRAIN part), test_hnorm.
# test_gacc (one Hungarian on the pooled test part) is reported too. test_mean_acc_refit / test_worst_acc_refit re-fit
# a Hungarian map on each client's few test points and are biased upward on small test sets (a 1-point test set
# always scores 1.0); they are kept only as a diagnostic.
TEST_KEYS = ["test_fcm_obj", "test_gacc_trmap", "test_mean_acc_trmap", "test_worst_acc_trmap", "test_gacc",
             "test_hnorm", "test_mean_acc_refit", "test_worst_acc_refit"]

# ---------------------------------------------------------------------------------------------------------------
# Environment handling and the lazily imported T3 module
# ---------------------------------------------------------------------------------------------------------------
_T = None


def protocol_env(L: int, rounds: int, seed_start: int = 0, no_earlystop: bool = False) -> dict:
    env = {"FFCM_MSTAR": "1", "FFCM_L": str(int(L)), "FFCM_ROUNDS": str(int(rounds)),
           "FFCM_SEED_START": str(int(seed_start))}
    if no_earlystop:          # provenance checks only (the prior L=1 GF files were run this way); never in T10 runs
        env["FFCM_NO_EARLYSTOP"] = "1"
    return env


def apply_env(env: dict) -> None:
    os.environ.update(env)
    os.environ.pop("FFCM_M", None)              # the pre-registered protocol uses m*
    if "FFCM_NO_EARLYSTOP" not in env:          # ... and T3's stopping rule
        os.environ.pop("FFCM_NO_EARLYSTOP", None)


def gate_fpmask(x, V, ubar):
    """1 where the footprint relevance exceeds its floor r_min, else r_min (no magnitude)."""
    T = _T
    return np.where(T.gate_footprint(x, V, ubar) > T.MIN_RELEVANCE, 1.0, T.MIN_RELEVANCE)


def gate_massmask(x, V, ubar):
    """1 where the normalized PRE mass q exceeds r_min, else r_min (no magnitude)."""
    T = _T
    return np.where(T.gate_mass(x, V, ubar) > T.MIN_RELEVANCE, 1.0, T.MIN_RELEVANCE)


def get_T():
    """Import T3 (after the environment is set), check its constants, register the two mask gates."""
    global _T
    if _T is None:
        if os.environ.get("FFCM_MSTAR") != "1":
            raise RuntimeError("FFCM_MSTAR must be '1' before T3 is imported")
        import T3_design_space as T
        _T = T
    T = _T
    want_L = int(os.environ.get("FFCM_L", "5"))
    want_R = int(os.environ.get("FFCM_ROUNDS", "50"))
    want_noes = os.environ.get("FFCM_NO_EARLYSTOP") == "1"
    if not (T.USE_MSTAR and T.FORCE_M is None and T.NO_EARLYSTOP == want_noes):
        raise RuntimeError("T3 imported with a non-protocol fuzzifier/stopping environment")
    if T.LOCAL_STEPS != want_L or T.ROUNDS != want_R:
        raise RuntimeError(f"T3 imported with L={T.LOCAL_STEPS}, R={T.ROUNDS}; requested L={want_L}, R={want_R}")
    if run_cell_default(T, "local_steps") != want_L or run_cell_default(T, "rounds") != want_R:
        raise RuntimeError("T3.run_cell default arguments disagree with the requested L/R")
    T.GATE_FNS["fpmask"] = gate_fpmask
    T.GATE_FNS["massmask"] = gate_massmask
    return T


def run_cell_default(T, name):
    import inspect
    return inspect.signature(T.run_cell).parameters[name].default


def _init_worker(env: dict) -> None:
    apply_env(env)
    get_T()


def spec_for(T, label: str) -> tuple:
    specs = [("synthetic", s) for s in T.SYNTHETIC_SCENARIOS] + [("real",) + tuple(d) for d in T.REAL_DATASETS]
    for s in specs:
        if s[1] == label:
            return s
    raise KeyError(label)


_SETUP_CACHE: dict = {}


def setup(label: str, seed: int, cache: bool = False):
    """(T, clients, k, X, init) exactly as T3.run_one_seed / T9b.setup build them. cache=True (calibration grids,
    which reuse one (label, seed) many times) keeps the last two builds per process; no method mutates them."""
    from fedfcmsim.fcm import initialize_centers_from_data
    T = get_T()
    T.M = T.fuzzifier_for(label)
    if cache and (label, seed) in _SETUP_CACHE:
        clients, k, X, init = _SETUP_CACHE[(label, seed)]
        return T, clients, k, X, init.copy()
    clients, k = T.build_clients(spec_for(T, label), seed)
    X = np.vstack([c.x for c in clients])
    init = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    if cache:
        while len(_SETUP_CACHE) >= 2:
            _SETUP_CACHE.pop(next(iter(_SETUP_CACHE)))
        _SETUP_CACHE[(label, seed)] = (clients, k, X, init.copy())
    return T, clients, k, X, init


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def pooled_obj(X, V, m) -> float:
    """Total pooled FCM objective (T9b.pooled_obj)."""
    from fedfcmsim.fcm import predict_membership
    u = predict_membership(X, V, m=m)
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    return float(((u ** m) * d2).sum())


def xie_beni(X, V, m) -> float:
    """sum_i sum_j u_ij^m ||x_i - v_j||^2 / (N * min_{j!=k} ||v_j - v_k||^2), u = predict_membership(X, V, m)."""
    from fedfcmsim.fcm import predict_membership
    V = np.asarray(V, float)
    u = predict_membership(X, V, m=m)
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    num = float(((u ** m) * d2).sum())
    dv = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dv, np.inf)
    sep = float(dv.min())
    return num / (X.shape[0] * sep) if sep > 0 else float("inf")


def pooled_eval(clients, centers_by_client, m) -> tuple[float, float]:
    """(gacc, hnorm): every point assigned / scored under ITS OWN client's evaluation prototypes;
    gacc = one Hungarian alignment of the pooled labels, hnorm = mean normalized membership entropy."""
    from fedfcmsim.fcm import predict_membership
    from fedfcmsim.metrics import clustering_accuracy
    ys, preds, ents = [], [], []
    c = np.asarray(centers_by_client[0]).shape[0]
    for cl, cen in zip(clients, centers_by_client):
        u = predict_membership(cl.x, cen, m=m)
        preds.append(np.argmax(u, axis=1))
        ys.append(np.asarray(cl.y))
        ents.append(-(u * np.log(np.maximum(u, 1e-300))).sum(1))
    gacc = clustering_accuracy(np.concatenate(ys), np.concatenate(preds))
    hnorm = float(np.concatenate(ents).mean() / np.log(c))
    return float(gacc), hnorm


def eval_metrics(T, clients, V, eval_centers) -> dict:
    """T3's min_dist / mean_acc / worst_acc / fcm_obj (T3's own functions, T.M) + gacc, xb, hnorm."""
    finite = V is not None and np.all(np.isfinite(V)) and all(np.all(np.isfinite(c)) for c in eval_centers)
    if not finite:
        return {k: float("nan") for k in METRIC_KEYS}
    ma, wa = T.client_accs(clients, eval_centers)
    gacc, hnorm = pooled_eval(clients, eval_centers, T.M)
    out = {"mean_acc": ma, "worst_acc": wa, "fcm_obj": T.fcm_objective_mean(clients, eval_centers),
           "gacc": gacc, "hnorm": hnorm, "min_dist": T.min_dist(V),
           "xb": xie_beni(np.vstack([c.x for c in clients]), V, T.M)}
    return {k: out[k] for k in METRIC_KEYS}


_UNMAPPED = -(10 ** 9)


def hungarian_map(y_true, y_pred) -> dict:
    """Predicted-cluster -> label map maximizing agreement: the contingency table and linear_sum_assignment of
    fedfcmsim.metrics.clustering_accuracy, returned as a map (so it can be applied to OTHER points).
    Applied to the points it was fitted on it reproduces clustering_accuracy (T10_selftest xi)."""
    from scipy.optimize import linear_sum_assignment
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tl, pl = np.unique(y_true), np.unique(y_pred)
    n = max(len(tl), len(pl))
    C = np.zeros((n, n), dtype=int)
    ti = {v: i for i, v in enumerate(tl)}
    pi = {v: i for i, v in enumerate(pl)}
    for t, p in zip(y_true, y_pred):
        C[ti[t], pi[p]] += 1
    r, c = linear_sum_assignment(-C)
    return {pl[j]: tl[i] for i, j in zip(r, c) if i < len(tl) and j < len(pl)}


def mapped_acc(mp: dict, y_true, y_pred) -> float:
    """Accuracy of predictions relabelled by a FIXED map; clusters absent from the map count as errors."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.array([mp.get(p, _UNMAPPED) for p in y_pred]) == y_true))


def heldout_metrics(T, train, test, eval_centers) -> dict:
    """E-C held-out metrics on the 20% part, each point under its own client's evaluation prototypes (fit on 80%).
    *_trmap: the label map is fitted on the TRAIN part (per client for mean/worst, pooled for gacc) and applied to the
    test part -- no label information of the test points is used for the mapping. *_refit: a separate Hungarian map on
    each client's test points (upward-biased on tiny test sets; diagnostic only)."""
    from fedfcmsim.fcm import predict_membership
    from fedfcmsim.metrics import clustering_accuracy
    finite = all(np.all(np.isfinite(c)) for c in eval_centers)
    if not finite:
        return {k: float("nan") for k in TEST_KEYS}
    pr_tr = [np.argmax(predict_membership(c.x, cen, m=T.M), axis=1) for c, cen in zip(train, eval_centers)]
    pr_te = [np.argmax(predict_membership(c.x, cen, m=T.M), axis=1) for c, cen in zip(test, eval_centers)]
    acc_tr = [mapped_acc(hungarian_map(a.y, pa), b.y, pb) for a, b, pa, pb in zip(train, test, pr_tr, pr_te)]
    acc_rf = [clustering_accuracy(b.y, pb) for b, pb in zip(test, pr_te)]
    gmap = hungarian_map(np.concatenate([np.asarray(a.y) for a in train]), np.concatenate(pr_tr))
    gacc_te, hnorm_te = pooled_eval(test, eval_centers, T.M)
    return {"test_fcm_obj": T.fcm_objective_mean(test, eval_centers),
            "test_gacc_trmap": mapped_acc(gmap, np.concatenate([np.asarray(b.y) for b in test]), np.concatenate(pr_te)),
            "test_mean_acc_trmap": float(np.mean(acc_tr)), "test_worst_acc_trmap": float(np.min(acc_tr)),
            "test_gacc": gacc_te, "test_hnorm": hnorm_te,
            "test_mean_acc_refit": float(np.mean(acc_rf)), "test_worst_acc_refit": float(np.min(acc_rf))}


# ---------------------------------------------------------------------------------------------------------------
# Cells: parser, partial-participation copy of T3.run_cell, split-sample
# ---------------------------------------------------------------------------------------------------------------
def cell_name(c) -> str:
    return "_".join(c)


def parse_cells(text: str | None) -> list[tuple]:
    if not text:
        return list(CELLS16)
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() in CELL_GROUPS:
            cand = CELL_GROUPS[tok.lower()]
        elif tok.upper() in CELL_ALIASES:
            cand = [CELL_ALIASES[tok.upper()]]
        else:
            parts = tok.lower().split("_")
            if len(parts) == 3 and parts[0] in MASS_POINTS and parts[1] in GATES and parts[2] in PERSONALIZE:
                cand = [tuple(parts)]
            else:
                raise SystemExit(f"unknown cell '{tok}': use <pre|post>_<{'|'.join(GATES)}>_<off|on>, "
                                 f"{sorted(CELL_ALIASES)} or {sorted(CELL_GROUPS)}")
        for c in cand:
            if c not in out:
                out.append(tuple(c))
    return out


def run_cell_pp(T, clients, init, mass_point, gate, personalize, fraction, rng_seed,
                rounds=None, local_steps=None, slr=None):
    """Copy of T3.run_cell with client sampling: each round max(1, int(P*fraction)) distinct clients are drawn with
    ONE generator default_rng(rng_seed) (advanced every round, also when all clients are drawn), aggregation runs over
    the sampled clients only (same weights beta_p * mass * gate, beta_p = N_p / N over ALL clients), non-sampled
    clients keep their offsets. With fraction = 1.0 this is bit-for-bit T3.run_cell (T10_selftest (i))."""
    from fedfcmsim.fcm import EPS, predict_membership, run_local_fcm_steps
    rounds = T.ROUNDS if rounds is None else rounds
    local_steps = T.LOCAL_STEPS if local_steps is None else local_steps
    slr = T.SERVER_LR if slr is None else slr
    M = T.M
    V = np.asarray(init, dtype=float).copy()
    n_clients = len(clients)
    n_clusters = V.shape[0]
    N = np.array([len(c.x) for c in clients], float)
    beta = N / N.sum()
    deltas = np.zeros((n_clients, n_clusters, V.shape[1]))
    gate_fn = T.GATE_FNS[gate]
    pers = personalize == "on"
    n_part = max(1, int(n_clients * fraction))
    rng = np.random.default_rng(rng_seed)
    n_done = 0
    for _ in range(rounds):
        previous = V.copy()
        num = np.zeros_like(V)
        den = np.zeros(n_clusters)
        selected = np.sort(rng.choice(n_clients, size=n_part, replace=False))
        for p in selected:
            c = clients[p]
            ubar = predict_membership(c.x, V, m=M)
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
                deltas[p] = (1.0 - T.DELTA_LR) * deltas[p] + T.DELTA_LR * (g[:, None] * desired)
                deltas[p] *= 1.0 - T.DELTA_REG
        T_ = V.copy()
        ok = den > EPS
        T_[ok] = num[ok] / den[ok, None]
        V = (1.0 - slr) * V + slr * T_
        n_done += 1
        if pers and not T.NO_EARLYSTOP and np.linalg.norm(V - previous) < T.TOL:
            break
    eval_centers = [V + deltas[p] for p in range(n_clients)] if pers else [V] * n_clients
    return V, eval_centers, {"n_rounds": n_done, "n_part": n_part}


def split_clients(T, clients, seed: int, frac: float):
    """80/20 split of every client (client order) with ONE generator default_rng(SEED_OFFSET+seed+999):
    perm = rng.permutation(n_p); test = the first n_test = clip(floor(frac*n_p + 0.5), 1, n_p-1) positions;
    both index sets sorted so each part keeps the client's original order."""
    from fedfcmsim.synthetic import ClientData
    rng = np.random.default_rng(T.SEED_OFFSET + seed + 999)
    train, test = [], []
    for c in clients:
        n = len(c.x)
        perm = rng.permutation(n)
        n_test = int(min(n - 1, max(1, math.floor(frac * n + 0.5))))
        te, tr = np.sort(perm[:n_test]), np.sort(perm[n_test:])
        train.append(ClientData(x=c.x[tr], y=np.asarray(c.y)[tr], name=c.name))
        test.append(ClientData(x=c.x[te], y=np.asarray(c.y)[te], name=c.name))
    return train, test


def task_cells(task):
    label, seed, cells, fraction, split, L, R = task
    from fedfcmsim.fcm import initialize_centers_from_data
    T = get_T()
    if T.LOCAL_STEPS != L or T.ROUNDS != R:
        raise RuntimeError("worker T3 constants differ from the task")
    T.M = T.fuzzifier_for(label)
    clients, k = T.build_clients(spec_for(T, label), seed)
    if split > 0:
        train, test = split_clients(T, clients, seed, split)
    else:
        train, test = clients, None
    pooled = np.vstack([c.x for c in train])
    init = initialize_centers_from_data(pooled, k, random_state=T.SEED_OFFSET + seed + 77)
    dim = int(pooled.shape[1])
    rows = []
    for mp, g, pe in cells:
        t0 = time.time()
        if fraction >= 1.0:
            V, ev, n_rounds = run_cell_counted(T, train, init, mp, g, pe)   # T3's run_cell, unchanged, default L/R
            n_part = len(train)
        else:
            V, ev, ex = run_cell_pp(T, train, init, mp, g, pe, fraction, T.SEED_OFFSET + seed + 555)
            n_part, n_rounds = ex["n_part"], ex["n_rounds"]
        row = {"dataset": label, "seed": seed, "method": "cells", "cell": cell_name((mp, g, pe)), "mass_point": mp,
               "gate": g, "personalize": pe, "L": L, "rounds": R, "n_rounds": n_rounds, "participation": fraction,
               "n_part": n_part, "n_clients": len(train), "split": split, "m": T.M, "n_clusters": k, "dim": dim,
               "upload_numbers": upload_numbers("cells", n_rounds, n_part, k, dim),
               "no_earlystop": int(T.NO_EARLYSTOP)}
        row.update(eval_metrics(T, train, V, ev))
        if test is not None:
            row.update(heldout_metrics(T, train, test, ev))
            row["n_test"] = int(sum(len(c.x) for c in test))
            row["min_client_n_test"] = int(min(len(c.x) for c in test))
            row["n_clients_test_le5"] = int(sum(len(c.x) <= 5 for c in test))
        row["runtime_s"] = round(time.time() - t0, 3)
        rows.append(row)
    return rows


def run_cell_counted(T, clients, init, mp, g, pe):
    """T.run_cell(clients, init, mp, g, pe) -- T3's function with its default L/R, unchanged -- plus the number of
    rounds it executed. run_cell looks up GATE_FNS[gate] once per call and calls it exactly once per client per
    round (record_gates=False), so rounds = gate calls / n_clients. The counting wrapper returns the gate's value
    untouched (bit-for-bit identical results, T10_selftest i)."""
    orig = T.GATE_FNS[g]
    calls = [0]

    def counted(x, V, ubar):
        calls[0] += 1
        return orig(x, V, ubar)
    T.GATE_FNS[g] = counted
    try:
        V, ev, _ = T.run_cell(clients, init, mp, g, pe)
    finally:
        T.GATE_FNS[g] = orig
    n_rounds, rem = divmod(calls[0], len(clients))
    if rem:
        raise RuntimeError(f"gate called {calls[0]} times for {len(clients)} clients")
    return V, ev, n_rounds


def upload_numbers(method: str, n_rounds: int, n_part: int, c: int, d: int):
    """Numbers uploaded client -> server over a run (decision-analysis criterion of the §3 addendum). Convention:
    T3 cells: every participating client uploads its c x d local prototypes and the c aggregation weights
    (beta_p * mass * gate; personalized offsets stay local) = c(d+1) per client per round; SC-FFCM: Delta_V and
    Delta_c, 2cd per client per round (fedfcmsim.federated.scffcm). Downlink is not counted. -1 if unknown."""
    if n_rounds is None or n_rounds < 0:
        return -1
    per = {"cells": c * (d + 1), "scffcm": 2 * c * d}.get(method)
    return -1 if per is None else int(n_rounds * n_part * per)


# ---------------------------------------------------------------------------------------------------------------
# SC-FFCM, Pedrycz, centralized FCM
# ---------------------------------------------------------------------------------------------------------------
def _run_scffcm(T, clients, init, L, R, a, b, cf, seed):
    from fedfcmsim.federated import scffcm
    try:
        with np.errstate(all="ignore"):
            res = scffcm(clients, init, rounds=R, local_steps=L, m=T.M, eta_l=a, eta_g=b, cfraction=cf,
                         seed=T.SEED_OFFSET + seed + 555, track_objective=False)
        return res.centers, res.n_rounds, ""
    except Exception as e:  # noqa: BLE001
        return None, -1, f"{type(e).__name__}: {e}"


def _run_pedrycz(T, clients, init, L, R, alpha):
    from fedfcmsim.federated import pedrycz_gradient_fcm
    try:
        with np.errstate(all="ignore"):
            res = pedrycz_gradient_fcm(clients, init, rounds=R, local_steps=L, m=T.M, alpha=alpha,
                                       track_objective=False)
        return res.centers, res.n_rounds, ""
    except Exception as e:  # noqa: BLE001
        return None, -1, f"{type(e).__name__}: {e}"


def task_method(task):
    """One evaluation run of scffcm / pedrycz / cfcm on (dataset, seed)."""
    method, label, seed, p = task
    T, clients, k, X, init = setup(label, seed)
    t0 = time.time()
    row = {"dataset": label, "seed": seed, "method": method, "m": T.M, "split": 0.0, "n_clients": len(clients),
           "n_clusters": k, "dim": int(X.shape[1])}
    if method == "scffcm":
        V, nr, err = _run_scffcm(T, clients, init, p["L"], p["rounds"], p["eta_l"], p["eta_g"], p["cfraction"], seed)
        n_part = max(1, int(len(clients) * p["cfraction"]))
        row.update(L=p["L"], rounds=p["rounds"], eta_l=p["eta_l"], eta_g=p["eta_g"], cfraction=p["cfraction"],
                   participation=p["cfraction"], n_part=n_part, n_rounds=nr,
                   upload_numbers=upload_numbers("scffcm", nr, n_part, k, int(X.shape[1])), error=err)
    elif method == "pedrycz":
        V, nr, err = _run_pedrycz(T, clients, init, p["L"], p["rounds"], p["alpha"])
        row.update(L=p["L"], rounds=p["rounds"], alpha=p["alpha"], participation=1.0, n_part=len(clients),
                   n_rounds=nr, error=err)
    elif method == "cfcm":
        from fedfcmsim.fcm import fcm
        res = fcm(X, k, init_centers=init, m=T.M, max_iter=p["max_iter"])   # == T6: fcm(pooled, k, init_centers=init, m=M)
        V, err = res.centers, ""
        row.update(max_iter=p["max_iter"], n_iter=res.n_iter, participation=1.0, error=err)
    else:
        raise ValueError(method)
    row.update(eval_metrics(T, clients, V, [V] * len(clients) if V is not None else [np.full(1, np.nan)]))
    row["runtime_s"] = round(time.time() - t0, 3)
    return [row]


def task_cal_ref(task):
    """Centralized reference objective Jc of the calibration rule (T9b.calibrate)."""
    label, cal_seed = task
    from fedfcmsim.fcm import fcm
    T, clients, k, X, init = setup(label, cal_seed)
    return [{"dataset": label, "m": T.M, "Jc": pooled_obj(X, fcm(X, k, m=T.M, init_centers=init, max_iter=300).centers, T.M)}]


def task_cal_point(task):
    method, label, cal_seed, L, R, params = task
    T, clients, k, X, init = setup(label, cal_seed, cache=True)
    t0 = time.time()
    if method == "scffcm":
        V, nr, err = _run_scffcm(T, clients, init, L, R, params[0], params[1], 1.0, cal_seed)
    else:
        V, nr, err = _run_pedrycz(T, clients, init, L, R, params[0])
    finite = V is not None and bool(np.all(np.isfinite(V)))
    with np.errstate(all="ignore"):
        J = pooled_obj(X, V, T.M) if finite else math.inf
    return [{"dataset": label, "params": list(params), "finite": finite, "J": J, "n_rounds": nr, "error": err,
             "runtime_s": round(time.time() - t0, 3)}]


def select_by_rule(grid, points, Jc):
    """T9b.calibrate: non-diverging = finite and J <= 5 Jc; pick min |J-Jc|/Jc; ties -> first in grid order."""
    best, n_stable, detail = None, 0, []
    for params in grid:
        pt = points[tuple(params)]
        ok = pt["finite"] and pt["J"] <= 5 * Jc
        gap = abs(pt["J"] - Jc) / Jc if ok else math.inf
        detail.append((params, pt, ok, gap))
        if ok:
            n_stable += 1
            if best is None or gap < best[1]:
                best = (params, gap)
    return best, n_stable, detail


def edge_label(v, axis_values) -> str:
    if len(axis_values) == 1:
        return "single"
    if v == min(axis_values):
        return "low"
    if v == max(axis_values):
        return "high"
    return "interior"


# ---------------------------------------------------------------------------------------------------------------
# E-F audit
# ---------------------------------------------------------------------------------------------------------------
def distinct(V, tol=1e-3):
    """Verbatim from paper2/compute_degeneracy.py."""
    k = len(V); lab = list(range(k))

    def f(i):
        while lab[i] != i:
            i = lab[i]
        return i
    for i in range(k):
        for j in range(i + 1, k):
            if np.linalg.norm(V[i] - V[j]) < tol:
                lab[f(i)] = f(j)
    return len({f(i) for i in range(k)})


def entropy_norm(X, V, m, k) -> float:
    from fedfcmsim.fcm import predict_membership
    u = predict_membership(X, V, m=m)
    return float(-(u * np.log(np.maximum(u, 1e-300))).sum(1).mean() / np.log(k))


def task_audit(task):
    label, m, s, iters, restarts = task
    from sklearn.cluster import kmeans_plusplus
    from fedfcmsim.fcm import fcm, initialize_centers_from_data, predict_membership
    T = get_T()
    clients, k = T.build_clients(spec_for(T, label), s)     # partition random_state 1201+s (as compute_degeneracy)
    X = np.vstack([c.x for c in clients])
    y = np.concatenate([np.asarray(c.y) for c in clients])
    V0 = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + s + 77)
    base = {"dataset": label, "m": m, "seed": s, "n": X.shape[0], "d": X.shape[1], "c": k}

    def row(init, V, n_iter, t0, **extra):
        with np.errstate(all="ignore"):
            r = dict(base, init=init, distinct=distinct(V), entropy_norm=entropy_norm(X, V, m, k),
                     J=pooled_obj(X, V, m), n_iter=n_iter)
        r["entropy_norm_r6"] = round(r["entropy_norm"], 6)
        r.update(extra)
        r["runtime_s"] = round(time.time() - t0, 3)
        return r

    rows = []
    t0 = time.time()
    ra = fcm(X, k, m=m, init_centers=V0, max_iter=iters)                      # (a) paper init
    rows.append(row("paper", ra.centers, ra.n_iter, t0))
    t0 = time.time()
    best, dlist = None, []
    for r in range(restarts):                                                 # (b) k-means++ best of 10
        C, _ = kmeans_plusplus(X, k, random_state=T.SEED_OFFSET + s + 77 + r)
        rr = fcm(X, k, m=m, init_centers=C, max_iter=iters)
        J = pooled_obj(X, rr.centers, m)
        dlist.append(distinct(rr.centers))
        if best is None or J < best[0]:
            best = (J, r, rr)
    rows.append(row("kmeanspp_best10", best[2].centers, best[2].n_iter, t0, best_restart=best[1],
                    kpp_distinct_min=min(dlist), kpp_distinct_max=max(dlist)))
    t0 = time.time()
    labels = np.unique(y)                                                     # (c) oracle class means
    C = np.vstack([X[y == l].mean(axis=0) for l in labels])
    rc = fcm(X, k, m=m, init_centers=C, max_iter=iters)
    rows.append(row("oracle", rc.centers, rc.n_iter, t0, n_classes=len(labels)))
    t0 = time.time()                                                          # (d) federated lossless exchange

    def cen_step(V):
        u = predict_membership(X, V, m=m) ** m
        return (u.T @ X) / np.maximum(u.sum(0), 1e-300)[:, None]

    def fed_step(V):
        num = np.zeros_like(V); den = np.zeros(k)
        for c in clients:
            u = predict_membership(c.x, V, m=m) ** m
            num += u.T @ c.x; den += u.sum(0)
        T_ = V.copy(); ok = den > 1e-300; T_[ok] = num[ok] / den[ok, None]
        return T_
    Vc, Vf, maxgap = V0.copy(), V0.copy(), 0.0
    for _ in range(iters):
        Vc = cen_step(Vc); Vf = fed_step(Vf)
        maxgap = max(maxgap, float(np.abs(Vf - Vc).max()))
    rows.append(row("fed_lossless_pre1", Vf, iters, t0, maxgap_fed_vs_cen_iter=maxgap,
                    gap_fed_vs_paper_final=float(np.abs(Vf - ra.centers).max()),
                    distinct_cen_iter=distinct(Vc)))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------------------------------------------
def out_paths(args, tag=None):
    outdir = Path(args.outdir).resolve()
    tag = tag or args.tag
    return outdir / f"{tag}.csv", outdir / f"{tag}.meta.json"


def check_outdir(args):
    outdir = Path(args.outdir).resolve()
    if outdir != OUT_DIR and OUT_DIR not in outdir.parents:
        raise SystemExit(f"--outdir must be results/t10 or below it (got {outdir})")
    outdir.mkdir(parents=True, exist_ok=True)


def refuse_existing(args, *paths):
    for p in paths:
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists; refusing to overwrite (use --force)")


def write_csv(path: Path, rows: list[dict], first: list[str] | None = None):
    keys = list(first or [])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def write_meta(path: Path, args, extra: dict):
    meta = {"argv": sys.argv, "args": {k: v for k, v in vars(args).items() if k != "func"},
            "env": {k: v for k, v in os.environ.items() if k.startswith("FFCM_")},
            "host": socket.gethostname(), "python": platform.python_version(),
            "numpy": np.__version__, "cpu_count": os.cpu_count(), "code_sha256": code_sha256(),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **extra}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def parse_datasets(text: str | None, default=None) -> list[str]:
    if not text:
        return list(default or CONFIGS)
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        for d in DATASET_ALIASES.get(tok.lower(), [tok]):
            if d not in CONFIGS:
                raise SystemExit(f"unknown dataset '{d}'; known {CONFIGS} or aliases {sorted(DATASET_ALIASES)}")
            if d not in out:
                out.append(d)
    return out


def seed_list(args) -> list[int]:
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    fresh = [s for s in seeds if s in FRESH_SEEDS]
    if fresh and not args.allow_fresh:
        raise SystemExit(f"seeds {fresh} belong to the pre-registered fresh block 20-29; pass --allow-fresh "
                         "(only in the final full run)")
    return seeds


def run_pool(fn, tasks, workers, env, label=""):
    """Run tasks (each returns a list of rows) in a spawn pool whose workers get `env` before importing T3."""
    apply_env(env)
    rows, t0, n = [], time.time(), len(tasks)
    tasks = sorted(tasks, key=lambda t: -COST.get(_task_label(t), 1))
    if workers <= 1:
        _init_worker(env)
        for i, t in enumerate(tasks, 1):
            rows.extend(fn(t))
            print(f"[{label} {i}/{n}] {_task_label(t)} ({time.time()-t0:.0f}s)", flush=True)
        return rows, time.time() - t0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(env,)) as ex:
        futs = {ex.submit(fn, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            print(f"[{label} {i}/{n}] {_task_label(futs[fut])} ({time.time()-t0:.0f}s)", flush=True)
    return rows, time.time() - t0


def _task_label(t):
    for x in t:
        if isinstance(x, str) and x in CONFIGS:
            return x
    return ""


def placeholder_row(env, d, s, method, L, rounds, **extra) -> dict:
    """Row for a configuration with no stable calibrated setting: every metric NaN, n_rounds = -1, and the same
    descriptive columns as a real row (m = T3.fuzzifier_for(d), participation, split)."""
    apply_env(env)
    T = get_T()
    return {"dataset": d, "seed": s, "method": method, "L": L, "rounds": rounds, "n_rounds": -1,
            "m": T.fuzzifier_for(d), "split": 0.0, "upload_numbers": -1, "error": "no_stable_setting",
            **extra, **{k: float("nan") for k in METRIC_KEYS}}


def check_cal_seed(args) -> None:
    if args.cal_seed in FRESH_SEEDS and not args.allow_fresh:
        raise SystemExit(f"--cal-seed {args.cal_seed} belongs to the pre-registered fresh block 20-29; "
                         "pass --allow-fresh (never needed: calibration uses seed 100)")


def code_sha256() -> dict:
    """sha256 of the code that produced an output (recorded in every meta.json; checked by argcheck)."""
    import hashlib
    files = [HERE / "T10_review.py", HERE / "T3_design_space.py"] + sorted((EXP / "fedfcmsim").glob("*.py"))
    return {str(f.relative_to(EXP)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}


def load_steps(path: Path, L: int, rounds: int, key_cols=("eta_l", "eta_g")) -> dict:
    rows = list(csv.DictReader(path.open()))
    steps = {}
    for r in rows:
        if int(r["L"]) != L:
            continue
        cal_rounds = int(r["rounds"]) if r.get("rounds") not in (None, "") else 50  # t9b/t9c: R = 50 implicit
        if cal_rounds != rounds:
            raise SystemExit(f"{path.name}: calibrated at R={cal_rounds}, evaluation requested R={rounds}")
        if any(r[c] == "" for c in key_cols):
            steps[r["dataset"]] = None                     # no stable setting on the grid
        else:
            steps[r["dataset"]] = tuple(float(r[c]) for c in key_cols)
    return steps


# ---------------------------------------------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------------------------------------------
def cmd_cells(args):
    check_outdir(args)
    csv_path, meta_path = out_paths(args)
    refuse_existing(args, csv_path, meta_path)
    if not (0 < args.participation <= 1.0):
        raise SystemExit("--participation must be in (0, 1]")
    if not (0 <= args.split < 1.0):
        raise SystemExit("--split must be in [0, 1)")
    cells = parse_cells(args.cells)
    datasets = parse_datasets(args.datasets)
    seeds = seed_list(args)
    env = protocol_env(args.L, args.rounds, args.seed_start, no_earlystop=args.no_earlystop)
    tasks = [(d, s, cells, args.participation, args.split, args.L, args.rounds) for d in datasets for s in seeds]
    rows, wall = run_pool(task_cells, tasks, args.workers, env, "cells")
    order = {cell_name(c): i for i, c in enumerate(cells)}
    rows.sort(key=lambda r: (CONFIGS.index(r["dataset"]), r["seed"], order[r["cell"]]))
    first = ["dataset", "seed", "method", "cell", "mass_point", "gate", "personalize", "L", "rounds", "n_rounds",
             "participation", "n_part", "n_clients", "split", "m", "n_clusters", "dim", "upload_numbers"] + \
        METRIC_KEYS + (TEST_KEYS + ["n_test", "min_client_n_test", "n_clients_test_le5"] if args.split > 0 else [])
    write_csv(csv_path, rows, first)
    write_meta(meta_path, args, {"wall_s": wall, "n_rows": len(rows), "n_tasks": len(tasks),
                                 "cells": [cell_name(c) for c in cells], "datasets": datasets, "seeds": seeds})
    print(f"wrote {len(rows)} rows to {csv_path} in {wall:.0f}s")


def cmd_scffcm(args):
    check_outdir(args)
    csv_path, meta_path = out_paths(args)
    refuse_existing(args, csv_path, meta_path)
    datasets = parse_datasets(args.datasets)
    seeds = seed_list(args)
    if args.steps_csv:
        steps = load_steps(Path(args.steps_csv), args.L, args.rounds)
        missing = [d for d in datasets if d not in steps]
        if missing:
            raise SystemExit(f"{args.steps_csv} has no L={args.L} row for {missing}")
        unstable = [d for d in datasets if steps[d] is None]
        if unstable:
            print(f"NOTE: no stable calibrated setting for {unstable}; rows are written with error=no_stable_setting")
    else:
        if args.eta_l is None or args.eta_g is None:
            raise SystemExit("give --steps-csv or both --eta-l and --eta-g")
        steps = {d: (args.eta_l, args.eta_g) for d in datasets}
    env = protocol_env(args.L, args.rounds, args.seed_start)
    tasks, rows = [], []
    for d in datasets:
        for s in seeds:
            if steps[d] is None:
                rows.append(placeholder_row(env, d, s, "scffcm", args.L, args.rounds, cfraction=args.cfraction,
                                            participation=args.cfraction))
                continue
            tasks.append(("scffcm", d, s, {"L": args.L, "rounds": args.rounds, "eta_l": steps[d][0],
                                           "eta_g": steps[d][1], "cfraction": args.cfraction}))
    new, wall = run_pool(task_method, tasks, args.workers, env, "scffcm")
    rows += new
    rows.sort(key=lambda r: (CONFIGS.index(r["dataset"]), r["seed"]))
    first = ["dataset", "seed", "method", "L", "rounds", "n_rounds", "eta_l", "eta_g", "cfraction", "participation",
             "n_part", "n_clients", "split", "m", "n_clusters", "dim", "upload_numbers"] + METRIC_KEYS
    write_csv(csv_path, rows, first)
    write_meta(meta_path, args, {"wall_s": wall, "n_rows": len(rows), "steps": {d: steps[d] for d in datasets},
                                 "seeds": seeds})
    print(f"wrote {len(rows)} rows to {csv_path} in {wall:.0f}s")


def _parse_grid(text, named, n_axes):
    if text in named:
        return named[text]
    axes = [tuple(float(v) for v in part.split(",") if v.strip()) for part in text.split(";")]
    if len(axes) != n_axes:
        raise SystemExit(f"grid '{text}' must have {n_axes} ';'-separated axes or be one of {sorted(named)}")
    return tuple(axes)


def _calibrate(args, method):
    check_outdir(args)
    csv_path, meta_path = out_paths(args)
    grid_path = csv_path.with_name(csv_path.stem + "_grid.csv")
    refuse_existing(args, csv_path, meta_path, grid_path)
    check_cal_seed(args)
    datasets = parse_datasets(args.datasets)
    if method == "scffcm":
        axes = _parse_grid(args.grid, SCFFCM_GRIDS, 2)
        grid = [(a, b) for a in axes[0] for b in axes[1]]            # eta_l outer, eta_g inner (T9b order)
        names = ["eta_l", "eta_g"]
    else:
        axes = _parse_grid(args.grid, PEDRYCZ_GRIDS, 1)
        grid = [(a,) for a in axes[0]]
        names = ["alpha"]
    env = protocol_env(args.L, args.rounds, 0)
    tasks = [(method, d, args.cal_seed, args.L, args.rounds, g) for d in datasets for g in grid]
    ref_tasks = [(d, args.cal_seed) for d in datasets]
    refs, w1 = run_pool(task_cal_ref, ref_tasks, args.workers, env, "Jc")
    pts, w2 = run_pool(task_cal_point, tasks, args.workers, env, f"{method}-grid")
    Jc = {r["dataset"]: r["Jc"] for r in refs}
    mstar = {r["dataset"]: r["m"] for r in refs}
    summary, detail_rows = [], []
    for d in datasets:
        points = {tuple(p["params"]): p for p in pts if p["dataset"] == d}
        best, n_stable, detail = select_by_rule(grid, points, Jc[d])
        for params, pt, ok, gap in detail:
            detail_rows.append({"dataset": d, "L": args.L, "rounds": args.rounds, **dict(zip(names, params)),
                                "J": pt["J"], "Jc": Jc[d], "rel_obj_gap": gap if ok else "", "stable": int(ok),
                                "finite": int(pt["finite"]), "n_rounds": pt["n_rounds"], "error": pt["error"],
                                "runtime_s": pt["runtime_s"]})
        row = {"dataset": d, "L": args.L, "rounds": args.rounds, "m": mstar[d]}
        if best is None:
            row.update({n: "" for n in names}, rel_obj_gap="", on_edge="")
        else:
            row.update(dict(zip(names, best[0])), rel_obj_gap=best[1])
            edges = [edge_label(v, ax) for v, ax in zip(best[0], axes)]
            for n, e in zip(names, edges):
                row[f"edge_{n}"] = e
            row["on_edge"] = int(any(e in ("low", "high") for e in edges))
        row.update(n_stable=n_stable, n_grid=len(grid), Jc=Jc[d], cal_seed=args.cal_seed, grid=args.grid)
        summary.append(row)
    write_csv(grid_path, detail_rows)
    first = ["dataset", "L", "rounds", "m"] + names + ["rel_obj_gap", "n_stable", "n_grid"] + \
        [f"edge_{n}" for n in names] + ["on_edge"]
    write_csv(csv_path, summary, first)
    write_meta(meta_path, args, {"wall_s": w1 + w2, "grid": grid, "n_points": len(tasks)})
    print(f"wrote {csv_path} and {grid_path} in {w1 + w2:.0f}s")
    for r in summary:
        print({k: r.get(k) for k in first})


def cmd_scffcm_calibrate(args):
    _calibrate(args, "scffcm")


def cmd_pedrycz_calibrate(args):
    _calibrate(args, "pedrycz")


def cmd_pedrycz(args):
    check_outdir(args)
    csv_path, meta_path = out_paths(args)
    refuse_existing(args, csv_path, meta_path)
    datasets = parse_datasets(args.datasets)
    seeds = seed_list(args)
    if args.alpha_csv:
        steps = load_steps(Path(args.alpha_csv), args.L, args.rounds, key_cols=("alpha",))
        missing = [d for d in datasets if d not in steps]
        if missing:
            raise SystemExit(f"{args.alpha_csv} has no L={args.L} row for {missing}")
    elif args.alpha is not None:
        steps = {d: (args.alpha,) for d in datasets}
    else:
        raise SystemExit("give --alpha-csv or --alpha")
    env = protocol_env(args.L, args.rounds, args.seed_start)
    rows, tasks = [], []
    for d in datasets:
        for s in seeds:
            if steps[d] is None:
                rows.append(placeholder_row(env, d, s, "pedrycz", args.L, args.rounds, participation=1.0))
            else:
                tasks.append(("pedrycz", d, s, {"L": args.L, "rounds": args.rounds, "alpha": steps[d][0]}))
    new, wall = run_pool(task_method, tasks, args.workers, env, "pedrycz")
    rows += new
    rows.sort(key=lambda r: (CONFIGS.index(r["dataset"]), r["seed"]))
    write_csv(csv_path, rows, ["dataset", "seed", "method", "L", "rounds", "n_rounds", "alpha", "participation",
                               "n_part", "n_clients", "split", "m", "n_clusters", "dim"] + METRIC_KEYS)
    write_meta(meta_path, args, {"wall_s": wall, "n_rows": len(rows), "seeds": seeds})
    print(f"wrote {len(rows)} rows to {csv_path} in {wall:.0f}s")


def cmd_cfcm(args):
    check_outdir(args)
    csv_path, meta_path = out_paths(args)
    refuse_existing(args, csv_path, meta_path)
    datasets = parse_datasets(args.datasets)
    seeds = seed_list(args)
    env = protocol_env(5, 50, args.seed_start)
    tasks = [("cfcm", d, s, {"max_iter": args.max_iter}) for d in datasets for s in seeds]
    rows, wall = run_pool(task_method, tasks, args.workers, env, "cfcm")
    rows.sort(key=lambda r: (CONFIGS.index(r["dataset"]), r["seed"]))
    write_csv(csv_path, rows, ["dataset", "seed", "method", "max_iter", "n_iter", "participation", "n_clients", "split",
                               "m", "n_clusters", "dim"] + METRIC_KEYS)
    write_meta(meta_path, args, {"wall_s": wall, "n_rows": len(rows), "seeds": seeds})
    print(f"wrote {len(rows)} rows to {csv_path} in {wall:.0f}s")


def cmd_audit(args):
    check_outdir(args)
    csv_path, meta_path = out_paths(args)
    ms_path = csv_path.with_name(csv_path.stem + "_mstar.csv")
    refuse_existing(args, csv_path, meta_path, ms_path)
    datasets = parse_datasets(args.datasets, default=CONFIGS)   # all 14 (the 8 real ones = compute_degeneracy.py)
    ms = [float(v) for v in args.ms.split(",")]
    seeds = seed_list(args)
    env = protocol_env(5, 50, args.seed_start)
    tasks = [(d, m, s, args.iters, args.restarts) for d in datasets for m in ms for s in seeds]
    rows, wall = run_pool(task_audit, tasks, args.workers, env, "audit")
    inits = ["paper", "kmeanspp_best10", "oracle", "fed_lossless_pre1"]
    rows.sort(key=lambda r: (CONFIGS.index(r["dataset"]), -r["m"], r["seed"], inits.index(r["init"])))
    write_csv(csv_path, rows, ["dataset", "m", "seed", "init", "n", "d", "c", "distinct", "entropy_norm",
                               "entropy_norm_r6", "J", "n_iter"])
    # the label-free m* rule under every init: largest m whose run keeps all c prototypes distinct on every seed
    apply_env(env)
    T = get_T()
    ms_rows = []
    for d in datasets:
        for ini in inits:
            ok_ms = [m for m in ms if all(r["distinct"] == r["c"] for r in rows
                                          if r["dataset"] == d and r["init"] == ini and r["m"] == m)]
            mstar = max(ok_ms) if ok_ms else ""
            paper = T.MSTAR.get(d, 2.0)
            ms_rows.append({"dataset": d, "init": ini, "mstar": mstar, "paper_mstar": paper,
                            "agrees": int(mstar == paper), "n_seeds": len(seeds),
                            "ms_all_distinct": ";".join(str(m) for m in ok_ms)})
    write_csv(ms_path, ms_rows)
    write_meta(meta_path, args, {"wall_s": wall, "n_rows": len(rows), "seeds": seeds})
    print(f"wrote {len(rows)} rows to {csv_path} and {ms_path} in {wall:.0f}s")
    for r in ms_rows:
        print(r)


# ---------------------------------------------------------------------------------------------------------------
# compare: bit-for-bit checks against the prior CSVs
# ---------------------------------------------------------------------------------------------------------------
T3D = RESULTS / "t3_design_space"
# Prior design-space CSVs by (L, rounds). main.csv predates the m* switch: it is valid only where m* = 2.
# kind "noES": produced with FFCM_NO_EARLYSTOP=1 (established by T10_selftest ii'): compared in ALL fields with new
# rows that also used --no-earlystop, and ADVISORY (acc only, reported, never fatal) with new early-stop rows of the
# personalized cells, because those are a different stopping protocol. Their non-personalized rows are strict.
PRIOR_CELLS = {
    (5, 50): [(T3D / "per_seed_metrics_main.csv", "m2only"), (T3D / "per_seed_metrics__mstar.csv", "any"),
              (T3D / "per_seed_metrics__confirm.csv", "any"),
              (RESULTS / "t6_baselines" / "per_seed_metrics__mstar.csv", "t6"),
              (T3D / "per_seed_metrics__T9_noES.csv", "noES"), (T3D / "per_seed_metrics__T9_noES_confirm.csv", "noES")],
    (1, 50): [(T3D / "per_seed_metrics__L1all.csv", "any"), (T3D / "per_seed_metrics__T9_L1_gf.csv", "noES"),
              (T3D / "per_seed_metrics__T9_L1_confirm.csv", "noES")],
}
T3_CELL_NAMES = {cell_name(c) for c in T3_CELLS}
CMP_FIELDS = ["min_dist", "mean_acc", "worst_acc", "fcm_obj"]
ADVISORY_TAG = " [prior noES, new early-stop: acc only, advisory]"


def _feq(a, b) -> bool:
    fa, fb = float(a), float(b)
    return fa == fb or (math.isnan(fa) and math.isnan(fb))


def compare_rows(new_rows, prior_rows, key_fn, fields, source, strict=True):
    idx = {}
    for r in prior_rows:
        idx.setdefault(key_fn(r), r)
    out = []
    for r in new_rows:
        k = key_fn(r)
        if k not in idx:
            continue
        p = idx[k]
        for f in fields:
            if f not in p or p[f] == "" or f not in r:
                continue
            out.append({"source": source, "key": "|".join(map(str, k)), "field": f, "prior": p[f], "new": r[f],
                        "match": int(_feq(p[f], r[f])), "strict": int(strict)})
    return out


def _uncovered(scope_prior, new_rows, key_fn, source):
    """In-scope prior keys that have no row in the new CSV (key drift, missing tasks)."""
    have = {key_fn(r) for r in new_rows}
    return [{"source": source, "key": "|".join(map(str, key_fn(p)))} for p in scope_prior if key_fn(p) not in have]


def _mstar_table() -> dict:
    """T3.MSTAR (a constant table; importing T3 here only reads it)."""
    os.environ["FFCM_MSTAR"] = "1"
    import T3_design_space as T
    return dict(T.MSTAR)


def _requested_datasets(meta: dict | None):
    """The datasets the new CSV was ASKED for (meta.json); None = all 14, so prior rows of any label are in scope
    and a label drift shows up as uncovered prior rows."""
    ds = (meta or {}).get("datasets") or (meta or {}).get("args", {}).get("datasets")
    if isinstance(ds, str):
        ds = parse_datasets(ds)
    if not ds or set(ds) >= set(CONFIGS):
        return None
    return set(ds)


def compare_cells(rows, L, R, req=None):
    mstar = _mstar_table()
    res, unc = [], []
    rows = [r for r in rows if float(r.get("participation", 1)) == 1.0 and float(r.get("split", 0)) == 0.0
            and int(r["L"]) == L and int(r["rounds"]) == R]
    seeds = {int(r["seed"]) for r in rows}
    cells = {r["cell"] for r in rows}
    new_noes = any(int(r.get("no_earlystop", 0)) for r in rows)
    key = lambda r: (r["dataset"], int(r["seed"]), r["cell"])  # noqa: E731
    for path, kind in PRIOR_CELLS.get((L, R), []):
        prior = list(csv.DictReader(path.open()))
        if kind == "m2only":
            prior = [p for p in prior if mstar.get(p["dataset"], 2.0) == 2.0]
        if kind == "t6":
            prior = [dict(p, cell=p["method"]) for p in prior if p["method"] in T3_CELL_NAMES]
        src = f"{path.parent.name}/{path.name}"
        if kind != "noES" and new_noes:
            continue                     # these priors used T3's early stopping; a --no-earlystop run is not comparable
        scope = [p for p in prior if int(p["seed"]) in seeds and p["cell"] in cells
                 and (req is None or p["dataset"] in req)]
        if kind == "noES":
            pers_on = lambda r: r.get("personalize", r["cell"].rsplit("_", 1)[-1]) == "on"  # noqa: E731
            off = [r for r in rows if not pers_on(r)]
            same = [r for r in rows if pers_on(r) and int(r.get("no_earlystop", 0)) == 1]
            diff = [r for r in rows if pers_on(r) and int(r.get("no_earlystop", 0)) == 0]
            res += compare_rows(off, prior, key, CMP_FIELDS, src)
            res += compare_rows(same, prior, key, CMP_FIELDS, src + " [noES both]")
            res += compare_rows(diff, prior, key, ["mean_acc", "worst_acc"], src + ADVISORY_TAG, strict=False)
        else:
            res += compare_rows(rows, prior, key, CMP_FIELDS, src)
        unc += _uncovered(scope, rows, key, src)
    return res, unc


def compare_scffcm(rows, req=None):
    res, unc = [], []
    cand = [r for r in rows if int(r["rounds"]) == 50 and float(r["cfraction"]) == 1.0 and r.get("eta_l", "") != ""]
    have_dls = {(r["dataset"], int(r["L"]), int(r["seed"])) for r in cand}
    for name in ("t9b_scffcm_fair.csv", "t9c_scffcm_fair.csv"):
        prior = list(csv.DictReader((RESULTS / name).open()))
        key = lambda r: (r["dataset"], int(r["L"]), int(r["seed"]), float(r["eta_l"]), float(r["eta_g"]))  # noqa: E731
        res += compare_rows(cand, prior, key, CMP_FIELDS, name)
        # scope: the prior runs at the same (dataset, L, seed); a different step size (wrong calibration file) or a
        # missing run shows up as uncovered
        scope = [p for p in prior if (p["dataset"], int(p["L"]), int(p["seed"])) in have_dls]
        unc += _uncovered(scope, cand, key, name)
    return res, unc


def compare_cfcm(rows, req=None):
    prior = [p for p in csv.DictReader((RESULTS / "t6_baselines" / "per_seed_metrics__mstar.csv").open())
             if p["method"] == "centralized_fcm"]
    rows = [r for r in rows if int(r["max_iter"]) == 150]
    key = lambda r: (r["dataset"], int(r["seed"]))  # noqa: E731
    seeds = {int(r["seed"]) for r in rows}
    src = "t6_baselines/per_seed_metrics__mstar.csv"
    scope = [p for p in prior if int(p["seed"]) in seeds and (req is None or p["dataset"] in req)]
    return (compare_rows(rows, prior, key, ["min_dist", "mean_acc", "worst_acc"], src),
            _uncovered(scope, rows, key, src))


def compare_audit(rows, req=None):
    prior = list(csv.DictReader((PAPER2 / "figures" / "degeneracy_data.csv").open()))
    rows = [dict(r, entropy_norm=r["entropy_norm_r6"]) for r in rows if r["init"] == "paper" and int(r["n_iter"]) >= 0]
    key = lambda r: (r["dataset"], float(r["m"]), int(r["seed"]))  # noqa: E731
    seeds, ms = {int(r["seed"]) for r in rows}, {float(r["m"]) for r in rows}
    src = "paper2/figures/degeneracy_data.csv"
    scope = [p for p in prior if int(p["seed"]) in seeds and float(p["m"]) in ms and (req is None or p["dataset"] in req)]
    return (compare_rows(rows, prior, key, ["distinct", "entropy_norm", "n", "d", "c"], src),
            _uncovered(scope, rows, key, src))


def run_compare_full(kind, csv_file: Path):
    """(value rows, uncovered in-scope prior keys)."""
    rows = list(csv.DictReader(csv_file.open()))
    meta_path = csv_file.with_name(csv_file.stem + ".meta.json")
    req = _requested_datasets(json.loads(meta_path.read_text()) if meta_path.exists() else None)
    if kind == "cells":
        out, unc = [], []
        for L, R in sorted({(int(r["L"]), int(r["rounds"])) for r in rows}):
            a, b = compare_cells(rows, L, R, req)
            out += a
            unc += b
        return out, unc
    return {"scffcm": compare_scffcm, "cfcm": compare_cfcm, "audit": compare_audit}[kind](rows, req)


def run_compare(kind, csv_file: Path):
    return run_compare_full(kind, csv_file)[0]


def cmd_compare(args):
    check_outdir(args)
    src = Path(args.csv).resolve()
    out_path = Path(args.outdir).resolve() / f"{args.tag}.csv"
    refuse_existing(args, out_path)
    res, unc = run_compare_full(args.kind, src)
    write_csv(out_path, res, ["source", "key", "field", "prior", "new", "match", "strict"])
    strict = [r for r in res if r["strict"]]
    adv = [r for r in res if not r["strict"]]
    n_bad = sum(1 for r in strict if not r["match"])
    n_adv_bad = sum(1 for r in adv if not r["match"])
    runs = {(r["source"], r["key"]) for r in strict}
    runs_adv = {(r["source"], r["key"]) for r in adv}
    n_acc = sum(1 for r in strict if r["field"] in ("mean_acc", "distinct"))
    for r in [r for r in strict if not r["match"]][:20]:
        print("  MISMATCH", r)
    for r in [r for r in adv if not r["match"]][:20]:
        print("  ADVISORY (acc differs across stopping rules; not fatal)", r)
    for u in unc[:20]:
        print("  UNCOVERED prior row", u)
    fail = []
    if n_bad:
        fail.append(f"{n_bad} strict mismatches")
    if not runs:
        fail.append("0 overlapping runs")
    if unc:
        fail.append(f"{len(unc)} in-scope prior rows not covered")
    if args.min_runs is not None and len(runs) < args.min_runs:
        fail.append(f"{len(runs)} strict overlapping runs < --min-runs {args.min_runs}")
    status = "FAIL: " + "; ".join(fail) if fail else "OK"
    print(f"compare {args.kind} {src.name}: {len(runs)} strict overlapping runs ({len(strict)} values, {n_acc} "
          f"acc/distinct values), {n_bad} strict mismatches; {len(unc)} uncovered prior rows; advisory: "
          f"{len(runs_adv)} runs, {n_adv_bad} acc differences; {status}")
    if fail:
        sys.exit(1)


def cmd_argcheck(args):
    """Exit 0 iff <meta> records the same command (same parsed arguments except --workers/--force; same resolved
    outdir); code-hash differences are reported but not fatal (they do not change a finished CSV)."""
    meta = json.loads(Path(args.meta).read_text())
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    new = vars(build_parser().parse_args(cmd))
    old = dict(meta.get("args", {}))
    ignore = {"func", "workers", "force"}
    diffs = []
    for k in sorted((set(new) | set(old)) - ignore):
        a, b = old.get(k), new.get(k)
        if k == "outdir":
            a, b = str(Path(str(a)).resolve()), str(Path(str(b)).resolve())
        if k in ("steps_csv", "alpha_csv") and a and b:
            a, b = str(Path(str(a)).resolve()), str(Path(str(b)).resolve())
        if str(a) != str(b):
            diffs.append(f"{k}: recorded {a!r}, now {b!r}")
    code_now, code_then = code_sha256(), meta.get("code_sha256")
    if code_then and code_then != code_now:
        changed = sorted(k for k in set(code_now) | set(code_then) if code_now.get(k) != code_then.get(k))
        print(f"NOTE: code changed since this output was written: {changed}")
    if diffs:
        print("ARGUMENTS DIFFER from the recorded run:\n  " + "\n  ".join(diffs))
        sys.exit(1)
    print("same arguments")


# ---------------------------------------------------------------------------------------------------------------
# analyze: the pre-registered statistics (frozen before the fresh block is run; see T10_run_all.sh)
# ---------------------------------------------------------------------------------------------------------------
BLOCKS = {"s00-09": tuple(range(0, 10)), "s10-19": tuple(range(10, 20)), "s20-29": tuple(range(20, 30))}
ZERO_TOL = 1e-12          # |d| <= ZERO_TOL counts as a zero difference (dropped, Wilcoxon's rule); as in T7-T9
TIE_DECIMALS = 12         # |d| rounded to 12 decimals before mid-ranking, so float noise does not break exact ties
ALPHA = 0.05
HIGHER_BETTER = {"mean_acc": True, "worst_acc": True, "gacc": True, "fcm_obj": False, "upload_numbers": False,
                 "test_fcm_obj": False, "test_gacc_trmap": True, "test_mean_acc_trmap": True,
                 "test_worst_acc_trmap": True, "test_gacc": True, "test_mean_acc_refit": True,
                 "test_worst_acc_refit": True}

# Where the plan writes each arm (T10 source). An arm is "L<L>:<cell>" (T3 cells, R=50, full participation, no split),
# "L1noES:<cell>" (the --no-earlystop runs of GF at L=1), "L1R250:<cell>" (E-D), "SC:L<L>" (SC-FFCM at the steps
# calibrated on seed 100, R=50, cfraction 1) or "SC:L1R250" (E-D calibration, R=250).
T10_SOURCES = [
    ("cells", "", ["cells16_L5_s00-09", "cells16_L5_s10-19", "FRESH_cells16_L5_s20-29",
                   "cells16_L1_s00-09", "cells16_L1_s10-19", "FRESH_cells16_L1_s20-29"]),
    ("cells", "noES", ["gf_L1_noES_s00-19", "FRESH_gf_L1_noES_s20-29"]),
    ("cells", "R250", ["ed_cells_L1_R250_s10-19", "FRESH_ed_cells_L1_R250_s20-29"]),
    ("scffcm", "", ["scffcm_cal_L1_s00-19", "scffcm_cal_L5_s00-19", "scffcm_cal_L50_s00-19",
                    "FRESH_scffcm_cal_L1_s20-29", "FRESH_scffcm_cal_L5_s20-29", "FRESH_scffcm_cal_L50_s20-29"]),
    ("scffcm", "R250", ["ed_scffcm_L1_R250_s10-19", "FRESH_ed_scffcm_L1_R250_s20-29"]),
]
# The same arms from the PRIOR CSVs (to test the analysis on the earlier blocks before the fresh block exists).
# Caveat: the prior L=1 GF rows were run without early stopping, so there "L1:post_footprint_on" = "L1noES:...".
PRIOR_SOURCES = [
    ("cells", 5, T3D / "per_seed_metrics_main.csv", "m2only"), ("cells", 5, T3D / "per_seed_metrics__mstar.csv", ""),
    ("cells", 5, T3D / "per_seed_metrics__confirm.csv", ""),
    ("cells", 1, T3D / "per_seed_metrics__L1all.csv", ""), ("cells", 1, T3D / "per_seed_metrics__T9_L1_gf.csv", "noES"),
    ("cells", 1, T3D / "per_seed_metrics__T9_L1_confirm.csv", "noES"),
    ("scffcm", None, RESULTS / "t9b_scffcm_fair.csv", ""), ("scffcm", None, RESULTS / "t9c_scffcm_fair.csv", ""),
]

# §1: primary hypotheses. kind 'pair': d = a - b; kind 'inter': d = sum(coef * arm). expect: 'a' (a better / contrast
# positive), 'b' (b better), None (reported as counts only). family: the configurations of the Holm family.
H_PRIMARY = [
    dict(id="H1", kind="pair", a="L5:pre_mass_off", b="L5:post_footprint_on", expect="a"),
    dict(id="H2", kind="pair", a="L5:pre_mass_off", b="L5:post_none_off", expect="a"),
    dict(id="H3", kind="inter", terms=[(1, "L5:pre_none_off"), (-1, "L5:post_none_off"),
                                       (-1, "L5:pre_footprint_off"), (1, "L5:post_footprint_off")], expect="a"),
    dict(id="H4", kind="pair", a="L1:pre_none_off", b="L1:post_none_off", expect="a"),
    dict(id="H5", kind="pair", a="L1:pre_mass_off", b="L1noES:post_footprint_on", expect="a",
         note="GF at L=1 run for the full 50 rounds (no early stop), the protocol of the earlier L=1 GF blocks; made primary before the fresh run (pre-registration addendum 2)"),
    dict(id="H6", kind="pair", a="L1:pre_mass_off", b="SC:L1", expect="a"),
    dict(id="H7", kind="pair", a="L1:pre_mass_off", b="SC:L5", expect="a"),
    dict(id="H8", kind="pair", a="L5:pre_mass_off", b="SC:L5", expect="b"),
    dict(id="H9", kind="pair", a="L5:post_footprint_on", b="L5:post_footprint_off", expect="none-better"),
]
H_SECONDARY = [
    dict(id="H5-ES", kind="pair", a="L1:pre_mass_off", b="L1:post_footprint_on", expect="a",
         note="H5 against GF at L=1 WITH T3's default early stop; declared secondary before the fresh run"),
    dict(id="S-L50", kind="pair", a="L1:pre_mass_off", b="SC:L50", expect="a", note="12 non-MNIST configurations"),
    dict(id="S-PRE-SC1", kind="pair", a="L1:pre_none_off", b="SC:L1", expect=None),
    dict(id="S-PRE-SC5", kind="pair", a="L1:pre_none_off", b="SC:L5", expect=None),
    dict(id="ED-mass", kind="pair", a="L1R250:pre_mass_off", b="SC:L1R250", expect=None, note="E-D, R=250"),
    dict(id="ED-none", kind="pair", a="L1R250:pre_none_off", b="SC:L1R250", expect=None, note="E-D, R=250"),
]
EA_GATES = ["footprint", "mass", "fpmask", "massmask"]
DECISION_ACTIONS = [("F-FCM (post_none_off, L=5)", "L5:post_none_off"), ("GF-PFedFCM (L=5)", "L5:post_footprint_on"),
                    ("ungated PRE (L=1)", "L1:pre_none_off"), ("pre_mass_off (L=1)", "L1:pre_mass_off"),
                    ("pre_mass_off (L=5)", "L5:pre_mass_off"), ("SC-FFCM (L=5)", "SC:L5"),
                    ("SC-FFCM (L=1, R=50)", "SC:L1")]
DECISION_CRITERIA = ["mean_acc", "worst_acc", "gacc", "fcm_obj", "upload_numbers"]
RELATIVE_REGRET = {"fcm_obj", "upload_numbers"}    # scale differs across configurations: regret relative to the best


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_arms_t10(t10_dir: Path) -> tuple[dict, list]:
    """table[(arm, dataset, seed)] = row (dict of str) from the plan's T10 CSVs; missing files are listed."""
    table, missing = {}, []
    for kind, flavour, tags in T10_SOURCES:
        for tag in tags:
            path = t10_dir / f"{tag}.csv"
            if not path.exists():
                missing.append(tag)
                continue
            for r in csv.DictReader(path.open()):
                if _num(r.get("participation", 1)) != 1.0 or _num(r.get("split", 0)) != 0.0:
                    continue
                L, R = int(r["L"]), int(r["rounds"])
                if kind == "cells":
                    if flavour == "noES":
                        if int(r.get("no_earlystop", 0)) != 1:
                            continue
                        arm = f"L{L}noES:{r['cell']}"
                    elif flavour == "R250":
                        arm = f"L{L}R{R}:{r['cell']}"
                    else:
                        if R != 50 or int(r.get("no_earlystop", 0)):
                            continue
                        arm = f"L{L}:{r['cell']}"
                else:
                    if _num(r.get("cfraction", 1)) != 1.0:
                        continue
                    arm = f"SC:L{L}" if R == 50 else f"SC:L{L}R{R}"
                table.setdefault((arm, r["dataset"], int(r["seed"])), r)
    return table, missing


def load_arms_prior() -> tuple[dict, list]:
    mstar = _mstar_table()
    table = {}
    for kind, L, path, flag in PRIOR_SOURCES:
        for r in csv.DictReader(path.open()):
            if flag == "m2only" and mstar.get(r["dataset"], 2.0) != 2.0:
                continue
            if kind == "cells":
                arms = [f"L{L}:{r['cell']}"]
                if flag == "noES" and r["cell"].endswith("_on"):
                    arms.append(f"L{L}noES:{r['cell']}")
            else:
                arms = [f"SC:L{int(r['L'])}"]
            for arm in arms:
                table.setdefault((arm, r["dataset"], int(r["seed"])), r)
    return table, []


def wilcoxon_exact(d) -> dict:
    """Two-sided exact Wilcoxon signed-rank test (§3): zeros (|d| <= ZERO_TOL) dropped, mid-ranks of |d| (rounded to
    TIE_DECIMALS; +-inf ranks last), p = P(|W+ - E| >= |w+ - E|) under the 2^n' equally likely sign assignments.
    The null distribution is built by the convolution recursion over the (doubled, hence integer) ranks, which counts
    exactly the same 2^n' assignments as explicit enumeration (T10_selftest xii checks both against each other).
    Returns n_eff (= n'), w_plus, p (1.0 when n' = 0)."""
    from scipy.stats import rankdata
    d = np.asarray([x for x in d if not math.isnan(x)], float)
    d = d[np.abs(d) > ZERO_TOL]
    n = len(d)
    if n == 0:
        return {"n_eff": 0, "w_plus": float("nan"), "p": 1.0}
    a = np.abs(d)
    a = np.where(np.isfinite(a), np.round(a, TIE_DECIMALS), np.inf)
    ranks = rankdata(a)                                   # average ranks for ties (mid-ranks)
    r2 = np.rint(2 * ranks).astype(np.int64)             # mid-ranks are multiples of 1/2
    tot = int(r2.sum())
    dist = np.zeros(tot + 1, dtype=np.int64)
    dist[0] = 1
    for r in r2:
        new = dist.copy()
        new[r:] += dist[:-r] if r > 0 else dist
        dist = new
    w2 = int(r2[d > 0].sum())
    dev = abs(2 * w2 - tot)                               # |2*(2W+) - tot| = 2 * |2W+ - tot/2|, integer arithmetic
    k = np.arange(tot + 1)
    p = float(dist[np.abs(2 * k - tot) >= dev].sum()) / float(2 ** n)
    return {"n_eff": n, "w_plus": w2 / 2.0, "p": min(1.0, p)}


def wilcoxon_legacy(x, y=None):
    """The earlier blocks' test: scipy.stats.wilcoxon(x, y).pvalue with the installed scipy defaults; None when all
    differences are ~0 or scipy raises (as in T7-T9 analysis scripts)."""
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    if len(d) == 0 or np.all(np.abs(d) < ZERO_TOL):
        return None
    try:
        return float(wilcoxon(d).pvalue)
    except Exception:  # noqa: BLE001
        return None


def holm(pvals) -> list:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(np.argsort(p, kind="stable")):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj.tolist()


def t_interval(d, level=0.95):
    from scipy.stats import t as tdist
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    half = float(tdist.ppf(0.5 + level / 2, n - 1) * d.std(ddof=1) / math.sqrt(n))
    return mean, mean - half, mean + half


def _arm_values(table, arm, ds, seeds, metric):
    return [(_num(table[(arm, ds, s)].get(metric)) if (arm, ds, s) in table and table[(arm, ds, s)].get(metric)
             not in (None, "") else None) for s in seeds]


def contrast(table, h, ds, seeds, metric):
    """Per-seed raw contrast (a - b, or the interaction), None where an arm has no row / no such column,
    NaN where a run diverged. Returns (list, n_missing)."""
    if h["kind"] == "pair":
        terms = [(1, h["a"]), (-1, h["b"])]
    else:
        terms = h["terms"]
    cols = [(c, _arm_values(table, arm, ds, seeds, metric)) for c, arm in terms]
    out = []
    for i in range(len(seeds)):
        vals = [(c, v[i]) for c, v in cols]
        if any(v is None for _, v in vals):
            out.append(None)
        else:
            out.append(sum(c * v for c, v in vals))
    return out


def family_for(h):
    arms = [h["a"], h["b"]] if h["kind"] == "pair" else [a for _, a in h["terms"]]
    return NON_MNIST if any(a.startswith("SC:L50") for a in arms) else CONFIGS


def run_tests(table, hyps, block, seeds, metric):
    """One row per (hypothesis, configuration). d is ORIENTED (positive = a better / contrast in the positive
    direction): d = raw for higher-is-better metrics, -raw for lower-is-better metrics."""
    sign = 1.0 if HIGHER_BETTER.get(metric, True) else -1.0
    rows = []
    for h in hyps:
        for ds in family_for(h):
            raw = contrast(table, h, ds, seeds, metric)
            present = [x for x in raw if x is not None]
            if not present:
                rows.append({"block": block, "hyp": h["id"], "metric": metric, "dataset": ds, "n": 0, "n_nan": 0,
                             "n_eff": 0, "p_exact": 1.0, "status": "no data"})
                continue
            n_nan = sum(1 for x in present if math.isnan(x))
            fin = [x for x in present if not math.isnan(x)]
            od = [sign * x for x in fin]
            ex = wilcoxon_exact(od)
            mean, lo, hi = t_interval(fin)
            # legacy p on the same finite pairs (T7-T9: scipy default)
            pl = wilcoxon_legacy(od)
            # sensitivity: a diverged run counts as a loss of the diverged side with the largest |d|
            nl = []
            if n_nan:
                sides = contrast_nan_sides(table, h, ds, seeds, metric)
                nl = od + [(-math.inf if side == "a" else math.inf) for side in sides]
            ex_nl = wilcoxon_exact(nl) if n_nan else ex
            omean = float(np.mean(od)) if od else float("nan")
            direction = "a" if omean > 0 else ("b" if omean < 0 else ("a" if ex["w_plus"] > ex["n_eff"] * (ex["n_eff"] + 1) / 4 else "0"))
            rows.append({"block": block, "hyp": h["id"], "metric": metric, "dataset": ds, "n": len(present),
                         "n_nan": n_nan, "n_eff": ex["n_eff"], "w_plus": ex["w_plus"], "p_exact": ex["p"],
                         "p_legacy": "" if pl is None else pl, "mean_diff_raw": mean, "ci95_lo": lo, "ci95_hi": hi,
                         "mean_diff_oriented": omean, "direction": direction,
                         "p_exact_nanloss": ex_nl["p"], "status": "ok" if ex["n_eff"] > 0 else "all zero"})
    # per-hypothesis Holm over its configuration family (§1)
    for h in hyps:
        sub = [r for r in rows if r["hyp"] == h["id"]]
        for key, col in (("p_exact", "p_holm"), ("p_exact_nanloss", "p_holm_nanloss")):
            adj = holm([_num(r.get(key, 1.0)) if r.get(key, "") != "" else 1.0 for r in sub])
            for r, a in zip(sub, adj):
                r[col] = a
    return rows


def contrast_nan_sides(table, h, ds, seeds, metric):
    """For pairs with a NaN: which side diverged ('a' or 'b'; interactions: the sign of the NaN terms' coefficients
    decides, ties -> dropped). Both sides NaN -> dropped."""
    terms = [(1, h["a"]), (-1, h["b"])] if h["kind"] == "pair" else h["terms"]
    cols = [(c, _arm_values(table, arm, ds, seeds, metric)) for c, arm in terms]
    sides = []
    for i in range(len(seeds)):
        vals = [(c, v[i]) for c, v in cols]
        if any(v is None for _, v in vals):
            continue
        nan_coefs = [c for c, v in vals if math.isnan(v)]
        if not nan_coefs:
            continue
        s = sum(nan_coefs)
        if s > 0:
            sides.append("a")
        elif s < 0:
            sides.append("b")
    return sides


def summarize_tests(rows, hyps, extra_cols=("p_holm_all",)):
    out = []
    for h in hyps:
        for metric in sorted({r["metric"] for r in rows if r["hyp"] == h["id"]}):
            for block in sorted({r["block"] for r in rows if r["hyp"] == h["id"] and r["metric"] == metric}):
                sub = [r for r in rows if r["hyp"] == h["id"] and r["metric"] == metric and r["block"] == block]
                tested = [r for r in sub if r.get("status") == "ok"]

                def cnt(col, side):
                    return sum(1 for r in tested if r.get(col, "") != "" and _num(r[col]) < ALPHA
                               and r["direction"] == side)
                s = {"block": block, "hyp": h["id"], "metric": metric, "family": len(sub), "n_tested": len(tested),
                     "n_with_nan": sum(1 for r in sub if r.get("n_nan", 0)), "expect": h.get("expect"),
                     "a": h.get("a", "interaction"), "b": h.get("b", "")}
                for col, name in (("p_exact", "unadj"), ("p_holm", "holm"), ("p_legacy", "legacy"),
                                  ("p_holm_nanloss", "holm_nanloss")) + tuple((c, c.replace("p_", "")) for c in extra_cols):
                    s[f"better_{name}"] = cnt(col, "a")
                    s[f"worse_{name}"] = cnt(col, "b")
                exp = h.get("expect")
                if exp in ("a", "b"):
                    st = s["better_holm"] if exp == "a" else s["worse_holm"]
                    op = s["worse_holm"] if exp == "a" else s["better_holm"]
                    s["replicates"] = int(st >= 1 and op <= st)
                elif exp == "none-better":
                    s["replicates"] = int(s["better_holm"] == 0)
                else:
                    s["replicates"] = ""
                out.append(s)
    return out


def decision_analysis(table, seeds):
    """§3 decision analysis on the given seeds (the held-out blocks 10-19 and 20-29). A (configuration, seed) enters
    only if every action has a row with that criterion. There, a DIVERGED run (NaN; or no stable calibrated setting)
    scores the WORST value any action obtained on that (configuration, seed) (counted as 'substituted'). State value =
    mean over seeds. Regret = best - value (ACC-type, absolute) or (value - best)/|best| (fcm_obj, upload numbers:
    their scale differs across configurations). Recommended = min mean regret / min max regret."""
    values, subs, per_state = {}, {}, []
    for crit in DECISION_CRITERIA:
        hb = HIGHER_BETTER[crit]
        for ds in CONFIGS:
            per_seed = {name: [] for name, _ in DECISION_ACTIONS}
            n_sub = {name: 0 for name, _ in DECISION_ACTIONS}
            ok_seeds = 0
            for s in seeds:
                vals = {}
                for name, arm in DECISION_ACTIONS:
                    r = table.get((arm, ds, s))
                    if r is None or r.get(crit) in (None, ""):
                        vals = None                     # a MISSING run/column (not a divergence): seed not usable
                        break
                    v = _num(r.get(crit))
                    if crit == "upload_numbers" and v < 0:
                        v = float("nan")                # no stable setting: no run, treated like a divergence
                    vals[name] = v
                if vals is None:
                    continue
                fin = [v for v in vals.values() if np.isfinite(v)]
                if not fin:
                    continue
                ok_seeds += 1
                worst = min(fin) if hb else max(fin)
                for name, v in vals.items():
                    if not np.isfinite(v):
                        v = worst
                        n_sub[name] += 1
                    per_seed[name].append(v)
            if not ok_seeds:
                continue
            for name, _ in DECISION_ACTIONS:
                values[(crit, ds, name)] = float(np.mean(per_seed[name]))
                subs[(crit, ds, name)] = n_sub[name]
    rows, table_rows = [], []
    for crit in DECISION_CRITERIA:
        hb = HIGHER_BETTER[crit]
        states = [ds for ds in CONFIGS if all((crit, ds, n) in values for n, _ in DECISION_ACTIONS)]
        if not states:
            rows.append({"criterion": crit, "action": "", "status": "no data"})
            continue
        reg = {n: [] for n, _ in DECISION_ACTIONS}
        for ds in states:
            vs = {n: values[(crit, ds, n)] for n, _ in DECISION_ACTIONS}
            best = max(vs.values()) if hb else min(vs.values())
            for n, v in vs.items():
                if crit in RELATIVE_REGRET:
                    rg = (v - best) / abs(best) if best != 0 else (0.0 if v == best else math.inf)
                else:
                    rg = best - v
                reg[n].append(rg)
                table_rows.append({"criterion": crit, "dataset": ds, "action": n, "value": v, "regret": rg,
                                   "n_substituted": subs[(crit, ds, n)], "best": int(v == best)})
        mean_r = {n: float(np.mean(r)) for n, r in reg.items()}
        max_r = {n: float(np.max(r)) for n, r in reg.items()}
        best_mean = min(mean_r.values())
        best_max = min(max_r.values())
        for n, _ in DECISION_ACTIONS:
            rows.append({"criterion": crit, "action": n, "n_states": len(states), "mean_regret": mean_r[n],
                         "max_regret": max_r[n], "argmax_state": states[int(np.argmax(reg[n]))],
                         "rec_mean_regret": int(mean_r[n] == best_mean), "rec_max_regret": int(max_r[n] == best_max),
                         "n_substituted": sum(subs[(crit, ds, n)] for ds in states), "status": "ok"})
    return rows, table_rows


def ec_analysis(t10_dir: Path):
    """E-C (exploratory): personalization on vs off at L=5 on the held-out 20% (and in-sample 80%), seeds 0-19."""
    path = t10_dir / "ec_split_L5_s00-19.csv"
    if not path.exists():
        return [], "E-C: ec_split_L5_s00-19.csv not found"
    table = {}
    small = {}
    for r in csv.DictReader(path.open()):
        table[(f"EC:{r['cell']}", r["dataset"], int(r["seed"]))] = r
        small[r["dataset"]] = min(small.get(r["dataset"], 10 ** 9), int(r.get("min_client_n_test", 10 ** 9)))
    seeds = sorted({k[2] for k in table})
    hyps = [dict(id="EC-GF", kind="pair", a="EC:post_footprint_on", b="EC:post_footprint_off", expect=None),
            dict(id="EC-PREMASS", kind="pair", a="EC:pre_mass_on", b="EC:pre_mass_off", expect=None)]
    rows = []
    for metric in ["test_fcm_obj", "test_gacc_trmap", "test_mean_acc_trmap", "test_worst_acc_trmap", "test_gacc",
                   "test_mean_acc_refit", "fcm_obj", "mean_acc", "gacc"]:
        for r in run_tests(table, hyps, f"s{seeds[0]:02d}-{seeds[-1]:02d}", seeds, metric):
            r["min_client_n_test"] = small.get(r["dataset"], "")
            r["test_acc_unreliable"] = int(metric.startswith("test_") and "acc" in metric
                                           and small.get(r["dataset"], 99) <= 5)
            rows.append(r)
    return rows, ""


def cmd_analyze(args):
    check_outdir(args)
    out = Path(args.outdir).resolve()
    paths = {k: out / f"{args.tag}{k}" for k in (".txt", "_tests.csv", "_summary.csv", "_count_changes.csv",
                                                  "_ea.csv", "_ec.csv", "_decision.csv", "_decision_values.csv",
                                                  ".meta.json")}
    refuse_existing(args, *paths.values())
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    for b in blocks:
        if b not in BLOCKS:
            raise SystemExit(f"unknown block {b}; known {sorted(BLOCKS)}")
    if "s20-29" in blocks and not args.allow_fresh:
        raise SystemExit("block s20-29 is the pre-registered fresh block; pass --allow-fresh (final analysis only)")
    t10_dir = Path(args.t10_dir).resolve()
    if args.source == "t10":
        table, missing = load_arms_t10(t10_dir)
    else:
        table, missing = load_arms_prior()
    lines = [f"# T10 analysis ({args.source} source), blocks {blocks}; written {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
             f"# exact two-sided Wilcoxon signed-rank (zeros |d|<={ZERO_TOL} dropped, mid-ranks), level {ALPHA}; "
             f"Holm per hypothesis over its configuration family; legacy = scipy.stats.wilcoxon default "
             f"(the earlier blocks' test)"]
    if missing:
        lines.append(f"# missing T10 CSVs (arms absent): {missing}")
    tests = []
    for b in blocks:
        seeds = BLOCKS[b]
        for metric in ("mean_acc", "fcm_obj", "worst_acc", "gacc"):
            tests += run_tests(table, H_PRIMARY + H_SECONDARY, b, seeds, metric)
    # §3 sensitivity: Holm over ALL primary tests H1-H8 x configurations of a block (mean_acc)
    for b in blocks:
        sub = [r for r in tests if r["block"] == b and r["metric"] == "mean_acc" and r["hyp"] in
               {f"H{i}" for i in range(1, 9)}]
        adj = holm([_num(r["p_exact"]) for r in sub])
        for r, a in zip(sub, adj):
            r["p_holm_all"] = a
    summ = summarize_tests(tests, H_PRIMARY + H_SECONDARY)
    # counts that change between the legacy (earlier-blocks) test and the exact test, unadjusted
    changes = [{"block": s["block"], "hyp": s["hyp"], "metric": s["metric"], "count": side,
                "legacy": s[f"{side}_legacy"], "exact": s[f"{side}_unadj"]}
               for s in summ for side in ("better", "worse") if s[f"{side}_legacy"] != s[f"{side}_unadj"]]
    # E-A: substitution interactions at L=5 for every gate G
    ea_h = [dict(id=f"EA-{g}", kind="inter", terms=[(1, "L5:pre_none_off"), (-1, "L5:post_none_off"),
                                                    (-1, f"L5:pre_{g}_off"), (1, f"L5:post_{g}_off")], expect=None)
            for g in EA_GATES]
    ea = []
    for b in blocks:
        for metric in ("mean_acc", "fcm_obj"):
            ea += run_tests(table, ea_h, b, BLOCKS[b], metric)
    ea_sum = summarize_tests(ea, ea_h, extra_cols=())
    ec, ec_note = ec_analysis(t10_dir) if args.source == "t10" else ([], "E-C: not available from the prior CSVs")
    ec_sum = summarize_tests(ec, [dict(id="EC-GF"), dict(id="EC-PREMASS")], extra_cols=()) if ec else []
    dec_seeds = [s for b in blocks if b in ("s10-19", "s20-29") for s in BLOCKS[b]]
    dec, dec_vals = decision_analysis(table, dec_seeds) if dec_seeds else ([], [])
    # ---- write
    write_csv(paths["_tests.csv"], tests, ["block", "hyp", "metric", "dataset", "n", "n_nan", "n_eff", "w_plus",
                                           "p_exact", "p_holm", "p_holm_all", "p_legacy", "mean_diff_raw", "ci95_lo",
                                           "ci95_hi", "direction"])
    write_csv(paths["_summary.csv"], summ + ea_sum + ec_sum)
    write_csv(paths["_count_changes.csv"], changes, ["block", "hyp", "metric", "count", "legacy", "exact"])
    write_csv(paths["_ea.csv"], ea, None)
    write_csv(paths["_ec.csv"], ec, None)
    write_csv(paths["_decision.csv"], dec, None)
    write_csv(paths["_decision_values.csv"], dec_vals, None)
    # ---- human-readable report
    lines.append("\n## §1 hypotheses: better/worse counts (a vs b; 'better' = a better). unadj | Holm | legacy"
                 " | Holm over all H1-H8 (sensitivity) | Holm with diverged runs as losses")
    for s in summ:
        if s["metric"] not in ("mean_acc", "fcm_obj"):
            continue
        rep = {1: "REPLICATES", 0: "does not replicate", "": ""}.get(s["replicates"], "")
        lines.append(f"{s['block']} {s['hyp']:<10} {s['metric']:<8} fam {s['family']:>2} tested {s['n_tested']:>2}: "
                     f"{s['better_unadj']}/{s['worse_unadj']} | {s['better_holm']}/{s['worse_holm']} | "
                     f"{s['better_legacy']}/{s['worse_legacy']} | {s.get('better_holm_all', '')}/"
                     f"{s.get('worse_holm_all', '')} | {s['better_holm_nanloss']}/{s['worse_holm_nanloss']}"
                     f"  expect {s['expect']}  {rep if s['block'] == 's20-29' and s['metric'] == 'mean_acc' else ''}")
    k_all = 8 * len(CONFIGS)
    lines.append(f"# NOTE (design property, known before the fresh run): with n' = 10 the smallest attainable exact "
                 f"two-sided p is 1/512 = {1 / 512:.5f}; the sensitivity Holm over all H1-H8 x configurations "
                 f"(K = {k_all}) needs p <= {ALPHA / k_all:.5f} for its first rejection, so it cannot reject any test "
                 f"with n' <= 10 (its counts are 0/0 by construction). Per-hypothesis Holm (K = 14) needs "
                 f"p <= {ALPHA / 14:.5f}: attainable.")
    lines.append("\n## counts that change between the legacy test and the exact test (unadjusted)")
    lines += [f"{c['block']} {c['hyp']} {c['metric']} {c['count']}: legacy {c['legacy']} -> exact {c['exact']}"
              for c in changes] or ["(none)"]
    lines.append("\n## E-A substitution interactions at L=5 (positive = gate substitutes for PRE): unadj | Holm")
    for s in ea_sum:
        lines.append(f"{s['block']} {s['hyp']:<12} {s['metric']:<8} tested {s['n_tested']:>2}: positive "
                     f"{s['better_unadj']}/negative {s['worse_unadj']} | {s['better_holm']}/{s['worse_holm']}")
    lines.append("\n## E-C (exploratory, L=5, seeds 0-19): personalization on vs off; unadj | Holm "
                 "(test ACC unreliable where a client has <= 5 test points)")
    if ec_note:
        lines.append(ec_note)
    for s in ec_sum:
        unrel = sorted({r["dataset"] for r in ec if r["hyp"] == s["hyp"] and r["metric"] == s["metric"]
                        and r.get("test_acc_unreliable")})
        lines.append(f"{s['hyp']:<11} {s['metric']:<22} on better {s['better_unadj']}/worse {s['worse_unadj']} | "
                     f"{s['better_holm']}/{s['worse_holm']}" + (f"  unreliable: {unrel}" if unrel else ""))
    lines.append(f"\n## decision analysis on seeds {dec_seeds[:1]}..{dec_seeds[-1:]} ({len(dec_seeds)} seeds; "
                 "mean regret | max regret; * = recommended)")
    for r in dec:
        if r.get("status") != "ok":
            lines.append(f"{r['criterion']}: no data")
            continue
        lines.append(f"{r['criterion']:<15} {r['action']:<28} {r['mean_regret']:.4g}{'*' if r['rec_mean_regret'] else ' '}"
                     f" | {r['max_regret']:.4g}{'*' if r['rec_max_regret'] else ' '}  (worst state {r['argmax_state']};"
                     f" substituted {r['n_substituted']})")
    paths[".txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_meta(paths[".meta.json"], args, {"blocks": blocks, "missing": missing, "scipy": __import__("scipy").__version__})
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, seeds=True, workers=True):
        p.add_argument("--tag", required=True, help="output name: results/t10/<tag>.csv")
        p.add_argument("--outdir", default=str(OUT_DIR))
        p.add_argument("--force", action="store_true", help="overwrite an existing output")
        p.add_argument("--datasets", default=None, help=f"comma list of labels or aliases {sorted(DATASET_ALIASES)}")
        if workers:
            p.add_argument("--workers", type=int, default=12)
        if seeds:
            p.add_argument("--seed-start", type=int, default=0)
            p.add_argument("--seeds", type=int, default=10)
            p.add_argument("--allow-fresh", action="store_true", help="permit seeds 20-29 (fresh block)")

    p = sub.add_parser("cells"); common(p)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--cells", default=None, help="comma list; groups all12/all16/masks; default all16")
    p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--split", type=float, default=0.0)
    p.add_argument("--no-earlystop", action="store_true",
                   help="FFCM_NO_EARLYSTOP=1 (provenance checks of prior files only; not part of the protocol)")
    p.set_defaults(func=cmd_cells)

    p = sub.add_parser("scffcm"); common(p)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--steps-csv", default=None)
    p.add_argument("--eta-l", type=float, default=None)
    p.add_argument("--eta-g", type=float, default=None)
    p.add_argument("--cfraction", type=float, default=1.0)
    p.set_defaults(func=cmd_scffcm)

    for name, fn, gdef in (("scffcm-calibrate", cmd_scffcm_calibrate, "t9b"),
                           ("pedrycz-calibrate", cmd_pedrycz_calibrate, "ee")):
        p = sub.add_parser(name); common(p, seeds=False)
        p.add_argument("--L", type=int, required=True)
        p.add_argument("--rounds", type=int, default=50)
        p.add_argument("--grid", default=gdef, help="named grid or explicit 'a1,a2;b1,b2'")
        p.add_argument("--cal-seed", type=int, default=CAL_SEED)
        p.add_argument("--allow-fresh", action="store_true", help="permit a --cal-seed in 20-29 (never needed)")
        p.set_defaults(func=fn)

    p = sub.add_parser("pedrycz"); common(p)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--alpha-csv", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.set_defaults(func=cmd_pedrycz)

    p = sub.add_parser("cfcm"); common(p)
    p.add_argument("--max-iter", type=int, default=150, help="T6_baselines: fcm() default max_iter = 150")
    p.set_defaults(func=cmd_cfcm)

    p = sub.add_parser("audit"); common(p)
    p.set_defaults(seeds=5)
    p.add_argument("--ms", default=",".join(str(m) for m in M_GRID))
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--restarts", type=int, default=10)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("compare")
    p.add_argument("--kind", choices=["cells", "scffcm", "cfcm", "audit"], required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-runs", type=int, default=None,
                   help="fail if fewer strict overlapping runs than this (expected coverage of the prior files)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("argcheck", help="does <meta> record the same command? (exit 0 = same)")
    p.add_argument("--meta", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_argcheck)

    p = sub.add_parser("analyze")
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--force", action="store_true")
    p.add_argument("--source", choices=["t10", "prior"], default="t10",
                   help="t10: the plan's CSVs in --t10-dir; prior: the earlier blocks' CSVs (test of this code)")
    p.add_argument("--t10-dir", default=str(OUT_DIR))
    p.add_argument("--blocks", default="s00-09,s10-19", help="comma list of s00-09, s10-19, s20-29")
    p.add_argument("--allow-fresh", action="store_true", help="include block s20-29 (final analysis only)")
    p.set_defaults(func=cmd_analyze)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
