"""R2b: a fair SC-FFCM on an EXTENDED step grid (the R2 grid was truncated: its corner was chosen on 14/14 configs at L=1). Step sizes are re-calibrated for EVERY (configuration, L) at that configuration's m*, on a
NON-evaluation seed (seed index 100) with a LABEL-FREE criterion: among non-diverging settings (finite prototypes,
final pooled FCM objective <= 5x centralized), pick the one whose final pooled FCM objective is closest to that of
centralized FCM from the same init. Grid eta_l in {0.05,0.1,0.2,0.5} x eta_g in {0.5,1,2}. Then evaluate on seeds 0-19
at L in {1,5} (all 14 configs) and at L=50, the source code's default, on seeds 0-9 for the 12 non-MNIST configs."""
import os, sys, csv, math
os.environ["FFCM_MSTAR"] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
CONFIGS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
           "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
GRID = [(a, b) for a in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0) for b in (0.5, 1.0, 2.0, 4.0, 8.0)]
def setup(label, seed):
    os.environ["FFCM_MSTAR"] = "1"; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data
    T.M = T.fuzzifier_for(label)
    spec = next(s for s in ([("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] + [("real",) + tuple(d) for d in T.REAL_DATASETS]) if s[1] == label)
    clients, k = T.build_clients(spec, seed)
    X = np.vstack([c.x for c in clients]); init = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    return T, clients, k, X, init
def pooled_obj(T, X, V):
    from fedfcmsim.fcm import predict_membership
    u = predict_membership(X, V, m=T.M); d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1); return float(((u ** T.M) * d2).sum())
def calibrate(arg):
    label, L = arg
    T, clients, k, X, init = setup(label, 100)
    from fedfcmsim.fcm import fcm
    from fedfcmsim.federated import scffcm
    Jc = pooled_obj(T, X, fcm(X, k, m=T.M, init_centers=init, max_iter=300).centers)
    best = None; tried = []
    for a, b in GRID:
        try: V = scffcm(clients, init, rounds=T.ROUNDS, local_steps=L, m=T.M, eta_l=a, eta_g=b, track_objective=False).centers
        except Exception: V = None
        ok = V is not None and np.all(np.isfinite(V))
        J = pooled_obj(T, X, V) if ok else math.inf
        ok = ok and J <= 5 * Jc; gap = abs(J - Jc) / Jc if ok else math.inf
        tried.append((a, b, gap))
        if ok and (best is None or gap < best[2]): best = (a, b, gap)
    return dict(dataset=label, L=L, m=T.M, eta_l=best[0] if best else "", eta_g=best[1] if best else "",
                rel_obj_gap=best[2] if best else "", n_stable=sum(1 for t in tried if t[2] < math.inf))
def evaluate(arg):
    label, L, seed, a, b = arg
    T, clients, k, X, init = setup(label, seed)
    from fedfcmsim.federated import scffcm
    V = scffcm(clients, init, rounds=T.ROUNDS, local_steps=L, m=T.M, eta_l=a, eta_g=b, track_objective=False).centers
    if not np.all(np.isfinite(V)): return dict(dataset=label, L=L, seed=seed, eta_l=a, eta_g=b, min_dist="nan", mean_acc="nan", worst_acc="nan", fcm_obj="nan")
    ma, wa = T.client_accs(clients, [V] * len(clients))
    return dict(dataset=label, L=L, seed=seed, eta_l=a, eta_g=b, min_dist=T.min_dist(V), mean_acc=ma, worst_acc=wa,
                fcm_obj=T.fcm_objective_mean(clients, [V] * len(clients)))
if __name__ == "__main__":
    R = HERE.parent / "results"
    cal_jobs = [(l, L) for l in CONFIGS for L in (1, 5)]
    with ProcessPoolExecutor(max_workers=5) as ex: cal = list(ex.map(calibrate, cal_jobs))
    with open(R / "t9b_scffcm_calibration.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal[0])); w.writeheader(); w.writerows(cal)
    print("calibration done:", sum(1 for c in cal if c["eta_l"] != ""), "/", len(cal), "found a stable setting", flush=True)
    steps = {(c["dataset"], c["L"]): (c["eta_l"], c["eta_g"]) for c in cal if c["eta_l"] != ""}
    ev_jobs = [(l, L, s, *steps[(l, L)]) for (l, L) in steps for s in (range(20) if L in (1, 5) else range(10))]
    with ProcessPoolExecutor(max_workers=5) as ex: ev = list(ex.map(evaluate, ev_jobs))
    with open(R / "t9b_scffcm_fair.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev[0])); w.writeheader(); w.writerows(ev)
    print("evaluation done:", len(ev), "runs")
