"""R6b: WHY do footprint and mass gates give similar aggregates? By Thm tilt, the one-round difference of the server
targets is alpha*Cov_pi(g, c)/E[g] with tilt g = r/q and base the mass-gated weights. It is small either because g is
(nearly) constant across clients (server-blind hypothesis; measured share was low) or because the local centres c_pj of
the clients that carry weight for prototype j nearly coincide. Along the post_footprint_off trajectory we log, per
round and prototype: disp = ||target(footprint) - target(mass)||, spread = weighted std of the c_pj (mass-gated
weights), rel = disp / spread, and the coefficient of variation of g over the weight. m*, seeds 0-4, 50 rounds, L=5."""
import os, sys, csv
os.environ["FFCM_MSTAR"] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
CONFIGS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
           "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
def job(arg):
    label, seed = arg
    os.environ["FFCM_MSTAR"] = "1"; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data, predict_membership, run_local_fcm_steps
    T.M = m = T.fuzzifier_for(label)
    spec = next(s for s in ([("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] + [("real",) + tuple(d) for d in T.REAL_DATASETS]) if s[1] == label)
    clients, k = T.build_clients(spec, seed)
    X = np.vstack([c.x for c in clients]); V = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    N = np.array([len(c.x) for c in clients], float); beta = N / N.sum(); P = len(clients)
    rels, cvs, disps, spreads = [], [], [], []
    for t in range(T.ROUNDS):
        ML = np.zeros((P, k)); C = np.zeros((P, k, V.shape[1])); R = np.zeros((P, k)); Q = np.zeros((P, k))
        for p, c in enumerate(clients):
            ub = predict_membership(c.x, V, m=m); loc = run_local_fcm_steps(c.x, V, steps=T.LOCAL_STEPS, m=m)
            ML[p] = (loc.membership ** m).sum(0); C[p] = loc.centers
            R[p] = T.gate_footprint(c.x, V, ub); Q[p] = T.gate_mass(c.x, V, ub)
        for j in range(k):
            wq = beta * ML[:, j] * Q[:, j]; wr = beta * ML[:, j] * R[:, j]
            if wq.sum() <= 0 or wr.sum() <= 0: continue
            pq = wq / wq.sum(); pr = wr / wr.sum()
            tq = (pq[:, None] * C[:, j]).sum(0); tr = (pr[:, None] * C[:, j]).sum(0)
            disp = float(np.linalg.norm(tr - tq)); spread = float(np.sqrt((pq * ((C[:, j] - tq) ** 2).sum(1)).sum()))
            g = R[:, j] / Q[:, j]; gm = (pq * g).sum(); cv = float(np.sqrt((pq * (g - gm) ** 2).sum()) / gm) if gm > 0 else np.nan
            disps.append(disp); spreads.append(spread); cvs.append(cv)
            if spread > 1e-12: rels.append(disp / spread)
        w = beta[:, None] * ML * R; num = (w[:, :, None] * C).sum(0); den = w.sum(0)
        Tt = V.copy(); ok = den > T.EPS; Tt[ok] = num[ok] / den[ok, None]; V = (1 - T.SERVER_LR) * V + T.SERVER_LR * Tt
    return dict(dataset=label, seed=seed, m=m, disp_median=float(np.median(disps)), spread_median=float(np.median(spreads)),
                rel_median=float(np.median(rels)), tilt_cv_median=float(np.nanmedian(cvs)))
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex: rows = list(ex.map(job, [(l, s) for l in CONFIGS for s in range(5)]))
    with open(HERE.parent / "results" / "t9_theory_link_b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import statistics as st
    print(f"{'dataset':<23}{'m*':>5}{'tilt CV (r/q over clients)':>28}{'target displacement':>21}{'centre spread':>15}{'disp/spread':>13}")
    for l in CONFIGS:
        rr = [r for r in rows if r["dataset"] == l]; f = lambda key: st.mean(r[key] for r in rr)
        print(f"{l:<23}{rr[0]['m']:>5}{f('tilt_cv_median'):>28.3f}{f('disp_median'):>21.4f}{f('spread_median'):>15.3f}{f('rel_median'):>13.3f}")
