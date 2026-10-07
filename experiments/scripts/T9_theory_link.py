"""R6: connect the theory to the runs. Along the trajectory of the gated POST cell (post_footprint_off, exactly the
T3 run_cell loop), at every round and prototype we log:
 (1) both sides of the substitution inequality eq:th-empirical-ineq:
       LHS_j = ||Cov_{pi^GATE}(omega, c)|| / E_{pi^GATE}[omega],  RHS_j = ||Cov_{pi^POST}(omega, c)|| / E_{pi^POST}[omega]
     with omega = Mbar/M^L, pi^POST = beta M^L / sum, pi^GATE = pi^POST r / sum, c = the local centres. Substitution in
     that round <=> LHS < RHS. We also verify Cor. same-tilt numerically (E_off from the formula equals the direct
     difference of the PRE and POST server targets).
 (2) the server-blind statistic of Thm server-blind: D_pj = log r_pj - log q_pj on unclipped pairs; the share of its
     variance explained by a prototype-only effect. Share = 1 <=> footprint and mass gates give identical server updates.
m*, seeds 0-4, 50 rounds, L=5."""
import os, sys, csv
os.environ["FFCM_MSTAR"] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
CONFIGS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
           "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
def wcov(pi, a, C):
    abar = (pi * a).sum(); cbar = (pi[:, None] * C).sum(0)
    return ((pi * (a - abar))[:, None] * (C - cbar)).sum(0)
def job(arg):
    label, seed = arg
    os.environ["FFCM_MSTAR"] = "1"; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data, predict_membership, run_local_fcm_steps
    T.M = m = T.fuzzifier_for(label)
    spec = next(s for s in ([("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] + [("real",) + tuple(d) for d in T.REAL_DATASETS]) if s[1] == label)
    clients, k = T.build_clients(spec, seed)
    X = np.vstack([c.x for c in clients]); V = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    N = np.array([len(c.x) for c in clients], float); beta = N / N.sum(); P = len(clients); rmin = T.MIN_RELEVANCE
    ineq_hold = ineq_tot = 0; ratios = []; shares = []; corgap = 0.0
    for t in range(T.ROUNDS):
        Mbar = np.zeros((P, k)); ML = np.zeros((P, k)); C = np.zeros((P, k, V.shape[1])); R = np.zeros((P, k)); Q = np.zeros((P, k))
        for p, c in enumerate(clients):
            ub = predict_membership(c.x, V, m=m); loc = run_local_fcm_steps(c.x, V, steps=T.LOCAL_STEPS, m=m)
            Mbar[p] = (ub ** m).sum(0); ML[p] = (loc.membership ** m).sum(0); C[p] = loc.centers
            R[p] = T.gate_footprint(c.x, V, ub); Q[p] = T.gate_mass(c.x, V, ub)
        for j in range(k):
            wpost = beta * ML[:, j]
            if wpost.sum() <= 0 or np.any(ML[:, j] <= 0): continue
            pip = wpost / wpost.sum(); om = Mbar[:, j] / ML[:, j]
            pig = pip * R[:, j]; pig = pig / pig.sum()
            lhs = np.linalg.norm(wcov(pig, om, C[:, j])) / (pig * om).sum()
            rhs = np.linalg.norm(wcov(pip, om, C[:, j])) / (pip * om).sum()
            ineq_tot += 1; ineq_hold += lhs < rhs
            if rhs > 1e-12: ratios.append(lhs / rhs)
            # Corollary check: target(PRE) - target(POST) == Cov_pi^POST(omega,c)/E[omega]
            wpre = beta * Mbar[:, j]; tpre = (wpre[:, None] * C[:, j]).sum(0) / wpre.sum(); tpost = (pip[:, None] * C[:, j]).sum(0)
            corgap = max(corgap, float(np.abs((tpre - tpost) - wcov(pip, om, C[:, j]) / (pip * om).sum()).max()))
        mask = (R > rmin + 1e-12) & (Q > rmin + 1e-12)
        if mask.sum() >= 4:
            D = np.where(mask, np.log(np.maximum(R, 1e-300)) - np.log(np.maximum(Q, 1e-300)), np.nan)
            allv = D[mask]; tot = ((allv - allv.mean()) ** 2).sum()
            within = sum(((D[:, j][mask[:, j]] - D[:, j][mask[:, j]].mean()) ** 2).sum() for j in range(k) if mask[:, j].sum() > 0)
            if tot > 1e-15: shares.append(1.0 - within / tot)
        # the gated POST server update (identical to run_cell for post_footprint_off)
        w = beta[:, None] * ML * R; num = (w[:, :, None] * C).sum(0); den = w.sum(0)
        Tt = V.copy(); ok = den > T.EPS; Tt[ok] = num[ok] / den[ok, None]; V = (1 - T.SERVER_LR) * V + T.SERVER_LR * Tt
    return dict(dataset=label, seed=seed, m=m, ineq_frac=ineq_hold / max(ineq_tot, 1),
                ratio_median=float(np.median(ratios)) if ratios else float("nan"),
                proto_share_median=float(np.median(shares)) if shares else float("nan"), corollary_maxgap=corgap)
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex: rows = list(ex.map(job, [(l, s) for l in CONFIGS for s in range(5)]))
    out = HERE.parent / "results" / "t9_theory_link.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import statistics as st
    print(f"{'dataset':<23}{'m*':>5}{'ineq holds (share of round x proto)':>37}{'median LHS/RHS':>16}{'proto-effect share':>20}{'Cor. check':>12}")
    for l in CONFIGS:
        rr = [r for r in rows if r["dataset"] == l]; f = lambda key: st.mean(r[key] for r in rr)
        print(f"{l:<23}{rr[0]['m']:>5}{f('ineq_frac'):>37.3f}{f('ratio_median'):>16.3f}{f('proto_share_median'):>20.3f}{max(r['corollary_maxgap'] for r in rr):>12.1e}")
