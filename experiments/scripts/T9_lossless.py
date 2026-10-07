"""R4: the exact lossless check (Corcuera Barcena et al.). One local step with the mass measured at the broadcast
prototypes, three server variants, versus centralized FCM iterated from the same init:
  (a) unweighted sums, no damping      V <- sum_p S_pj / sum_p Mbar_pj           (should equal centralized FCM exactly)
  (b) unweighted sums, damping 0.75    same fixed point as (a), slower path
  (c) size-weighted beta_p, damping 0.75  = the design-space cell pre_none_off at L=1 used in the paper
S_pj = sum_k ubar^m x, Mbar_pj = sum_k ubar^m, ubar at the broadcast V. 50 rounds, m*, seeds 0-9."""
import os, sys, csv
os.environ["FFCM_MSTAR"] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
CONFIGS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
           "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
def job(a):
    label, seed = a
    os.environ["FFCM_MSTAR"] = "1"; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data, predict_membership
    T.M = m = T.fuzzifier_for(label)
    spec = next(s for s in ([("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] + [("real",) + tuple(d) for d in T.REAL_DATASETS]) if s[1] == label)
    clients, k = T.build_clients(spec, seed)
    X = np.vstack([c.x for c in clients]); N = np.array([len(c.x) for c in clients], float); beta = N / N.sum()
    V0 = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    def cen_step(V):
        u = predict_membership(X, V, m=m) ** m; return (u.T @ X) / np.maximum(u.sum(0), 1e-300)[:, None]
    def fed_step(V, weighted):
        num = np.zeros_like(V); den = np.zeros(k)
        for p, c in enumerate(clients):
            u = predict_membership(c.x, V, m=m) ** m
            s = beta[p] if weighted else 1.0
            num += s * (u.T @ c.x); den += s * u.sum(0)
        T_ = V.copy(); ok = den > 1e-300; T_[ok] = num[ok] / den[ok, None]; return T_
    Vc = V0.copy(); Va = V0.copy(); Vb = V0.copy(); Vcw = V0.copy(); max_gap_a = 0.0
    for _ in range(50):
        Vc = cen_step(Vc); Va = fed_step(Va, False)
        max_gap_a = max(max_gap_a, float(np.abs(Va - Vc).max()))
        Vb = 0.25 * Vb + 0.75 * fed_step(Vb, False)
        Vcw = 0.25 * Vcw + 0.75 * fed_step(Vcw, True)
    acc = lambda V: T.client_accs(clients, [V] * len(clients))
    (ac, wc), (aa, wa), (ab, wb), (aw, ww) = acc(Vc), acc(Va), acc(Vb), acc(Vcw)
    return dict(dataset=label, seed=seed, m=m, unequal_sizes=int(N.std() > 0),
                maxgap_a_vs_cen=max_gap_a, gap_b_final=float(np.abs(Vb - Vc).max()), gap_c_final=float(np.abs(Vcw - Vc).max()),
                acc_cen=ac, acc_a=aa, acc_b=ab, acc_c=aw, worst_cen=wc, worst_c=ww)
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex: rows = list(ex.map(job, [(l, s) for l in CONFIGS for s in range(10)]))
    out = HERE.parent / "results" / "t9_lossless.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import statistics as st
    print(f"{'dataset':<23}{'m*':>5}{'unequal N':>10}{'max|a-cen| (all iters)':>24}{'|b-cen| end':>13}{'|c-cen| end':>13}{'ACC cen':>9}{'ACC a':>8}{'ACC c':>8}")
    for l in CONFIGS:
        rr = [r for r in rows if r["dataset"] == l]; f = lambda k: st.mean(r[k] for r in rr)
        print(f"{l:<23}{rr[0]['m']:>5}{rr[0]['unequal_sizes']:>10}{max(r['maxgap_a_vs_cen'] for r in rr):>24.2e}{f('gap_b_final'):>13.2e}{f('gap_c_final'):>13.2e}{f('acc_cen'):>9.3f}{f('acc_a'):>8.3f}{f('acc_c'):>8.3f}")
