"""E6: at each config's m*, how often does the footprint gate hit its floor r_min (clipping), and how much does
Phi = T*N/Mbar vary across prototypes within a client (if Phi were constant in j the footprint gate would equal
the mass gate exactly -- Prop. r=q iff Phi constant in j). Cell post_footprint_off, 5 seeds, T3 protocol."""
import os, sys, csv
os.environ["FFCM_MSTAR"] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
CONFIGS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise",
           "quantity_skew_extreme","wine","satimage","pendigits","digits_pca16","digits_pca32","letter",
           "mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
def job(a):
    label, seed = a
    os.environ["FFCM_MSTAR"] = "1"
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data, predict_membership
    from fedfcmsim.federated import granular_footprint
    T.M = T.fuzzifier_for(label)
    spec = next(s for s in ([("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] +
                            [("real",) + tuple(d) for d in T.REAL_DATASETS]) if s[1] == label)
    clients, k = T.build_clients(spec, seed)
    pooled = np.vstack([c.x for c in clients])
    init = initialize_centers_from_data(pooled, k, random_state=T.SEED_OFFSET + seed + 77)
    V, _, ex = T.run_cell(clients, init, "post", "footprint", "off", record_gates=True)
    fp = np.array(ex["fp_gates"]) if isinstance(ex.get("fp_gates"), list) else None
    # fp_hist: list over rounds of per-client gate arrays
    hist = ex.get("fp_gates") or []
    allg = np.concatenate([np.concatenate([np.ravel(g) for g in rnd]) for rnd in hist]) if hist else np.array([])
    last = np.concatenate([np.ravel(g) for g in hist[-1]]) if hist else np.array([])
    clip_all = float(np.mean(allg <= T.MIN_RELEVANCE + 1e-12)) if allg.size else float("nan")
    clip_last = float(np.mean(last <= T.MIN_RELEVANCE + 1e-12)) if last.size else float("nan")
    cvs, ratios = [], []
    for c in clients:
        ub = predict_membership(c.x, V, m=T.M); Mb = (ub ** T.M).sum(0)
        Tfp, _ = granular_footprint(c.x, V, ub, m=T.M, grid_size=T.FOOTPRINT_GRID)
        keep = Mb > 1e-3 * len(c.x)
        if keep.sum() >= 2:
            phi = Tfp[keep] * len(c.x) / Mb[keep]
            if phi.min() > 0:
                cvs.append(float(phi.std() / phi.mean())); ratios.append(float(phi.max() / phi.min()))
    return dict(dataset=label, seed=seed, m=T.M, clip_rate_all_rounds=round(clip_all,4),
                clip_rate_final=round(clip_last,4), phi_cv_median=round(float(np.median(cvs)),4) if cvs else "",
                phi_maxmin_median=round(float(np.median(ratios)),4) if ratios else "")
if __name__ == "__main__":
    jobs = [(l, s) for l in CONFIGS for s in range(5)]
    with ProcessPoolExecutor(max_workers=3) as ex: rows = list(ex.map(job, jobs))
    out = HERE.parent / "results" / "t8_e6_clipping_phi.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import statistics as st
    print(f"{'dataset':<23}{'m*':>5}{'clip% (all rounds)':>20}{'clip% (final)':>15}{'Phi CV (median)':>17}{'Phi max/min':>13}")
    for l in CONFIGS:
        rr = [r for r in rows if r["dataset"] == l]
        f = lambda key: st.mean(float(r[key]) for r in rr if r[key] != "")
        print(f"{l:<23}{rr[0]['m']:>5}{100*f('clip_rate_all_rounds'):>19.1f}%{100*f('clip_rate_final'):>14.1f}%{f('phi_cv_median'):>17.3f}{f('phi_maxmin_median'):>13.2f}")
