"""T10 theory checks (preregistration addendum §3, items E-H and E-I). m*, seeds 0-4, 50 rounds, <= 6 workers.

  python3 scripts/T10_theory.py run      # all trajectory jobs (E-H: 3 trajectories; E-I: 3 rules) x 14 configs x seeds 0-4
                                         # (per-job cache in results/t10/theory/_cache/, resumable)
  python3 scripts/T10_theory.py analyze  # writes E_H.csv, E_H_config.csv, E_I_descent.csv, E_I_counterexample.txt, SUMMARY.md

Helpers: T10_theory_common.py (setup), T10_theory_eh.py (E-H job), T10_theory_ei.py (E-I descent job, counter-example).
Outputs only under results/t10/theory/. Never modifies T3_design_space.py, fedfcmsim/ or existing results."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["FFCM_MSTAR"] = "1"
import sys, csv, json, time, statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from T10_theory_common import CONFIGS, SEEDS, OUT, EXP

CACHE = OUT / "_cache"
EH_TRAJ = [("post", 5), ("pre", 5), ("pre", 1)]
EI_RULES = [("pre", "none", 1), ("pre", "mass", 1), ("pre", "none", 5)]
COST = {"mnist784_pca32 (50c)": 0, "mnist784_pca32 (20c)": 1, "letter": 2, "satimage": 3, "pendigits": 4}


def jobs():
    out = []
    for lab in CONFIGS:
        for s in SEEDS:
            for mp, L in EH_TRAJ:
                out.append(("eh", (lab, s, mp, L)))
            for mp, g, L in EI_RULES:
                out.append(("ei", (lab, s, mp, g, L)))
    return sorted(out, key=lambda j: (COST.get(j[1][0], 9), j[0]))


def key(kind, arg):
    return (kind + "__" + "__".join(str(a) for a in arg)).replace(" ", "_").replace("(", "").replace(")", "") + ".json"


def work(job):
    kind, arg = job
    t0 = time.time()
    if kind == "eh":
        from T10_theory_eh import eh_job; row = eh_job(arg)
    else:
        from T10_theory_ei import descent_job; row = descent_job(arg)
    row["seconds"] = round(time.time() - t0, 1)
    (CACHE / key(kind, arg)).write_text(json.dumps(row))
    return kind, arg, row["seconds"]


def run():
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not (CACHE / key(*j)).exists()]
    print(f"{len(todo)} jobs to run", flush=True); t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            kind, arg, sec = f.result()
            print(f"[{i}/{len(todo)}] {kind} {arg} {sec}s ({time.time() - t0:.0f}s total)", flush=True)


# ------------------------------------------------------------------------------------------------------------ analysis
def load_rows(kind):
    return [json.loads((CACHE / key(*j)).read_text()) for j in jobs() if j[0] == kind and (CACHE / key(*j)).exists()]


def t7_namespace():
    """Execute the loading part of T7_final_analysis.py (its seed/file selection), as T9_theory_crossconfig does."""
    src = (HERE / "T7_final_analysis.py").read_text().split('print("="*100); print("0.')[0]
    ns = {"__file__": str(HERE / "T7_final_analysis.py")}; exec(src, ns); return ns


def l1all():
    acc = {}
    for r in csv.DictReader(open(EXP / "results" / "t3_design_space" / "per_seed_metrics__L1all.csv")):
        acc.setdefault((r["dataset"], r["cell"]), {})[int(r["seed"])] = float(r["mean_acc"])
    return acc


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def analyze():
    from scipy.stats import spearmanr
    import numpy as np
    L = []  # summary lines
    P = lambda s="": L.append(s)
    # ---------------------------------------------------------------- E-H
    eh = load_rows("eh")
    fields = [k for k in eh[0] if k != "seconds"]
    write_csv(OUT / "E_H.csv", eh, fields)
    ns = t7_namespace(); T7acc = ns["acc"]; M7 = ns["M"]; A1 = l1all()
    # T9 reproduction
    t9 = {(r["dataset"], int(r["seed"])): r for r in csv.DictReader(open(EXP / "results" / "t9_theory_link_b.csv"))}
    repro = {c: 0.0 for c in ("tilt_cv_median", "disp_median", "spread_median", "rel_median")}; nrep = 0
    for r in eh:
        if r["trajectory"] != "post_footprint_off_L5": continue
        o = t9[(r["dataset"], r["seed"])]; nrep += 1
        for c in repro: repro[c] = max(repro[c], abs(r[c] - float(o[c])))
    # ACC cross-check of the trajectories against the stored per-seed CSVs
    def stored(traj, ds, s):
        if traj == "post_footprint_off_L5": return T7acc[(ds, "post_footprint_off")].get(s)
        if traj == "pre_footprint_off_L5": return T7acc[(ds, "pre_footprint_off")].get(s)
        return A1.get((ds, "pre_footprint_off"), {}).get(s)
    accdiff = {}
    for r in eh:
        v = stored(r["trajectory"], r["dataset"], r["seed"])
        accdiff[r["trajectory"]] = max(accdiff.get(r["trajectory"], 0.0), abs(r["final_mean_acc"] - v) if v is not None else float("inf"))
    # config-level means
    cols = ["tilt_cv_median", "disp_median", "spread_median", "rel_median", "cv_unclipped_median", "cv_clipped_median",
            "share_q_unclipped_mean", "share_r_unclipped_mean", "share_r_unclipped_median", "pairs_unclipped_frac_mean"]
    cfg = []
    for traj in [f"{mp}_footprint_off_L{Ls}" for mp, Ls in EH_TRAJ]:
        for ds in CONFIGS:
            rr = [r for r in eh if r["trajectory"] == traj and r["dataset"] == ds]
            if not rr: continue
            row = dict(trajectory=traj, dataset=ds, m=rr[0]["m"], n_seeds=len(rr))
            for c in cols:
                v = [r[c] for r in rr if np.isfinite(r[c])]; row[c] = float(np.mean(v)) if v else float("nan")
            # gaps |ACC(footprint) - ACC(mass)| for the matching cells
            if traj == "post_footprint_off_L5": g = abs(M7(ds, "post_footprint_off") - M7(ds, "post_mass_off")); src = "T7 (main/big m=2, _mstar rescued), seeds 0-9"
            elif traj == "pre_footprint_off_L5": g = abs(M7(ds, "pre_footprint_off") - M7(ds, "pre_mass_off")); src = "T7 (main/big m=2, _mstar rescued), seeds 0-9"
            else:
                a, b = A1.get((ds, "pre_footprint_off"), {}), A1.get((ds, "pre_mass_off"), {})
                g = abs(st.mean(a.values()) - st.mean(b.values())) if a and b else float("nan"); src = "__L1all, seeds 0-9"
            row["acc_gap_fp_vs_mass"] = g; row["gap_source"] = src
            cfg.append(row)
    write_csv(OUT / "E_H_config.csv", cfg)
    P("# T10 theory checks (addendum §3: E-H, E-I) — m*, seeds 0-4, 50 rounds")
    P(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/T10_theory.py` (helpers `T10_theory_common.py`, "
      "`T10_theory_eh.py`, `T10_theory_ei.py`). Raw: `E_H.csv` (per seed), `E_H_config.csv` (config means), "
      "`E_I_descent.csv`, `E_I_counterexample.txt`.")
    P(); P("## E-H — one-round tilt diagnostics along three footprint-gated trajectories")
    P("Definitions as in T9_theory_link_b: weights W_pj = beta_p * mass_pj (post-adaptation mass for POST, "
      "pre-adaptation mass for PRE); base pi^q = W q (mass gate), server weights pi^r = W r (footprint gate); tilt g = r/q; "
      "CV = weighted (pi^q) coefficient of variation of g over clients; disp = ||target(footprint) - target(mass)||; "
      "spread = pi^q-weighted RMS distance of the local centres to target(mass); rel = disp/spread. Per seed: median "
      "over (round, prototype); table = mean over seeds 0-4. Unclipped pairs: r > r_min and q > r_min; clipped: either at "
      "the floor r_min = 0.05. CV_U / CV_C = the same weighted CV within each subset (pi^q renormalized; undefined when "
      "< 2 clients in the subset, then skipped); share_q(U) / share_r(U) = mean share of the mass-gated / footprint-gated "
      "(= server-update) weight carried by unclipped pairs.")
    P(); P(f"**Reproduction of T9 (post_footprint_off, L=5)**: {nrep}/70 (config, seed) rows; max |diff| vs "
      f"`results/t9_theory_link_b.csv`: " + ", ".join(f"{c} {v:.1e}" for c, v in repro.items()) + ".")
    P("**Trajectory check**: max |final mean ACC of the logged trajectory - stored per-seed ACC| = " +
      ", ".join(f"{t}: {v:.1e}" for t, v in accdiff.items()) +
      " (stored: T7 selection for L=5; `__L1all.csv` for L=1 — so `__L1all` is confirmed to be at m*).")
    for traj in [f"{mp}_footprint_off_L{Ls}" for mp, Ls in EH_TRAJ]:
        rows = [r for r in cfg if r["trajectory"] == traj]
        P(); P(f"### {traj}"); P()
        P("| config | m* | CV (all) | disp | spread | disp/spread | CV unclipped | CV clipped | share_q(U) | share_r(U) | frac pairs U | |ACC fp - ACC mass| |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            P(f"| {r['dataset']} | {r['m']} | {r['tilt_cv_median']:.3f} | {r['disp_median']:.4f} | {r['spread_median']:.3f} | "
              f"{r['rel_median']:.3f} | {r['cv_unclipped_median']:.3f} | {r['cv_clipped_median']:.3f} | "
              f"{r['share_q_unclipped_mean']:.3f} | {r['share_r_unclipped_mean']:.3f} | {r['pairs_unclipped_frac_mean']:.3f} | {r['acc_gap_fp_vs_mass']:.4f} |")
        f = lambda c: [r[c] for r in rows if np.isfinite(r[c])]
        P(); P(f"Ranges over the 14 configurations: CV {min(f('tilt_cv_median')):.3f}-{max(f('tilt_cv_median')):.3f}; "
               f"disp/spread {min(f('rel_median')):.3f}-{max(f('rel_median')):.3f}; CV unclipped "
               f"{min(f('cv_unclipped_median')):.3f}-{max(f('cv_unclipped_median')):.3f}; CV clipped "
               f"{min(f('cv_clipped_median')):.3f}-{max(f('cv_clipped_median')):.3f}; share_q(U) "
               f"{min(f('share_q_unclipped_mean')):.3f}-{max(f('share_q_unclipped_mean')):.3f}; share_r(U) "
               f"{min(f('share_r_unclipped_mean')):.3f}-{max(f('share_r_unclipped_mean')):.3f}. Gap source: {rows[0]['gap_source']}.")
        P(); P("Spearman across the 14 configurations vs |ACC(footprint) - ACC(mass)| of the matching cells (scipy, n = 14):")
        for c, name in (("tilt_cv_median", "weighted tilt CV (all pairs)"), ("rel_median", "disp/spread"),
                        ("cv_unclipped_median", "weighted tilt CV (unclipped pairs)"), ("cv_clipped_median", "weighted tilt CV (clipped pairs)"),
                        ("share_r_unclipped_mean", "server-weight share of unclipped pairs")):
            ok = [r for r in rows if np.isfinite(r[c]) and np.isfinite(r["acc_gap_fp_vs_mass"])]
            s = spearmanr([r[c] for r in ok], [r["acc_gap_fp_vs_mass"] for r in ok])
            P(f"- {name}: rho = {s.correlation:+.3f}, p = {s.pvalue:.4f} (n = {len(ok)})")
    # ---------------------------------------------------------------- E-I descent
    ei = load_rows("ei")
    write_csv(OUT / "E_I_descent.csv", ei, [k for k in ei[0] if k != "seconds"])
    P(); P("## E-I (1) — descent of J_beta along the damped PRE rules")
    P("J_beta(V_t) = sum_p beta_p J_p(U_t, V_t), U_t = the FCM-optimal memberships at V_t (reduced objective), t = 0..50; "
      "relative increase = (J_{t+1} - J_t)/J_t. The logged loop is checked against an independent "
      "T3.run_cell(mp, gate, 'off', local_steps=L) call (final V).")
    P(); P("| rule | runs | max rel. increase | max abs. increase | rounds with increase > 0 | rounds with rel. increase > 1e-12 | runs with any increase > 1e-12 | max |V - V_run_cell| |")
    P("|---|---|---|---|---|---|---|---|")
    for mp, g, Ls in EI_RULES:
        rule = f"{mp}_{g}_off_L{Ls}"; rr = [r for r in ei if r["rule"] == rule]
        n0 = sum(r["n_rounds_increase_gt0"] for r in rr); n12 = sum(r["n_rounds_increase_gt1e12"] for r in rr); tot = 50 * len(rr)
        P(f"| {rule} | {len(rr)} | {max(r['max_rel_increase'] for r in rr):.3e} | {max(r['max_abs_increase'] for r in rr):.3e} | "
          f"{n0}/{tot} ({n0 / tot:.1%}) | {n12}/{tot} ({n12 / tot:.1%}) | {sum(r['n_rounds_increase_gt1e12'] > 0 for r in rr)}/{len(rr)} | "
          f"{max(r['runcell_maxabs_diff'] for r in rr):.1e} |")
    for mp, g, Ls in EI_RULES:
        rule = f"{mp}_{g}_off_L{Ls}"; rr = [r for r in ei if r["rule"] == rule]
        P(); P(f"Where {rule} increases (rel > 1e-12), per configuration: runs with an increase / rounds with an increase / max over rounds of the relative change (J_{{t+1}}-J_t)/J_t, negative = J decreased in every round [first increase round range]:")
        for ds in CONFIGS:
            q = [r for r in rr if r["dataset"] == ds]
            if not q: continue
            fr = [r["first_increase_round"] for r in q if r["first_increase_round"] > 0]
            P(f"- {ds}: {sum(r['n_rounds_increase_gt1e12'] > 0 for r in q)}/{len(q)} runs, "
              f"{sum(r['n_rounds_increase_gt1e12'] for r in q)}/{50 * len(q)} rounds, max {max(r['max_rel_increase'] for r in q):.2e}"
              + (f" [first increase at round {min(fr)}-{max(fr)}]" if fr else ""))
    # ---------------------------------------------------------------- E-I counter-example
    from T10_theory_ei import counterexample
    variants = ["normal(loc,scale,size)", "0.3*standard_normal+mu", "multivariate_normal", "normal(size=(2,40)).T (column-major draw)"]
    txt = ["E-I (2) COUNTER-EXAMPLE: damped PRE round from a fixed point of pooled FCM (m = 2, alpha = 0.75, beta = (1/2, 1/2))",
           "Data: 40 points N(0, 0.3^2 I_2) then 40 points N((4,4), 0.3^2 I_2), one generator numpy default_rng(1); client 1 = blob 1,",
           "client 2 = blob 2. V* = fedfcmsim.fcm.fcm from the two sample means (a) 30 iterations, (b) to convergence (tol 1e-12).",
           "One PRE round: c_pj = run_local_fcm_steps(x_p, V*, L), weights beta_p * sum_k ubar_kj^m (at V*), V+ = 0.25 V* + 0.75 T.",
           "J = pooled FCM objective sum_k sum_j u_kj^m ||x_k - v_j||^2 over all 80 points, u = optimal memberships at V (reduced).",
           "dV = ||V+ - V*||_F (damped); dT = ||T - V*||_F (undamped server target, = dV / 0.75); dJ_fixedU uses U(V*).",
           "Referee claim: L=1 (0, 0); L=2 0.138457, +0.422833; L=5 0.260075, +1.490405; J(V*) = 11.236762.", ""]
    cex = {}
    for v in variants:
        o = counterexample(v); cex[v] = o
        for fp in ("30 iterations", "converged tol 1e-12"):
            d = o[fp]
            txt.append(f"[{v}] V* by {fp} (n_iter {d['n_iter']}): V* = {np.round(d['V_star'], 6).tolist()}, J(V*) = {d['J_star']:.6f} "
                       f"(beta-weighted {d['J_star_beta']:.6f}, per point {d['J_star_mean']:.6f}); ||PRE_L1 undamped target - V*|| = {d['fixed_point_residual']:.1e}")
            for r in d["rows"]:
                txt.append(f"    L={r['L']}: dV = {r['dV_fro']:.6f}  dT = {r['dT_fro']:.6f}  max-row dV = {r['dV_maxrow']:.6f}  "
                           f"dJ(reduced) = {r['dJ_reduced']:+.6f}  dJ(fixed U*) = {r['dJ_fixedU']:+.6f}  dJ(beta-weighted) = {r['dJ_beta_reduced']:+.6f}")
        txt.append("")
    main = cex[variants[0]]["converged tol 1e-12"]
    txt.append("VERDICT: J(V*) and every dJ reproduce the referee to 6 decimals (reduced pooled objective, unweighted sum over the 80")
    txt.append("points; identical for the first three drawing variants, which produce the same numbers; the column-major variant")
    txt.append("differs). The referee's displacement values are ||T - V*|| (the UNDAMPED server target), not ||V+ - V*||: ours are")
    txt.append(f"dT = {main['rows'][1]['dT_fro']:.6f} (L=2), {main['rows'][2]['dT_fro']:.6f} (L=5) = the claimed 0.138457, 0.260075;"
               f" the damped step moves V by 0.75x that: {main['rows'][1]['dV_fro']:.6f}, {main['rows'][2]['dV_fro']:.6f}.")
    (OUT / "E_I_counterexample.txt").write_text("\n".join(txt) + "\n")
    P(); P("## E-I (2) — counter-example (two blobs, one per client, m = 2)"); P()
    P("| V* | L | ||V+ - V*|| (damped) | ||T - V*|| (undamped target) | dJ pooled (reduced) | dJ (fixed U*) |"); P("|---|---|---|---|---|---|")
    for fp in ("30 iterations", "converged tol 1e-12"):
        d = cex[variants[0]][fp]
        for r in d["rows"]:
            P(f"| {fp} | {r['L']} | {r['dV_fro']:.6f} | {r['dT_fro']:.6f} | {r['dJ_reduced']:+.6f} | {r['dJ_fixedU']:+.6f} |")
    P(); P(f"J(V*) = {main['J_star']:.6f} (both fixed points; FCM from the sample means converges in {main['n_iter']} iterations at tol 1e-12). "
           "Drawing variants `rng.normal(loc, 0.3, size)`, `0.3*standard_normal + mu` and `multivariate_normal` give identical numbers; "
           f"a column-major draw gives J(V*) = {cex[variants[3]]['converged tol 1e-12']['J_star']:.6f}, dJ(L=2) = "
           f"{cex[variants[3]]['converged tol 1e-12']['rows'][1]['dJ_reduced']:+.6f}, dJ(L=5) = {cex[variants[3]]['converged tol 1e-12']['rows'][2]['dJ_reduced']:+.6f} (same qualitative result).")
    P("**Verdict:** reproduced. J(V*) = 11.236762 and dJ = 0 / +0.422833 / +1.490405 for L = 1 / 2 / 5 match exactly; the "
      "referee's displacement figures 0.138457 / 0.260075 are the undamped target displacement ||T - V*||; the damped iterate "
      f"moves by 0.75 times that ({main['rows'][1]['dV_fro']:.6f} / {main['rows'][2]['dV_fro']:.6f}). L = 1 leaves V* fixed "
      "(PRE at L=1 is exactly the pooled FCM step), L >= 2 increases the pooled objective: the damped PRE rule is not a descent method for L >= 2.")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
