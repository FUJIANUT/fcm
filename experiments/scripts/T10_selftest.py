"""T10 self-test. Seeds used: 0-1 and 10-11 only (plus the calibration seed 100 in test vii).

  (i)    run_cell_pp with participation 1.0 == T3.run_cell bit for bit (2 configs x 2 seeds x 5 cells, L=5 and L=1);
         the round-counting wrapper run_cell_counted == T3.run_cell bit for bit and its n_rounds == run_cell_pp's;
         early stopping (n_rounds < R) occurs for personalized cells and is recorded in the cells CSV
  (ii)   'cells' reproduces the prior design-space CSVs bit for bit (L=5: main[m*=2 configs] / __mstar / __confirm /
         t6 __mstar; L=1: __L1all / __T9_L1_gf / __T9_L1_confirm) on wine, cluster_skew_hard, digits_pca16,
         seeds 0-1 and 10-11, every cell present in those files; plus the provenance check that the T9 L=1 files'
         personalized rows are reproduced in ALL fields with FFCM_NO_EARLYSTOP=1, and the same for the L=5 files
         T9_noES (s0-9) / T9_noES_confirm (s10-19)
  (iii)  'scffcm' with the t9b steps reproduces results/t9b_scffcm_fair.csv (2 configs, seeds 0-1, L in {1,5})
  (iv)   'cfcm' reproduces T6's centralized_fcm row (all 14 configs, seeds 0-1)
  (v)    'audit' init (a) reproduces paper2/figures/degeneracy_data.csv (digits_pca16, wine; all m, seeds 0-4)
  (vi)   gacc == mean_acc with one client holding all data; xb and hnorm on a toy example computed by hand;
         gacc on a two-client toy vs brute-force label matching
  (vii)  'scffcm-calibrate' on the t9b grid reproduces results/t9b_scffcm_calibration.csv (2 configs, L in {1,5})
  (viii) mask gates: registered in T3.GATE_FNS, values in {r_min, 1}, equal to the thresholded T3 gates;
         cell parser accepts the 16 cells
  (ix)   pedrycz_gradient_fcm uses the fuzzifier m (no hard-coded squares)
  (x)    split-sample partition: disjoint, covering, 80/20, deterministic; fresh-seed guards (seeds, --cal-seed);
         no_stable_setting placeholder rows carry m / n_rounds / participation
  (xi)   E-C held-out ACC: the train-fitted label map reproduces clustering_accuracy on the points it was fitted on;
         a 1-point test set scores 1.0 under the re-fitted map but not necessarily under the train map; a split run
         writes every held-out column
  (xii)  analysis: exact Wilcoxon == explicit enumeration of the 2^n' sign assignments (ties, zeros) and == scipy's
         exact test without ties; Holm and t-interval by hand; the analysis subcommand on the PRIOR CSVs reproduces
         the documented (legacy) counts of T8 E7 / T9 R2b-R2c / T9 R3 on seeds 0-9 and 10-19
  (xiii) guards: compare fails on zero overlap, on an uncovered prior row (label drift) and below --min-runs;
         argcheck accepts the same command and rejects a different one
  (xiv)  the analysis subcommand on T10-format CSVs (the (ii) outputs for seeds 10-11 + SC-FFCM runs, saved under the
         plan's tags): every arm is found, the decision analysis covers all five criteria incl. uploaded numbers

Scratch outputs: results/t10/selftest/ (overwritten on every run). Exit status 1 if any test fails.
"""
from __future__ import annotations

import csv
import inspect
import itertools
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import T10_review as R  # noqa: E402  (does not import T3)

SCR = R.OUT_DIR / "selftest"
PY = sys.executable
DS3 = ["wine", "cluster_skew_hard", "digits_pca16"]
RESULTS: list[tuple[str, bool, str]] = []


def record(name, ok, info=""):
    RESULTS.append((name, bool(ok), info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}", flush=True)


def run(*args):
    cmd = [PY, str(HERE / "T10_review.py"), *args, "--outdir", str(SCR), "--force"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise RuntimeError(f"command failed: {' '.join(args)}")
    return time.time() - t0


def compare(kind, tag):
    return R.run_compare(kind, SCR / f"{tag}.csv")


def compare_full(kind, tag):
    return R.run_compare_full(kind, SCR / f"{tag}.csv")


def summarize(res):
    """strict rows only (advisory acc-only rows across stopping rules are counted separately by the callers)."""
    res = [r for r in res if r.get("strict", 1)]
    bad = [r for r in res if not r["match"]]
    runs = {(r["source"], r["key"]) for r in res}
    n_acc = sum(1 for r in res if r["field"] == "mean_acc")
    return bad, runs, n_acc


# ------------------------------------------------------------------------------------------------ (i)
def test_i():
    R.apply_env(R.protocol_env(5, 50))
    T = R.get_T()
    cells = [("post", "none", "off"), ("pre", "mass", "off"), ("post", "footprint", "on"),
             ("pre", "mass", "on"), ("pre", "fpmask", "off")]
    n, bad, nrs = 0, [], []
    for L in (5, 1):
        for label in ("wine", "digits_pca16"):
            for seed in (0, 1):
                _, clients, k, X, init = R.setup(label, seed)
                for c in cells:
                    V1, e1, _ = T.run_cell(clients, init, *c, local_steps=L)
                    V2, e2, ex = R.run_cell_pp(T, clients, init, *c, 1.0, T.SEED_OFFSET + seed + 555, local_steps=L)
                    ok = np.array_equal(V1, V2) and all(np.array_equal(a, b) for a, b in zip(e1, e2))
                    if L == T.LOCAL_STEPS:     # run_cell_counted uses T3's default L (the environment's)
                        orig = T.GATE_FNS[c[1]]
                        V3, e3, nr = R.run_cell_counted(T, clients, init, *c)
                        ok &= np.array_equal(V1, V3) and all(np.array_equal(a, b) for a, b in zip(e1, e3))
                        ok &= nr == ex["n_rounds"] and T.GATE_FNS[c[1]] is orig     # wrapper removed again
                        nrs.append((c[2], nr))
                    n += 1
                    if not ok:
                        bad.append((L, label, seed, c))
    early = sorted({nr for pe, nr in nrs if pe == "on"})
    full = {nr for pe, nr in nrs if pe == "off"}
    record("(i) participation-1.0 copy == T3.run_cell; round counter", not bad and full == {50} and min(early) < 50,
           f"{n} runs (L in {{5,1}} x wine,digits_pca16 x seeds 0,1 x {len(cells)} cells incl. 2 personalized), "
           f"{len(bad)} differ {bad[:3]}; run_cell_counted identical, n_rounds == run_cell_pp's; n_rounds of the "
           f"personalized L=5 runs {early} (< 50 = early stop), non-personalized {sorted(full)}")
    # partial participation sanity: the right number of distinct clients, and a different trajectory
    _, clients, k, X, init = R.setup("wine", 0)
    V1, _, _ = T.run_cell(clients, init, "pre", "mass", "off")
    V2, _, ex = R.run_cell_pp(T, clients, init, "pre", "mass", "off", 0.5, T.SEED_OFFSET + 555)
    V3, _, _ = R.run_cell_pp(T, clients, init, "pre", "mass", "off", 0.5, T.SEED_OFFSET + 555)
    record("(i') participation 0.5 samples max(1,int(P*0.5)) clients, deterministic, differs from full",
           ex["n_part"] == 5 and np.array_equal(V2, V3) and not np.array_equal(V1, V2),
           f"n_part={ex['n_part']} of {len(clients)}")


# ------------------------------------------------------------------------------------------------ (ii)
def prior_keys(L, seeds, no_es=False):
    """Every (source, dataset, seed, cell) the prior files hold for DS3 x seeds."""
    mstar = R._mstar_table()
    keys = set()
    for path, kind in R.PRIOR_CELLS[(L, 50)]:
        for p in csv.DictReader(path.open()):
            if p["dataset"] not in DS3 or int(p["seed"]) not in seeds:
                continue
            if kind == "m2only" and mstar.get(p["dataset"], 2.0) != 2.0:
                continue
            cell = p.get("cell") or p.get("method")
            if kind == "t6" and cell not in R.T3_CELL_NAMES:
                continue
            keys.add((f"{path.parent.name}/{path.name}", p["dataset"], int(p["seed"]), cell))
    return keys


def test_ii():
    tot_t = 0.0
    for L in (5, 1):
        seeds = [0, 1, 10, 11]
        tags = []
        for ss in (0, 10):
            tag = f"st_cells_L{L}_s{ss:02d}"
            tot_t += run("cells", "--datasets", ",".join(DS3), "--seed-start", str(ss), "--seeds", "2", "--L", str(L),
                         "--cells", "all16", "--workers", "6", "--tag", tag)
            tags.append(tag)
        res, unc = [], []
        for tag in tags:
            a, b = compare_full("cells", tag)
            res += a
            unc += b
        bad, runs, n_acc = summarize(res)
        adv = [r for r in res if not r["strict"]]
        adv_runs = {(r["source"], r["key"]) for r in adv}
        adv_bad = [r for r in adv if not r["match"]]
        want = prior_keys(L, seeds)
        got = {(r["source"].split(" [")[0], *r["key"].split("|")) for r in res}
        got = {(s, d, int(se), c) for s, d, se, c in got}
        missing = want - got
        srcs = sorted({r["source"].split(" [")[0] for r in res})
        record(f"(ii) cells L={L} vs prior CSVs", not bad and not missing and not unc and n_acc > 0,
               f"{len(runs)} strict overlapping runs, {n_acc} mean_acc values, {len(bad)} mismatches, "
               f"{len(missing)} + {len(unc)} prior rows not covered; advisory (prior noES vs new early stop, acc "
               f"only): {len(adv_runs)} runs, {len(adv_bad)} acc differences; sources {srcs}")
        for b in bad[:5]:
            print("     mismatch", b)
        # n_rounds is recorded for every cells row; personalized rows early-stop (< R) at least once
        rows = [r for t in tags for r in csv.DictReader((SCR / f"{t}.csv").open())]
        nr_on = sorted({int(r["n_rounds"]) for r in rows if r["personalize"] == "on"})
        nr_off = {int(r["n_rounds"]) for r in rows if r["personalize"] == "off"}
        ups = all(int(r["upload_numbers"]) == int(r["n_rounds"]) * int(r["n_part"]) * int(r["n_clusters"])
                  * (int(r["dim"]) + 1) for r in rows)
        record(f"(ii-n) cells L={L}: n_rounds / upload_numbers recorded in every row",
               all(r["n_rounds"] != "" for r in rows) and nr_off == {50} and min(nr_on) < 50 and ups,
               f"{len(rows)} rows; personalized n_rounds in {nr_on[:3]}..{nr_on[-1:]} (min {min(nr_on)}), "
               f"non-personalized {sorted(nr_off)}; upload_numbers = n_rounds*n_part*c*(d+1) on all rows")
    # provenance: the T9 L=1 GF files and the T9 L=5 noES files: personalized rows need FFCM_NO_EARLYSTOP=1
    for L, cells in ((1, "post_footprint_on"), (5, "post_footprint_on,pre_footprint_on,pre_mass_on,post_mass_on")):
        res = []
        for ss in (0, 10):
            tag = f"st_cells_L{L}_noES_s{ss:02d}"
            tot_t += run("cells", "--datasets", ",".join(DS3), "--seed-start", str(ss), "--seeds", "2", "--L", str(L),
                         "--cells", cells, "--no-earlystop", "--workers", "6", "--tag", tag)
            res += compare("cells", tag)
        bad, runs, _ = summarize(res)
        full = [r for r in res if "noES both" in r["source"]]
        files = "T9_L1_gf / T9_L1_confirm" if L == 1 else "T9_noES / T9_noES_confirm"
        record(f"(ii') {files} personalized rows were run with FFCM_NO_EARLYSTOP=1 (L={L})",
               not bad and {r["field"] for r in full} >= {"min_dist", "fcm_obj", "mean_acc"} and len(runs) > 0,
               f"{len(runs)} runs, {len(res)} values incl. min_dist/fcm_obj, {len(bad)} mismatches")
        early = []
        for tag in (f"st_cells_L{L}_s00", f"st_cells_L{L}_s10"):
            new = {(r["dataset"], r["seed"], r["cell"]): r for r in csv.DictReader((SCR / f"{tag}.csv").open())
                   if r["personalize"] == "on" and r["cell"] in cells.split(",")}
            noes = {(r["dataset"], r["seed"], r["cell"]): r for t in (f"st_cells_L{L}_noES_s00", f"st_cells_L{L}_noES_s10")
                    for r in csv.DictReader((SCR / f"{t}.csv").open())}
            for k2, r in new.items():
                early.append((float(r["mean_acc"]) == float(noes[k2]["mean_acc"]),
                              abs(float(r["min_dist"]) - float(noes[k2]["min_dist"]))))
        print(f"     note: early-stop vs no-early-stop at L={L} on these {len(early)} personalized runs: mean_acc equal"
              f" on {sum(e[0] for e in early)}/{len(early)}, max |d min_dist| = {max(e[1] for e in early):.2e}")
    return tot_t


# ------------------------------------------------------------------------------------------------ (iii)
def test_iii():
    t = 0.0
    res = []
    for L in (1, 5):
        tag = f"st_scffcm_L{L}"
        t += run("scffcm", "--L", str(L), "--steps-csv", str(R.RESULTS / "t9b_scffcm_calibration.csv"),
                 "--datasets", "wine,digits_pca16", "--seeds", "2", "--workers", "4", "--tag", tag)
        res += compare("scffcm", tag)
    bad, runs, n_acc = summarize(res)
    record("(iii) scffcm (t9b steps) vs t9b_scffcm_fair.csv", not bad and len(runs) == 8,
           f"{len(runs)}/8 runs, {len(res)} values ({n_acc} mean_acc), {len(bad)} mismatches")
    return t


# ------------------------------------------------------------------------------------------------ (iv)
def test_iv():
    t = run("cfcm", "--seeds", "2", "--workers", "12", "--tag", "st_cfcm")
    res = compare("cfcm", "st_cfcm")
    bad, runs, n_acc = summarize(res)
    record("(iv) cfcm vs T6 centralized_fcm", not bad and len(runs) == 28,
           f"{len(runs)}/28 runs (14 configs x seeds 0-1), {len(res)} values ({n_acc} mean_acc), {len(bad)} mismatches")
    return t


# ------------------------------------------------------------------------------------------------ (v)
def test_v():
    t = run("audit", "--datasets", "digits_pca16,wine", "--workers", "12", "--tag", "st_audit")
    res = compare("audit", "st_audit")
    bad, runs, _ = summarize(res)
    record("(v) audit (a) vs degeneracy_data.csv", not bad and len(runs) == 50,
           f"{len(runs)}/50 runs (2 configs x 5 m x 5 seeds), fields distinct/entropy_norm/n/d/c, {len(bad)} mismatches")
    rows = list(csv.DictReader((SCR / "st_audit.csv").open()))
    fed = [r for r in rows if r["init"] == "fed_lossless_pre1"]
    ms = list(csv.DictReader((SCR / "st_audit_mstar.csv").open()))
    print(f"     fed lossless: max_iter-wise |V_fed - V_cen| = "
          f"{max(float(r['maxgap_fed_vs_cen_iter']) for r in fed):.2e}; m* agrees with paper on "
          f"{sum(int(r['agrees']) for r in ms)}/{len(ms)} (dataset, init) pairs")
    return t


# ------------------------------------------------------------------------------------------------ (vi)
def test_vi():
    from fedfcmsim.synthetic import ClientData
    R.apply_env(R.protocol_env(5, 50))
    T = R.get_T()
    # gacc == mean_acc with one client holding all data (wine and digits_pca16, a few prototype sets)
    ok_all, n = True, 0
    for label in ("wine", "digits_pca16"):
        for seed in (0, 1):
            _, clients, k, X, init = R.setup(label, seed)
            one = [ClientData(x=X, y=np.concatenate([c.y for c in clients]), name="all")]
            for mp, g, pe in [("pre", "mass", "off"), ("post", "none", "off")]:
                V, _, _ = T.run_cell(clients, init, mp, g, pe)
                for Vx in (V, init):
                    m = R.eval_metrics(T, one, Vx, [Vx])
                    ok_all &= m["gacc"] == m["mean_acc"] == m["worst_acc"]
                    n += 1
    record("(vi-a) gacc == mean_acc when one client holds all data", ok_all, f"{n} prototype sets")
    # toy by hand, m = 2
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    V = np.array([[0.0, 0.0], [2.0, 2.0]])
    # memberships by hand (u_ij = (1/d2_ij) / sum_k (1/d2_ik)):  (1,0), (5/6,1/6), (1/2,1/2), (1/10,9/10)
    num = 0.0 + (25 / 36 * 1 + 1 / 36 * 5) + (0.25 * 4 + 0.25 * 4) + (0.01 * 18 + 0.81 * 2)   # = 4.6333...
    xb_hand = num / (4 * 8.0)                                                                 # min ||v1-v2||^2 = 8
    ent = lambda p: -sum(q * math.log(q) for q in p if q > 0)                                  # noqa: E731
    h_hand = (ent([1, 0]) + ent([5 / 6, 1 / 6]) + ent([0.5, 0.5]) + ent([0.1, 0.9])) / 4 / math.log(2)
    xb = R.xie_beni(X, V, 2.0)
    toy = [ClientData(x=X[:2], y=np.array([0, 0]), name="a"), ClientData(x=X[2:], y=np.array([1, 1]), name="b")]
    gacc, hn = R.pooled_eval(toy, [V, V], 2.0)
    record("(vi-b) xb and hnorm on the toy vs hand values", abs(xb - xb_hand) < 1e-12 and abs(hn - h_hand) < 1e-12,
           f"xb={xb:.12f} (hand {xb_hand:.12f}), hnorm={hn:.12f} (hand {h_hand:.12f})")
    # gacc vs brute-force best label permutation; personalized-looking prototypes (client b gets swapped/shifted ones)
    Vb = np.array([[2.5, 2.5], [0.2, 1.9]])
    gacc2, _ = R.pooled_eval(toy, [V, Vb], 2.0)
    from fedfcmsim.fcm import predict_membership
    pred = np.concatenate([np.argmax(predict_membership(X[:2], V, 2.0), 1), np.argmax(predict_membership(X[2:], Vb, 2.0), 1)])
    y = np.array([0, 0, 1, 1])
    brute = max(np.mean(np.array([perm[p] for p in pred]) == y) for perm in itertools.permutations(range(2)))
    record("(vi-c) gacc (own-client prototypes, one Hungarian) vs brute force", gacc2 == brute,
           f"gacc={gacc2}, brute={brute}, pred={pred.tolist()}")


# ------------------------------------------------------------------------------------------------ (vii)
def test_vii():
    t = 0.0
    prior = {(r["dataset"], int(r["L"])): r for r in csv.DictReader((R.RESULTS / "t9b_scffcm_calibration.csv").open())}
    bad, n = [], 0
    for L in (1, 5):
        tag = f"st_scffcm_cal_t9b_L{L}"
        t += run("scffcm-calibrate", "--L", str(L), "--grid", "t9b", "--datasets", "wine,cluster_skew_hard",
                 "--workers", "12", "--tag", tag)
        for r in csv.DictReader((SCR / f"{tag}.csv").open()):
            p = prior[(r["dataset"], L)]
            n += 1
            for f in ("eta_l", "eta_g", "rel_obj_gap", "n_stable"):
                if float(r[f]) != float(p[f]):
                    bad.append((r["dataset"], L, f, r[f], p[f]))
    record("(vii) scffcm-calibrate (t9b grid, seed 100) vs t9b_scffcm_calibration.csv", not bad and n == 4,
           f"{n}/4 (dataset, L) rows, fields eta_l/eta_g/rel_obj_gap/n_stable, {len(bad)} mismatches {bad[:2]}")
    return t


# ------------------------------------------------------------------------------------------------ (viii)
def test_viii():
    R.apply_env(R.protocol_env(5, 50))
    T = R.get_T()
    from fedfcmsim.fcm import predict_membership
    ok = T.GATE_FNS.get("fpmask") is R.gate_fpmask and T.GATE_FNS.get("massmask") is R.gate_massmask
    n = 0
    for label in ("wine", "digits_pca16", "cluster_skew_hard"):
        _, clients, k, X, init = R.setup(label, 0)
        for c in clients:
            ub = predict_membership(c.x, init, m=T.M)
            fp, ms = T.gate_footprint(c.x, init, ub), T.gate_mass(c.x, init, ub)
            a, b = R.gate_fpmask(c.x, init, ub), R.gate_massmask(c.x, init, ub)
            ok &= set(np.unique(a)) <= {T.MIN_RELEVANCE, 1.0} and set(np.unique(b)) <= {T.MIN_RELEVANCE, 1.0}
            ok &= np.array_equal(a == 1.0, fp > T.MIN_RELEVANCE) and np.array_equal(b == 1.0, ms > T.MIN_RELEVANCE)
            n += 1
    names = [R.cell_name(c) for c in R.parse_cells("all16")]
    ok &= len(names) == 16 and R.parse_cells("pre_fpmask_off,post_massmask_off,GF") == [
        ("pre", "fpmask", "off"), ("post", "massmask", "off"), ("post", "footprint", "on")]
    record("(viii) mask gates and cell parser", ok, f"{n} clients checked; cells: {', '.join(names[12:])}")


# ------------------------------------------------------------------------------------------------ (ix)
def test_ix():
    from fedfcmsim.federated import pedrycz_gradient_fcm
    src = inspect.getsource(pedrycz_gradient_fcm)
    body = src.split('"""')[2]   # code after the docstring
    hard = [s for s in ("**2", "** 2", "np.square") if s in body]
    uses_m = "membership**m" in body and "m=m" in body
    record("(ix) pedrycz_gradient_fcm uses m (no hard-coded square)", uses_m and not hard,
           f"'membership**m' and run_local_fcm_steps(..., m=m) present; hard-coded squares: {hard or 'none'}")


# ------------------------------------------------------------------------------------------------ (x)
def test_x():
    R.apply_env(R.protocol_env(5, 50))
    T = R.get_T()
    ok, info = True, []
    for label in ("wine", "cluster_skew_hard"):
        _, clients, k, X, init = R.setup(label, 0)
        tr, te = R.split_clients(T, clients, 0, 0.2)
        tr2, te2 = R.split_clients(T, clients, 0, 0.2)
        for c, a, b, a2 in zip(clients, tr, te, tr2):
            rows_c = {tuple(r) for r in c.x}
            ra, rb = {tuple(r) for r in a.x}, {tuple(r) for r in b.x}
            ok &= ra.isdisjoint(rb) and (ra | rb) == rows_c and len(a.x) + len(b.x) == len(c.x)
            ok &= np.array_equal(a.x, a2.x)
        info.append(f"{label}: test share {sum(len(b.x) for b in te) / len(X):.3f}")

    class A:  # fresh-seed guard
        seed_start, seeds, allow_fresh = 18, 3, False
    try:
        R.seed_list(A())
        ok = False
    except SystemExit:
        pass

    class B:  # --cal-seed guard
        cal_seed, allow_fresh = 25, False
    try:
        R.check_cal_seed(B())
        ok = False
    except SystemExit:
        pass
    B.cal_seed = 100
    R.check_cal_seed(B())
    ph = R.placeholder_row(R.protocol_env(5, 50), "digits_pca16", 0, "scffcm", 5, 50, cfraction=1.0, participation=1.0)
    ok &= ph["m"] == T.fuzzifier_for("digits_pca16") and ph["n_rounds"] == -1 and math.isnan(ph["mean_acc"])
    record("(x) split-sample partition + fresh-seed guards + placeholder rows", ok,
           "; ".join(info) + "; seeds 18-20 and --cal-seed 25 refused without --allow-fresh, --cal-seed 100 accepted; "
           f"no_stable_setting row: m={ph['m']}, n_rounds={ph['n_rounds']}, participation={ph['participation']}")


# ------------------------------------------------------------------------------------------------ (xi)
def test_xi():
    from fedfcmsim.metrics import clustering_accuracy
    rng = np.random.default_rng(0)
    ok, n = True, 0
    for _ in range(300):                      # random label / cluster vectors incl. unequal label sets
        k1, k2, m = rng.integers(1, 6), rng.integers(1, 6), rng.integers(1, 40)
        y, p = rng.integers(0, k1, m), rng.integers(0, k2, m)
        ok &= abs(R.mapped_acc(R.hungarian_map(y, p), y, p) - clustering_accuracy(y, p)) < 1e-15
        n += 1
    # the bias the reviewer found: one test point always scores 1.0 when the map is re-fitted on it
    one_refit = clustering_accuracy(np.array([3]), np.array([0]))
    mp = R.hungarian_map(np.array([0, 0, 3, 3]), np.array([0, 0, 1, 1]))      # train: cluster 0 -> label 0
    one_tr = R.mapped_acc(mp, np.array([3]), np.array([0]))                   # test point of label 3 in cluster 0
    ok &= one_refit == 1.0 and one_tr == 0.0
    t = run("cells", "--datasets", "wine,quantity_skew_extreme", "--seeds", "1", "--L", "5", "--split", "0.2",
            "--cells", "post_footprint_off,post_footprint_on", "--workers", "2", "--tag", "st_split")
    rows = list(csv.DictReader((SCR / "st_split.csv").open()))
    cols = set(R.TEST_KEYS) | {"n_test", "min_client_n_test", "n_clients_test_le5"}
    ok &= all(all(r.get(c, "") != "" for c in cols) for r in rows)
    ok &= all(float(r["test_mean_acc_refit"]) >= float(r["test_mean_acc_trmap"]) - 1e-12 for r in rows)
    info = "; ".join(f"{r['dataset'][:12]} {r['cell']}: test ACC refit {float(r['test_mean_acc_refit']):.3f} vs "
                     f"train-map {float(r['test_mean_acc_trmap']):.3f}, worst {float(r['test_worst_acc_refit']):.3f} vs "
                     f"{float(r['test_worst_acc_trmap']):.3f}, min client n_test {r['min_client_n_test']}" for r in rows)
    record("(xi) E-C held-out ACC with the train-fitted label map", ok,
           f"{n} random cases map == clustering_accuracy; 1-point test set: refit {one_refit}, train map {one_tr}; "
           f"{info}")
    return t


# ------------------------------------------------------------------------------------------------ (xii)
def brute_wilcoxon(d):
    from scipy.stats import rankdata
    d = np.asarray([x for x in d if abs(x) > R.ZERO_TOL], float)
    n = len(d)
    if n == 0:
        return 1.0
    r = rankdata(np.round(np.abs(d), R.TIE_DECIMALS))
    w = r[d > 0].sum()
    e = r.sum() / 2
    cnt = 0
    for signs in itertools.product((0, 1), repeat=n):
        if abs(sum(ri for ri, s in zip(r, signs) if s) - e) >= abs(w - e) - 1e-9:
            cnt += 1
    return cnt / 2 ** n


def test_xii():
    from scipy.stats import wilcoxon, t as tdist
    rng = np.random.default_rng(1)
    ok, n, maxdiff, n_sc = True, 0, 0.0, 0
    for _ in range(200):
        m = int(rng.integers(1, 12))
        d = rng.choice([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.05], m) if rng.random() < 0.5 else rng.normal(size=m)
        pe, pb = R.wilcoxon_exact(d)["p"], brute_wilcoxon(d)
        maxdiff = max(maxdiff, abs(pe - pb))
        ok &= abs(pe - pb) < 1e-12
        n += 1
        if len(set(np.abs(d))) == m and np.all(d != 0) and m >= 1:      # no ties / zeros: scipy's exact test
            ps = wilcoxon(d, method="exact").pvalue
            ok &= abs(pe - ps) < 1e-12
            n_sc += 1
    ok &= abs(R.wilcoxon_exact(np.arange(1, 11, dtype=float))["p"] - 1 / 512) < 1e-15
    h = R.holm([0.01, 0.04, 0.03, 0.005])
    ok &= np.allclose(h, [0.03, 0.06, 0.06, 0.02])
    d = [0.1, 0.3, -0.2, 0.4, 0.05]
    mean, lo, hi = R.t_interval(d)
    lo2, hi2 = tdist.interval(0.95, 4, loc=np.mean(d), scale=np.std(d, ddof=1) / np.sqrt(5))
    ok &= abs(lo - lo2) < 1e-12 and abs(hi - hi2) < 1e-12
    record("(xii-a) exact Wilcoxon / Holm / t-interval", ok,
           f"{n} random vectors (ties, zeros, n' <= 11): exact == 2^n' enumeration (max |dp| {maxdiff:.1e}); "
           f"== scipy exact on {n_sc} tie-free vectors; p(n'=10, all positive) = 1/512; Holm and t-interval by hand")
    # the analysis subcommand on the PRIOR CSVs reproduces the documented counts (legacy scipy test)
    t0 = time.time()
    subprocess.run([PY, str(HERE / "T10_review.py"), "analyze", "--source", "prior", "--tag", "st_analysis_prior",
                    "--outdir", str(SCR), "--force"], check=True, capture_output=True, text=True)
    summ = {(r["block"], r["hyp"]): r for r in csv.DictReader((SCR / "st_analysis_prior_summary.csv").open())
            if r["metric"] == "mean_acc"}
    doc = {("s10-19", "H1"): (8, 1), ("s10-19", "H2"): (11, 1), ("s10-19", "H3"): (9, 1),   # T8 E7 (sec. 1, 3)
           ("s10-19", "H4"): (12, 1), ("s10-19", "H5"): (4, 0), ("s00-09", "H5"): (7, 0),    # T9 R3
           ("s10-19", "H6"): (6, 2), ("s00-09", "H6"): (4, 3), ("s10-19", "H7"): (6, 2),     # T9 R2b/R2c
           ("s00-09", "H7"): (4, 1), ("s10-19", "H8"): (0, 6), ("s10-19", "S-L50"): (4, 2),
           ("s00-09", "S-L50"): (3, 1), ("s10-19", "S-PRE-SC1"): (0, 2)}
    bad = [(k, v, (int(summ[k]["better_legacy"]), int(summ[k]["worse_legacy"]))) for k, v in doc.items()
           if (int(summ[k]["better_legacy"]), int(summ[k]["worse_legacy"])) != v]
    same = sum(1 for k in doc if (summ[k]["better_legacy"], summ[k]["worse_legacy"]) ==
               (summ[k]["better_unadj"], summ[k]["worse_unadj"]))
    record("(xii-b) analysis on the prior CSVs reproduces the documented counts", not bad,
           f"{len(doc)} documented better/worse counts (T8 E7, T9 R2b/R2c/R3), {len(bad)} differ {bad[:3]}; "
           f"exact test gives the same unadjusted counts on {same}/{len(doc)}")
    return time.time() - t0


# ------------------------------------------------------------------------------------------------ (xiv)
def test_xiv():
    import shutil
    t0 = time.time()
    d = SCR / "ana_t10"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for L in (5, 1):
        shutil.copy(SCR / f"st_cells_L{L}_s10.csv", d / f"cells16_L{L}_s10-19.csv")
    # H5 (primary) uses GF at L=1 without early stop (pre-registration addendum 2): the plan's gf_L1_noES arm
    shutil.copy(SCR / "st_cells_L1_noES_s10.csv", d / "gf_L1_noES_s00-19.csv")
    for L in (5, 1):
        subprocess.run([PY, str(HERE / "T10_review.py"), "scffcm", "--L", str(L), "--steps-csv",
                        str(R.RESULTS / "t9b_scffcm_calibration.csv"), "--datasets", ",".join(DS3), "--seed-start",
                        "10", "--seeds", "2", "--workers", "6", "--tag", f"scffcm_cal_L{L}_s00-19", "--outdir", str(d),
                        "--force"], check=True, capture_output=True, text=True)
    subprocess.run([PY, str(HERE / "T10_review.py"), "analyze", "--source", "t10", "--t10-dir", str(d), "--blocks",
                    "s10-19", "--tag", "st_analysis_t10", "--outdir", str(SCR), "--force"],
                   check=True, capture_output=True, text=True)
    dec = list(csv.DictReader((SCR / "st_analysis_t10_decision.csv").open()))
    tests = list(csv.DictReader((SCR / "st_analysis_t10_tests.csv").open()))
    crit_ok = {r["criterion"] for r in dec if r.get("status") == "ok"}
    h_ok = {r["hyp"] for r in tests if r["metric"] == "mean_acc" and r["status"] in ("ok", "all zero")
            and int(r["n"]) == 2}
    ok = crit_ok == set(R.DECISION_CRITERIA) and {"H1", "H2", "H3", "H4", "H5", "H5-ES", "H6", "H7", "H8", "H9"} <= h_ok
    ok &= all(int(r["n_states"]) == 3 for r in dec if r.get("status") == "ok")
    up = [r for r in dec if r["criterion"] == "upload_numbers" and r["rec_mean_regret"] == "1"]
    record("(xiv) analysis on T10-format CSVs (seeds 10-11, 3 configs)", ok,
           f"hypotheses with data: {sorted(h_ok)}; decision criteria with data: {sorted(crit_ok)} on 3 states; "
           f"fewest uploaded numbers: {[r['action'] for r in up]}")
    return time.time() - t0


# ------------------------------------------------------------------------------------------------ (xiii)
def test_xiii():
    import json
    t0 = time.time()
    base = SCR / "st_cells_L5_s00.csv"
    rows = list(csv.DictReader(base.open()))
    meta = json.loads((SCR / "st_cells_L5_s00.meta.json").read_text())

    def cmp_exit(tag, rows_, meta_, *extra):
        R.write_csv(SCR / f"{tag}.csv", rows_, list(rows_[0].keys()))
        (SCR / f"{tag}.meta.json").write_text(json.dumps(meta_))
        p = subprocess.run([PY, str(HERE / "T10_review.py"), "compare", "--kind", "cells", "--csv",
                            str(SCR / f"{tag}.csv"), "--tag", f"cmp_{tag}", "--outdir", str(SCR), "--force", *extra],
                           capture_output=True, text=True)
        return p.returncode, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
    rc_ok, _ = cmp_exit("st_guard_same", rows, meta, "--min-runs", "10")
    drift = [dict(r, dataset=r["dataset"].replace("wine", "wine_v2")) for r in rows]      # label drift on one config
    rc_drift, msg_drift = cmp_exit("st_guard_drift", drift, meta)
    zero = [dict(r, seed=str(int(r["seed"]) + 100)) for r in rows]                        # nothing overlaps
    rc_zero, msg_zero = cmp_exit("st_guard_zero", zero, meta)
    rc_min, msg_min = cmp_exit("st_guard_min", rows, meta, "--min-runs", "100000")
    ok = rc_ok == 0 and rc_drift == 1 and rc_zero == 1 and rc_min == 1
    # argcheck: same command -> 0; a different command -> 1
    cmd = meta["argv"][1:]
    p1 = subprocess.run([PY, str(HERE / "T10_review.py"), "argcheck", "--meta", str(SCR / "st_cells_L5_s00.meta.json"),
                         "--", *cmd], capture_output=True, text=True)
    cmd2 = [c if c != "all16" else "all12" for c in cmd]
    p2 = subprocess.run([PY, str(HERE / "T10_review.py"), "argcheck", "--meta", str(SCR / "st_cells_L5_s00.meta.json"),
                         "--", *cmd2], capture_output=True, text=True)
    ok &= p1.returncode == 0 and p2.returncode == 1
    record("(xiii) compare / argcheck guards", ok,
           f"same rows pass; label drift -> exit {rc_drift} ({msg_drift.split(';')[-1].strip()}); zero overlap -> "
           f"exit {rc_zero}; --min-runs too high -> exit {rc_min}; argcheck same -> {p1.returncode}, "
           f"--cells all12 instead of all16 -> {p2.returncode}")
    return time.time() - t0


if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings = {}
    for name, fn in [("i", test_i), ("vi", test_vi), ("viii", test_viii), ("ix", test_ix), ("x", test_x),
                     ("xii", test_xii), ("ii", test_ii), ("xiii", test_xiii), ("xiv", test_xiv), ("xi", test_xi), ("iii", test_iii),
                     ("iv", test_iv), ("v", test_v), ("vii", test_vii)]:
        t1 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(f"({name}) raised", False, f"{type(e).__name__}: {e}")
        timings[name] = time.time() - t1
    print("\nSUMMARY")
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed; wall {time.time() - t0:.0f}s; per test (s): "
          + ", ".join(f"{k}={v:.0f}" for k, v in timings.items()))
    sys.exit(1 if n_fail else 0)
