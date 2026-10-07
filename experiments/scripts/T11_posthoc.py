"""T11 (post hoc, added 2026-10-07 after the Fable 5.1 pre-submission review; NOT part of the pre-registration).

Two re-runs of the decision analysis of T10 (the analysis specified in the §3 addendum), on the held-out seeds 10-29:

  U  common stopping rule: every mass-weighted action stops once its global prototypes move by less than
     T3.TOL = 1e-5 between two rounds -- the rule that GF-PFedFCM and SC-FFCM already had in T10. alpha_s = 0.75.
  C  as U, with the server step alpha_s of each mass-weighted action calibrated per configuration on seed index 100
     by the label-free rule used for SC-FFCM in T10 (T10_review.select_by_rule: finite and J <= 5 Jc; argmin
     |J - Jc| / Jc; ties -> first in grid order), on ALPHA_GRID, under the same stopping rule.

SC-FFCM keeps its T10 runs (calibrated steps, 1e-5 stop). T10_review.py (frozen; sha256 in results/t10/ANALYSIS_FROZEN)
and T3_design_space.py are imported and never modified; the analysis calls T10_review.decision_analysis unchanged.
run_cell_fair is T3.run_cell with two changes only: the stopping test applies to every cell (stop_all=True), and it
returns the number of rounds executed. With stop_all=False it reproduces T3.run_cell bit for bit (selftest).

Subcommands (outputs in results/t11_posthoc/, never overwritten without --force):
  selftest   run_cell_fair(stop_all=False) == T3.run_cell bit for bit, and == the stored T10 rows.
  calibrate  --L 1|5 : alpha_s grid on seed 100 for the mass-weighted actions with that L -> alpha_cal_L<L>.csv
  runs       --variant U|C --L 1|5 : seeds 10-29 -> runs_<variant>_L<L>.csv
  analyze    decision analysis for T10 as run, U and C (pooled seeds 10-29 and per block) -> decision_*.csv, summary.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for p in (str(HERE), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import T10_review as R  # noqa: E402  (frozen; imports T3 lazily, after the environment is set)

OUT = EXP / "results" / "t11_posthoc"
T10_DIR = EXP / "results" / "t10"
ROUNDS = 50
ALPHA_DEFAULT = 0.75
ALPHA_GRID = (0.25, 0.5, 0.75, 1.0)
SEEDS = list(range(10, 30))
BLOCKS = {"s10-29": SEEDS, "s10-19": list(range(10, 20)), "s20-29": list(range(20, 30))}
# the mass-weighted decision actions of T10 (name, T10 arm, cell, L)
MW_ACTIONS = [("F-FCM (post_none_off, L=5)", "L5:post_none_off", ("post", "none", "off"), 5),
              ("GF-PFedFCM (L=5)", "L5:post_footprint_on", ("post", "footprint", "on"), 5),
              ("ungated PRE (L=1)", "L1:pre_none_off", ("pre", "none", "off"), 1),
              ("pre_mass_off (L=1)", "L1:pre_mass_off", ("pre", "mass", "off"), 1),
              ("pre_mass_off (L=5)", "L5:pre_mass_off", ("pre", "mass", "off"), 5)]
assert [a[0] for a in MW_ACTIONS] == [n for n, _ in R.DECISION_ACTIONS[:5]]
assert [a[1] for a in MW_ACTIONS] == [arm for _, arm in R.DECISION_ACTIONS[:5]]
KEEP = ["mean_acc", "worst_acc", "gacc", "fcm_obj", "upload_numbers", "n_rounds", "min_dist", "xb", "hnorm"]


# ---------------------------------------------------------------------------------------------------------------
def run_cell_fair(T, clients, init, mass_point, gate, personalize, L, rounds=ROUNDS, slr=ALPHA_DEFAULT, stop_all=True):
    """T3.run_cell (record_gates=False) with the stopping test applied to every cell when stop_all is True.
    Returns (V, eval_centers, n_rounds). Every arithmetic step is T3's, in T3's order."""
    V = np.asarray(init, dtype=float).copy()
    n_clients = len(clients)
    n_clusters = V.shape[0]
    N = np.array([len(c.x) for c in clients], float)
    beta = N / N.sum()
    deltas = np.zeros((n_clients, n_clusters, V.shape[1]))
    gate_fn = T.GATE_FNS[gate]
    pers = personalize == "on"
    M = T.M
    n_done = 0
    for _ in range(rounds):
        previous = V.copy()
        num = np.zeros_like(V)
        den = np.zeros(n_clusters)
        for p, c in enumerate(clients):
            ubar = T.predict_membership(c.x, V, m=M)
            start = V + deltas[p] if pers else V
            loc = T.run_local_fcm_steps(c.x, start, steps=L, m=M)
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
        Tm = V.copy()
        ok = den > T.EPS
        Tm[ok] = num[ok] / den[ok, None]
        V = (1.0 - slr) * V + slr * Tm
        n_done += 1
        if (pers or stop_all) and not T.NO_EARLYSTOP and np.linalg.norm(V - previous) < T.TOL:
            break
    eval_centers = [V + deltas[p] for p in range(n_clients)] if pers else [V] * n_clients
    return V, eval_centers, n_done


def env_for(L: int) -> dict:
    return R.protocol_env(L, ROUNDS, seed_start=0)


def _init(env):
    R._init_worker(env)


def task_selftest(task):
    label, seed, cell, L = task
    T, clients, k, X, init = R.setup(label, seed)
    mp, g, pe = cell
    V0, ev0, n0 = R.run_cell_counted(T, clients, init, mp, g, pe)
    V1, ev1, n1 = run_cell_fair(T, clients, init, mp, g, pe, L, stop_all=False)
    same = bool(np.array_equal(V0, V1) and n0 == n1 and all(np.array_equal(a, b) for a, b in zip(ev0, ev1)))
    met = R.eval_metrics(T, clients, V1, ev1)
    return [{"dataset": label, "seed": seed, "cell": R.cell_name(cell), "L": L, "n_rounds": n1,
             "bit_identical_to_T3": int(same), "mean_acc": met["mean_acc"], "fcm_obj": met["fcm_obj"]}]


def task_cal(task):
    name, cell, L, label, alpha = task
    T, clients, k, X, init = R.setup(label, R.CAL_SEED, cache=True)
    t0 = time.time()
    with np.errstate(all="ignore"):
        V, ev, nr = run_cell_fair(T, clients, init, *cell, L, slr=alpha, stop_all=True)
        finite = bool(np.all(np.isfinite(V)))
        J = R.pooled_obj(X, V, T.M) if finite else math.inf
    return [{"action": name, "L": L, "dataset": label, "alpha_s": alpha, "finite": int(finite), "J": J,
             "n_rounds": nr, "runtime_s": round(time.time() - t0, 3)}]


def task_run(task):
    variant, name, cell, L, label, seed, alpha = task
    T, clients, k, X, init = R.setup(label, seed)
    t0 = time.time()
    V, ev, nr = run_cell_fair(T, clients, init, *cell, L, slr=alpha, stop_all=True)
    row = {"variant": variant, "action": name, "dataset": label, "seed": seed, "cell": R.cell_name(cell), "L": L,
           "alpha_s": alpha, "rounds": ROUNDS, "n_rounds": nr, "n_part": len(clients), "n_clusters": k,
           "dim": int(X.shape[1]), "m": T.M,
           "upload_numbers": R.upload_numbers("cells", nr, len(clients), k, int(X.shape[1]))}
    row.update(R.eval_metrics(T, clients, V, ev))
    row["runtime_s"] = round(time.time() - t0, 3)
    return [row]


def pool(fn, tasks, workers, env, label):
    os.environ.update(env)
    rows, t0, n = [], time.time(), len(tasks)
    tasks = sorted(tasks, key=lambda t: -R.COST.get(next(x for x in t if isinstance(x, str) and x in R.CONFIGS), 1))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init, initargs=(env,)) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 20 == 0 or i == n:
                print(f"[{label} {i}/{n}] {time.time() - t0:.0f}s", flush=True)
    return rows, time.time() - t0


# ---------------------------------------------------------------------------------------------------------------
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], force: bool):
    if path.exists() and not force:
        raise SystemExit(f"{path} exists (use --force)")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    meta = {"argv": sys.argv, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(),
            "python": platform.python_version(), "numpy": np.__version__, "rows": len(rows),
            "sha256": {"T11_posthoc.py": sha(HERE / "T11_posthoc.py"), "T10_review.py": sha(HERE / "T10_review.py"),
                       "T3_design_space.py": sha(HERE / "T3_design_space.py")}}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", path, len(rows), "rows")


def frozen_ok():
    want = (T10_DIR / "ANALYSIS_FROZEN").read_text().split()[0]
    if sha(HERE / "T10_review.py") != want:
        raise SystemExit("T10_review.py differs from the frozen hash")


def cmd_selftest(args):
    frozen_ok()
    stored = {}
    for tag in ("cells16_L5_s10-19", "cells16_L1_s10-19"):
        for r in csv.DictReader((T10_DIR / f"{tag}.csv").open()):
            stored[(r["dataset"], int(r["seed"]), r["cell"], int(r["L"]))] = r
    rows = []
    for L in (5, 1):
        tasks = [(lab, 10, cell, L) for lab in ("cluster_skew_overlap", "wine", "digits_pca16")
                 for _, _, cell, LL in MW_ACTIONS if LL == L]
        out, _ = pool(task_selftest, tasks, args.workers, env_for(L), f"selftest L{L}")
        rows += out
    bad = 0
    for r in rows:
        s = stored[(r["dataset"], r["seed"], r["cell"], r["L"])]
        r["equals_stored_T10"] = int(repr(float(s["mean_acc"])) == repr(r["mean_acc"])
                                     and repr(float(s["fcm_obj"])) == repr(r["fcm_obj"])
                                     and int(s["n_rounds"]) == r["n_rounds"])
        bad += (not r["bit_identical_to_T3"]) + (not r["equals_stored_T10"])
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "selftest.csv", rows, True)
    print("SELFTEST", "PASS" if bad == 0 else f"FAIL ({bad})")
    if bad:
        raise SystemExit(1)


def cmd_calibrate(args):
    frozen_ok()
    OUT.mkdir(exist_ok=True)
    L = args.L
    acts = [a for a in MW_ACTIONS if a[3] == L]
    labels = args.datasets.split(",") if args.datasets else R.CONFIGS
    ref, _ = pool(R.task_cal_ref, [(lab, R.CAL_SEED) for lab in labels], args.workers, env_for(L), f"Jc L{L}")
    Jc = {r["dataset"]: r["Jc"] for r in ref}
    pts, _ = pool(task_cal, [(n, cell, L, lab, a) for n, _, cell, _ in acts for lab in labels for a in ALPHA_GRID],
                  args.workers, env_for(L), f"calibrate L{L}")
    rows = []
    for n, _, cell, _ in acts:
        for lab in labels:
            points = {(p["alpha_s"],): {"finite": bool(p["finite"]), "J": p["J"]}
                      for p in pts if p["action"] == n and p["dataset"] == lab}
            best, n_stable, detail = R.select_by_rule([(a,) for a in ALPHA_GRID], points, Jc[lab])
            for (params, pt, ok, gap) in detail:
                rows.append({"action": n, "L": L, "dataset": lab, "alpha_s": params[0], "Jc": Jc[lab], "J": pt["J"],
                             "stable": int(ok), "gap": gap, "chosen": int(best is not None and params == best[0]),
                             "n_rounds": next(p["n_rounds"] for p in pts if p["action"] == n and p["dataset"] == lab
                                              and p["alpha_s"] == params[0])})
    write_csv(OUT / f"alpha_cal_L{L}.csv", rows, args.force)


def chosen_alpha() -> dict:
    out = {}
    for L in (1, 5):
        for r in csv.DictReader((OUT / f"alpha_cal_L{L}.csv").open()):
            if r["chosen"] == "1":
                out[(r["action"], r["dataset"])] = float(r["alpha_s"])
    return out


def cmd_runs(args):
    frozen_ok()
    OUT.mkdir(exist_ok=True)
    L, v = args.L, args.variant
    labels = args.datasets.split(",") if args.datasets else R.CONFIGS
    seeds = SEEDS[:args.max_seeds] if args.max_seeds else SEEDS
    alpha = chosen_alpha() if v == "C" else {}
    tasks = []
    for n, _, cell, LL in MW_ACTIONS:
        if LL != L:
            continue
        for lab in labels:
            a = alpha.get((n, lab), ALPHA_DEFAULT) if v == "C" else ALPHA_DEFAULT
            if v == "C" and a == ALPHA_DEFAULT:
                continue                     # identical to variant U (or, for GF-PFedFCM, to its T10 runs)
            if v == "U" and cell[2] == "on":
                continue                     # GF-PFedFCM already stopped at 1e-5 in T10 (alpha 0.75): reused as run
            tasks += [(v, n, cell, L, lab, s, a) for s in seeds]
    tag = args.tag or f"runs_{v}_L{L}"
    if not tasks:
        write_csv(OUT / f"{tag}.csv", [], args.force)
        return
    rows, el = pool(task_run, tasks, args.workers, env_for(L), f"{v} L{L}")
    print(f"elapsed {el:.0f}s")
    write_csv(OUT / f"{tag}.csv", rows, args.force)


def load_table(variant: str) -> dict:
    table, missing = R.load_arms_t10(T10_DIR)
    if missing:
        raise SystemExit(f"missing T10 CSVs: {missing}")
    if variant == "T10":
        return table
    arm_of = {n: arm for n, arm, _, _ in MW_ACTIONS}
    srcs = [OUT / f"runs_U_L{L}.csv" for L in (1, 5)]
    if variant == "C":
        srcs += [OUT / f"runs_C_L{L}.csv" for L in (1, 5)]   # later files override U rows
    for path in srcs:
        for r in csv.DictReader(path.open()):
            table[(arm_of[r["action"]], r["dataset"], int(r["seed"]))] = r
    return table


def cmd_analyze(args):
    frozen_ok()
    lines = ["# T11 post hoc decision analyses (not pre-registered); regrets as in T10_review.decision_analysis",
             f"# written {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"]
    all_rows, all_vals = [], []
    for variant in ("T10", "U", "C"):
        table = load_table(variant)
        for blk, seeds in BLOCKS.items():
            dec, vals = R.decision_analysis(table, seeds)
            for r in dec:
                r.update(variant=variant, block=blk)
            for r in vals:
                r.update(variant=variant, block=blk)
            all_rows += dec
            all_vals += vals
            if blk == "s10-29":
                lines.append(f"\n## {variant}, seeds 10-29 (mean regret | max regret; * recommended)")
                for r in dec:
                    if r.get("status") == "ok":
                        lines.append(f"{r['criterion']:<15} {r['action']:<28} {r['mean_regret']:.4f}"
                                     f"{'*' if r['rec_mean_regret'] else ' '} | {r['max_regret']:.4f}"
                                     f"{'*' if r['rec_max_regret'] else ' '}  ({r['argmax_state']})")
            rec = {}
            for r in dec:
                if r.get("status") == "ok":
                    for col in ("rec_mean_regret", "rec_max_regret"):
                        if r[col]:
                            rec.setdefault((r["criterion"], col), []).append(r["action"])
            lines.append(f"# {variant} {blk} recommended: " + "; ".join(f"{c}/{col[4:8]}: {', '.join(v)}"
                                                                       for (c, col), v in sorted(rec.items())))
    # how far the common stopping rule moves the four non-personalized actions (variant U vs T10 as run)
    t10, u = load_table("T10"), load_table("U")
    diffs = {}
    for n, arm, cell, L in MW_ACTIONS:
        if cell[2] == "on":
            continue
        for (a, ds, s), r in u.items():
            if a != arm or s not in SEEDS:
                continue
            r0 = t10[(a, ds, s)]
            d = diffs.setdefault(n, {"max_abs_dacc": 0.0, "max_rel_dobj": 0.0, "rounds": []})
            d["max_abs_dacc"] = max(d["max_abs_dacc"], abs(float(r["mean_acc"]) - float(r0["mean_acc"])))
            d["max_rel_dobj"] = max(d["max_rel_dobj"], abs(float(r["fcm_obj"]) - float(r0["fcm_obj"])) / abs(float(r0["fcm_obj"])))
            d["rounds"].append(int(r["n_rounds"]))
    lines.append("\n## common stopping rule vs T10 as run (non-personalized mass-weighted actions, seeds 10-29)")
    for n, d in diffs.items():
        rr = np.array(d["rounds"])
        lines.append(f"{n:<28} max|dACC| {d['max_abs_dacc']:.2e}  max rel dOBJ {d['max_rel_dobj']:.2e}  rounds: "
                     f"mean {rr.mean():.1f}, min {rr.min()}, max {rr.max()}, share stopped early {np.mean(rr < ROUNDS):.3f}")
    write_csv(OUT / "decision_all.csv", all_rows, args.force)
    write_csv(OUT / "decision_values_all.csv", all_vals, args.force)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "calibrate", "runs", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=12)
        s.add_argument("--force", action="store_true")
        s.add_argument("--datasets", default="")
        s.add_argument("--tag", default="")
        if name in ("calibrate", "runs"):
            s.add_argument("--L", type=int, choices=(1, 5), required=True)
        if name == "runs":
            s.add_argument("--variant", choices=("U", "C"), required=True)
            s.add_argument("--max-seeds", type=int, default=0)
    args = ap.parse_args()
    {"selftest": cmd_selftest, "calibrate": cmd_calibrate, "runs": cmd_runs, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
