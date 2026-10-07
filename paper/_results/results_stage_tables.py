#!/usr/bin/env python3
"""results_stage_tables.py -- tables and numbers of the Results section that make_tables_fodm.py does not produce.

Writes
  paper2_fodm/_results/tables/tab_L1.tex             (label tab:L1; main text, Section 7.3, one local step, fresh
                                                     block and the count rows of all three blocks)
  paper2_fodm/_results/tables/tab_substitution.tex   (label tab:substitution; Appendix A)
  paper2_fodm/_results/tables/tab_gfgain.tex         (label tab:gfgain; Appendix A)
  paper2_fodm/_results/tables/tab_fuzziness.tex      (label tab:fuzziness; Appendix A)
  paper2_fodm/_results/tables/tabS_main.tex          (label tab:main; Online Resource 1, moved out of the paper)
  paper2_fodm/_results/tables/tabS_footprintmass.tex (label tab:footprintmass; Online Resource 1, moved out)
  paper2_fodm/_results/tables/tabS_L1_expl.tex       (label tab:L1-expl; Online Resource 1: the exploratory panel of
                                                     the one-step table)
  paper2_fodm/_results/RESULTS_STAGE_EVIDENCE.md     (every number of src/07 and src/11 that is not in
                                                     EVIDENCE_T10*.md, _prep/FACTS_R2.md or a generated table)
  paper2_fodm/_results/results_stage_tests.csv       (per-configuration test rows of every comparison below)
The src chunks pull the tables in with a line  %%INSERT _results/tables/<name>.tex  (flatten.py); ESM_1_src.tex
pulls the tabS_* files in the same way (flatten_esm.py). Tables for Online Resource 1 cannot \\ref the paper: their
references into the paper are typed, resolved from main.aux when this script runs, and the paper's references into
Online Resource 1 are resolved from ESM_1.aux. Order: build main.tex and ESM_1.tex, run this script, flatten and
build both again (a reference that is not yet in an .aux file prints '??' and is listed at the end of the run).

Round budgets (Section 7.7): experiments/results/t3_design_space/per_seed_metrics__R{5,10,20}_L{1,5}.csv, and for
R=50 the T10 re-runs of the L=1 and L=5 cells on seeds 0-9 (the same values as the stored runs).

Data: the T10 re-runs in experiments/results/t10/ (cells16_*, gf_L1_noES_*, cfcm_*, FRESH_*; the re-runs of seeds
0-19 reproduce the stored outputs of the earlier analyses bit for bit, IMPLEMENTATION_NOTES.md) and, for the fuzzifier
reruns, experiments/results/t3_design_space/per_seed_metrics__T9_m{13,15,20}.csv.
Statistics: the frozen ones, imported from experiments/scripts/T10_review.py (sha256 checked against
results/t10/ANALYSIS_FROZEN): exact two-sided paired Wilcoxon signed-rank test (|d| <= 1e-12 dropped, mid-ranks of
|d| rounded to 12 decimals; n' = effective n), level 0.05, Holm per comparison over its configuration family
(14 configurations; 8 for the fuzzifier reruns), direction = sign of the paired mean difference. The helper
family_tests() re-implements run_tests() of the frozen code for an arbitrary family and is checked against the frozen
per-configuration rows (analysis_final_tests.csv) before any table is written.

Run:  python3 paper2_fodm/_results/results_stage_tables.py      (exits non-zero if a consistency check fails)
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("FFCM_MSTAR", "1")
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
PROJECT = PAPER.parent
T10 = PROJECT / "experiments" / "results" / "t10"
T3D = PROJECT / "experiments" / "results" / "t3_design_space"
SCRIPT = PROJECT / "experiments" / "scripts" / "T10_review.py"
OUT = HERE / "tables"
MD = HERE / "RESULTS_STAGE_EVIDENCE.md"
TESTS_CSV = HERE / "results_stage_tests.csv"

frozen = (T10 / "ANALYSIS_FROZEN").read_text().split()[0]
if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != frozen:
    sys.exit("T10_review.py differs from ANALYSIS_FROZEN; refusing to run")
_spec = importlib.util.spec_from_file_location("T10_review", SCRIPT)
R = importlib.util.module_from_spec(_spec)
sys.modules["T10_review"] = R
_spec.loader.exec_module(R)
CONFIGS, BLOCKS, ALPHA, _num = R.CONFIGS, R.BLOCKS, R.ALPHA, R._num

CHECKS: list[str] = []


def check(cond, msg):
    if not cond:
        sys.exit(f"CONSISTENCY CHECK FAILED: {msg}")
    CHECKS.append(msg)


# ------------------------------------------------------------------------------------------------ data
TAB, missing = R.load_arms_t10(T10)
check(not missing, "all T10 source CSVs of the frozen analysis are present")
for tag in ("cfcm_s00-19", "FRESH_cfcm_s20-29_descriptive"):
    for r in csv.DictReader((T10 / f"{tag}.csv").open()):
        TAB[("CFCM", r["dataset"], int(r["seed"]))] = r
MS = {"1.3": "T9_m13", "1.5": "T9_m15", "2.0": "T9_m20"}
for mval, tag in MS.items():
    for r in csv.DictReader((T3D / f"per_seed_metrics__{tag}.csv").open()):
        TAB[(f"M{mval}:{r['cell']}", r["dataset"], int(r["seed"]))] = r
FUZZ_CONFIGS = sorted({k[1] for k in TAB if k[0].startswith("M2.0:")}, key=CONFIGS.index)
check(len(FUZZ_CONFIGS) == 8, "the fuzzifier reruns cover eight configurations")

NAME = {"cluster_skew_hard": "Cluster skew", "cluster_skew_overlap": "Cluster skew (overlap)",
        "dirichlet_0.03": "Dirichlet $0.03$", "dirichlet_0.1": "Dirichlet $0.1$", "overlap_noise": "Overlap/noise",
        "quantity_skew_extreme": "Quantity skew", "wine": "Wine", "satimage": "Satimage", "pendigits": "Pendigits",
        "digits_pca16": "Digits PCA-16", "digits_pca32": "Digits PCA-32", "letter": "Letter",
        "mnist784_pca32 (20c)": "MNIST PCA-32 (20 clients)", "mnist784_pca32 (50c)": "MNIST PCA-32 (50 clients)"}
PLAIN = {c: NAME[c].replace("$", "") for c in CONFIGS}
SYNTH = CONFIGS[:6]
_mst = {}
for r in csv.DictReader((T10 / "ef_audit_s00-04_mstar.csv").open()):
    if r["init"] == "paper":
        _mst[r["dataset"]] = float(r["paper_mstar"])
MSTAR = {c: _mst[c] for c in CONFIGS}
GROUPS = [(r"Synthetic scenarios ($m^*=2$)", [c for c in CONFIGS if c in SYNTH]),
          (r"Real data, non-degenerate at $m=2$ ($m^*=2$)", [c for c in CONFIGS if c not in SYNTH and MSTAR[c] == 2.0]),
          (r"Real data, degenerate at $m=2$ (rescued, $m^*<2$)", [c for c in CONFIGS if MSTAR[c] < 2.0])]
check(len(GROUPS[2][1]) == 6, "six rescued configurations")

# ------------------------------------------------------------------------------------------------ tests
HB = dict(R.HIGHER_BETTER)
HB["err_acc"] = True                      # -log(1-ACC) increases with ACC: higher is better


def pair(a, b, hid=""):
    return dict(id=hid, kind="pair", a=a, b=b)


def inter(pre_none, post_none, pre_g, post_g, hid=""):
    return dict(id=hid, kind="inter", terms=[(1, pre_none), (-1, post_none), (-1, pre_g), (1, post_g)])


def one_test(h, ds, seeds, metric):
    """The frozen test on one configuration (re-implementation of R.run_tests for one row, without the legacy and
    NaN-loss columns). d oriented: positive = a better (higher ACC, lower OBJ) / interaction positive."""
    sign = 1.0 if HB.get(metric, True) else -1.0
    raw = R.contrast(TAB, h, ds, seeds, metric)
    present = [x for x in raw if x is not None]
    if not present:
        return dict(dataset=ds, metric=metric, n=0, n_nan=0, n_eff=0, p_exact=1.0, status="no data")
    fin = [x for x in present if not math.isnan(x)]
    od = [sign * x for x in fin]
    ex = R.wilcoxon_exact(od)
    mean, lo, hi = R.t_interval(fin)
    omean = float(np.mean(od)) if od else float("nan")
    direction = "a" if omean > 0 else ("b" if omean < 0 else
                                       ("a" if ex["w_plus"] > ex["n_eff"] * (ex["n_eff"] + 1) / 4 else "0"))
    return dict(dataset=ds, metric=metric, n=len(present), n_nan=len(present) - len(fin), n_eff=ex["n_eff"],
                w_plus=ex["w_plus"], p_exact=ex["p"], mean_diff_raw=mean, ci95_lo=lo, ci95_hi=hi,
                mean_diff_oriented=omean, direction=direction, status="ok" if ex["n_eff"] > 0 else "all zero")


def family_tests(h, block, metric, family=None):
    fam = CONFIGS if family is None else family
    rows = [one_test(h, ds, BLOCKS[block], metric) for ds in fam]
    adj = R.holm([_num(r["p_exact"]) for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["hyp"] = h["id"]
        r["block"] = block
    return rows


def counts(rows, col="p_exact"):
    ok = [r for r in rows if r["status"] == "ok"]
    return (sum(1 for r in ok if r[col] < ALPHA and r["direction"] == "a"),
            sum(1 for r in ok if r[col] < ALPHA and r["direction"] == "b"))


def mrd(rows):
    ok = [r for r in rows if r["status"] == "ok" and r["p_exact"] < ALPHA and abs(r["mean_diff_raw"]) >= 0.005]
    return sum(1 for r in ok if r["direction"] == "a"), sum(1 for r in ok if r["direction"] == "b")


def uh(rows):
    u, h = counts(rows, "p_exact"), counts(rows, "p_holm")
    return f"{u[0]}/{u[1]} ({h[0]}/{h[1]})"


def row_of(rows, ds):
    return next(r for r in rows if r["dataset"] == ds)


ALL_ROWS: list[dict] = []


def T(h, block, metric="mean_acc", family=None):
    rows = family_tests(h, block, metric, family)
    ALL_ROWS.extend(rows)
    return rows


# ---- validation of the re-implementation against the frozen per-configuration rows
FROZEN = list(csv.DictReader((T10 / "analysis_final_tests.csv").open()))
_n_val = 0
for h in R.H_PRIMARY + R.H_SECONDARY:
    for block in BLOCKS:
        for metric in ("mean_acc", "fcm_obj"):
            fr = [r for r in FROZEN if r["hyp"] == h["id"] and r["block"] == block and r["metric"] == metric]
            if not fr or all(r["status"] == "no data" for r in fr):
                continue
            mine = {r["dataset"]: r for r in family_tests(h, block, metric, R.family_for(h))}
            for f in fr:
                m = mine[f["dataset"]]
                same = (f["status"] == m["status"] and abs(_num(f["p_exact"]) - m["p_exact"]) < 1e-12 and
                        abs(_num(f["p_holm"]) - m["p_holm"]) < 1e-12 and int(f["n_eff"]) == m["n_eff"])
                if f["status"] == "ok":
                    same = same and f["direction"] == m["direction"] and \
                        abs(_num(f["mean_diff_raw"]) - m["mean_diff_raw"]) < 1e-12
                check(same, f"re-implemented test equals the frozen row {h['id']} {block} {metric} {f['dataset']}")
                _n_val += 1
CHECKS.append(f"{_n_val} frozen per-configuration test rows reproduced by family_tests()")


# ------------------------------------------------------------------------------------------------ formatting
def d3(x, marks=""):
    x = _num(x)
    if not math.isfinite(x):
        return "--"
    s = f"{x:+.3f}"
    if s in ("+0.000", "-0.000"):
        s = "0.000"
    return f"${s}^{{{marks}}}$" if marks else f"${s}$"


def v3(x):
    x = _num(x)
    return "--" if not math.isfinite(x) else f"{x:.3f}"


def marks(r):
    m = ""
    if r["status"] == "ok" and r["p_exact"] < ALPHA:
        m += "*"
    if r["status"] == "ok" and r["p_holm"] < ALPHA:
        m += r"\dagger"
    return m


def cell(rows, ds, nseeds=10):
    r = row_of(rows, ds)
    if r["status"] == "no data":
        return "n.r."
    if r["status"] == "all zero":
        return r"$0.000$\,(0)"
    note = rf"\,({r['n_eff']})" if r["n_eff"] < nseeds else ""
    return d3(r["mean_diff_raw"], marks(r)) + note


def mean_of(arm, ds, block, metric="mean_acc"):
    v = [_num(TAB[(arm, ds, s)].get(metric)) for s in BLOCKS[block] if (arm, ds, s) in TAB]
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def config_block(ncols, cells_fn, configs=None):
    lines = []
    for title, members in GROUPS:
        members = [c for c in members if configs is None or c in configs]
        if not members:
            continue
        if lines:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for c in members:
            lines.append(" & ".join([NAME[c], f"{MSTAR[c]:.1f}"] + list(cells_fn(c))) + r" \\")
    return lines


def lab(n, text):
    return rf"\multicolumn{{{n}}}{{@{{}}l}}{{{text}}}"


# ---- cross-document references (Online Resource 1 is compiled on its own and cannot \ref the paper)
def aux_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", path.read_text(errors="ignore")):
        out.setdefault(m.group(1), m.group(2))
    return out


MAINLAB = aux_labels(PAPER / "main.aux")
ESMLAB = aux_labels(PAPER / "ESM_1.aux")
XREF_WARN: list[str] = []


def paper_ref(label, kind="Section"):
    """Typed reference from Online Resource 1 into the paper, e.g. 'Section~6.9 of the paper'."""
    if label in MAINLAB:
        return f"{kind}~{MAINLAB[label]} of the paper"
    XREF_WARN.append(f"main.aux has no label {label}")
    return f"{kind}~?? of the paper"


def esm_ref(label, kind="Table"):
    """Typed reference from the paper into Online Resource 1, e.g. 'Online Resource 1, Table~S29'."""
    if label in ESMLAB:
        return f"Online Resource 1, {kind}~{ESMLAB[label]}"
    XREF_WARN.append(f"ESM_1.aux has no label {label}")
    return f"Online Resource 1, {kind}~S??"


def table(label, caption, colspec, head, body, sep="3pt"):
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\tablefontsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\setlength{{\tabcolsep}}{{{sep}}}",
        r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}%", r"}\end{tabular}", r"\end{table}", ""])


MARKS_TXT = (r"$^{*}$: $p<0.05$ (exact test, unadjusted); $^{\dagger}$: Holm-adjusted $p<0.05$ over the "
             r"configurations of the comparison; in parentheses after an entry: the effective sample size $n'$ where "
             r"it is below the number of seeds")
STATS = r"Tests and marks: Section~\ref{sec:protocol-stats}"
STATS_ESM = f"Tests and marks: {paper_ref('sec:protocol-stats')}"
SL = r"$^{\S}$"
BLK = {"s00-09": r"$0$--$9$", "s10-19": r"$10$--$19$", "s20-29": r"$20$--$29$"}
OUT.mkdir(exist_ok=True)
md: list[str] = ["# RESULTS_STAGE_EVIDENCE: numbers of src/07_results.tex and src/11_appendixA_tables.tex",
                 "", "Generated by `paper2_fodm/_results/results_stage_tables.py` (do not edit by hand). Statistics: "
                 "the frozen exact test of `experiments/scripts/T10_review.py` (sha256 checked), Holm per comparison "
                 "over its configuration family. Counts `u (H)` = first-named better/worse, unadjusted (Holm). "
                 "Seeds 10-19 contrasts at L=1 are a SECOND LOOK. Comparisons on seeds 20-29 other than H1-H9 and "
                 "their declared secondary pairs are post hoc (not pre-registered). C-FCM on seeds 20-29 is "
                 "descriptive (no test).", ""]


def mdline(s=""):
    md.append(s)


def cfg_list(rows, side, col="p_exact"):
    out = []
    for r in rows:
        if r["status"] == "ok" and r[col] < ALPHA and r["direction"] == side:
            out.append(f"{PLAIN[r['dataset']]} {r['mean_diff_raw']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] "
                       f"(n'={r['n_eff']}; p={r['p_exact']:.3g}; Holm {r['p_holm']:.3g})")
    return "; ".join(out) or "none"


# ================================================================================================ (1) tab:L1
H4 = pair("L1:pre_none_off", "L1:post_none_off", "H4")
MG1 = pair("L1:pre_mass_off", "L1:pre_none_off", "MG-L1")
L1 = {}
for b in BLOCKS:
    L1[("MG", b, "mean_acc")] = T(MG1, b, "mean_acc")
    L1[("MG", b, "fcm_obj")] = T(MG1, b, "fcm_obj")
    L1[("H4", b, "mean_acc")] = T(H4, b, "mean_acc")
    L1[("H4", b, "fcm_obj")] = T(H4, b, "fcm_obj")
# H4 counts equal the frozen summary
SUMM = list(csv.DictReader((T10 / "analysis_final_summary.csv").open()))


def frozen_uh(block, hyp, metric):
    s = next(r for r in SUMM if r["block"] == block and r["hyp"] == hyp and r["metric"] == metric)
    return f"{s['better_unadj']}/{s['worse_unadj']} ({s['better_holm']}/{s['worse_holm']})"


for b in BLOCKS:
    for mt in ("mean_acc", "fcm_obj"):
        check(uh(L1[("H4", b, mt)]) == frozen_uh(b, "H4", mt), f"H4 {b} {mt} equals the frozen summary")

F = "s20-29"
EB = list(csv.DictReader((HERE / "extra_A_eb_values.csv").open()))


def eb(block, ds, arm, col):
    r = next(x for x in EB if x["block"] == block and x["dataset"] == ds and x["arm"] == arm)
    return _num(r[col])


for ds in CONFIGS:
    for arm in ("L1:pre_none_off", "L1:pre_mass_off", "CFCM"):
        for col in ("mean_acc", "hnorm"):
            check(abs(mean_of(arm, ds, F, col) - eb(F, ds, arm, col)) < 1e-12,
                  f"fresh {arm} {col} {ds} equals extra_A_eb_values.csv")


def mwv(arm, c, b):
    return f"{mean_of(arm, c, b):.3f}/{mean_of(arm, c, b, 'worst_acc'):.3f}"


PRE_CF = {b: T(pair("L1:pre_none_off", "CFCM", "PRE-CFCM"), b) for b in ("s00-09", "s10-19")}
L1A = ["L1:pre_none_off", "L1:pre_mass_off", "SC:L1", "SC:L5", "CFCM"]


def l1a_cells(c):
    return [mwv(a, c, E0) for a in L1A] + [cell(PRE_CF["s00-09"], c)]


def l1b_cells(c):
    return [mwv("L1:pre_none_off", c, F), mwv("L1:pre_mass_off", c, F), mwv("CFCM", c, F),
            v3(mean_of("L1:pre_none_off", c, F, "hnorm")), v3(mean_of("L1:pre_mass_off", c, F, "hnorm")),
            cell(L1[("MG", F, "mean_acc")], c), cell(L1[("MG", F, "fcm_obj")], c),
            cell(L1[("H4", F, "mean_acc")], c), cell(L1[("H4", F, "fcm_obj")], c)]


E0 = "s00-09"
# the exploratory values reproduce the previous version of tab:L1 (backup of src/07_results.tex)
_old = (PAPER / "_prep" / "bak_results_stage" / "07_results.tex").read_text()
_old = _old[_old.index(r"\label{tab:L1}"):]
_old = _old[:_old.index(r"\bottomrule")]
_nrep = 0
for c in CONFIGS:
    line = next(ln for ln in _old.splitlines() if ln.startswith(NAME[c] + " &"))
    cols = [x.strip() for x in line.split("&")]
    old_vals = [cols[2], cols[3], cols[4], cols[5], cols[7]]      # PRE, pre_mass_off, SC L=1, SC L=5, C-FCM
    for a, ov in zip(L1A, old_vals):
        check(mwv(a, c, E0) == ov, f"tab:L1 exploratory {a} {c} reproduces the previous table ({ov})")
        _nrep += 1
CHECKS.append(f"{_nrep} exploratory mean/worst entries of the previous tab:L1 reproduced")

# (a) the exploratory panel moves to Online Resource 1 (label tab:L1-expl); (b) the fresh block stays in the paper
body_a = config_block(8, l1a_cells)
body_a += [r"\midrule", lab(7, r"\emph{PRE vs C-FCM, significantly better/worse, unadjusted (Holm)}") + r" & \\",
           lab(7, r"Seeds $0$--$9$") + f" & {uh(PRE_CF['s00-09'])} \\\\",
           lab(7, r"Seeds $10$--$19$" + SL) + f" & {uh(PRE_CF['s10-19'])} \\\\"]
head_a = [r" & & & & \multicolumn{2}{c}{SC-FFCM} & & \\", r"\cmidrule(lr){5-6}",
          r"Configuration & $m^*$ & PRE & \texttt{pre\_mass\_off} & $L=1$ & $L=5$ & C-FCM & PRE$-$C-FCM \\"]
cap_a = (r"One local step ($L=1$), no personalization, $R=50$: mean/worst client ACC on the exploratory seeds "
         r"$0$--$9$ (the fresh block: Table~\ref{tab:L1-full}). PRE: the ungated PRE rule "
         r"\texttt{pre\_none\_off}; SC-FFCM with steps calibrated per configuration and $L$ ("
         + paper_ref("sec:protocol-baselines") + r"); C-FCM: centralized FCM at $m^*$ from the same prototypes, at "
         r"most $150$ iterations. PRE$-$C-FCM: difference in mean client ACC. Bottom: PRE against C-FCM, significantly "
         r"better/worse, unadjusted and, in parentheses, Holm-adjusted over the 14 configurations; " + SL + r": second "
         r"look at seeds $10$--$19$. " + MARKS_TXT + ". " + STATS_ESM + ".")
(OUT / "tabS_L1_expl.tex").write_text(table("tab:L1-expl", cap_a, "@{}lccccccc@{}", head_a, body_a),
                                      encoding="utf-8")
body_b = config_block(11, l1b_cells)
body_b += [r"\midrule",
           lab(7, r"\emph{Significantly better/worse, unadjusted (Holm)}") + r" & \multicolumn{2}{c}{"
           r"\texttt{pre\_mass\_off} vs PRE} & \multicolumn{2}{c}{PRE vs POST (H4)} \\"]
for b in BLOCKS:
    lbl = f"Seeds {BLK[b]}" + (SL if b == "s10-19" else "")
    body_b.append(lab(7, lbl) + " & " + " & ".join(uh(L1[(k, b, mt)]) for k in ("MG", "H4")
                                                     for mt in ("mean_acc", "fcm_obj")) + r" \\")
head_b = [r" & & \multicolumn{3}{c}{Mean/worst client ACC} & \multicolumn{2}{c}{$H/\ln c$} & \multicolumn{2}{c}{Mass gate} "
          r"& \multicolumn{2}{c}{PRE$-$POST} \\",
          r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
          r"Configuration & $m^*$ & PRE & \texttt{pre\_mass\_off} & C-FCM & PRE & \texttt{pre\_mass\_off} & ACC & OBJ "
          r"& ACC & OBJ \\"]
cap = (r"One local step ($L=1$), no personalization, $R=50$, on the fresh block (seeds $20$--$29$). PRE: the ungated "
       r"PRE rule \texttt{pre\_none\_off}; C-FCM: centralized FCM at $m^*$ from the same prototypes, at most $150$ "
       r"iterations, a descriptive reference on these seeds (not pre-registered, not tested). Normalized membership "
       r"entropy $H/\ln c$ next to the accuracies. Mass gate: \texttt{pre\_mass\_off} minus PRE in mean client ACC and "
       r"in OBJ \eqref{eq:protocol-obj}, where a positive OBJ difference is a higher, i.e.\ worse, objective (post "
       r"hoc); PRE$-$POST: \texttt{pre\_none\_off} minus \texttt{post\_none\_off}, the pre-registered H4. Count rows: "
       r"configurations (of 14) on which the first-named rule is significantly better/worse (ACC higher/lower, OBJ "
       r"lower/higher), unadjusted and, in parentheses, Holm-adjusted over the 14 configurations, in the three seed "
       r"blocks; " + SL + r": second look at seeds $10$--$19$; OBJ of the exploratory one-step cells comes from the "
       r"re-runs of Section~\ref{sec:protocol-metrics}. Exploratory values, SC-FFCM and the tests against C-FCM on "
       r"seeds $0$--$19$: " + esm_ref("tab:L1-expl") + ". " + MARKS_TXT + ". " + STATS + ".")
# 2026-10-07 (compression): the full fresh-block table moves to Online Resource 1 (tab:L1-full); the paper keeps the
# columns that its Section 7.3 discusses (tab:L1).
(OUT / "tab_L1_full.tex").write_text(table("tab:L1-full", cap, "@{}lcccccccccc@{}", head_b, body_b), encoding="utf-8")


def l1r_cells(c):
    return [mwv("L1:pre_none_off", c, F), mwv("L1:pre_mass_off", c, F),
            cell(L1[("MG", F, "mean_acc")], c), cell(L1[("MG", F, "fcm_obj")], c)]


body_r = config_block(6, l1r_cells)
body_r += [r"\midrule", lab(4, r"\emph{\texttt{pre\_mass\_off} vs PRE, significantly better/worse, unadjusted (Holm)}")
           + r" & & \\"]
for b in BLOCKS:
    lbl = f"Seeds {BLK[b]}" + (SL if b == "s10-19" else "")
    body_r.append(lab(4, lbl) + " & " + " & ".join(uh(L1[("MG", b, mt)]) for mt in ("mean_acc", "fcm_obj")) + r" \\")
head_r = [r" & & \multicolumn{2}{c}{Mean/worst client ACC} & \multicolumn{2}{c}{Mass gate} \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
          r"Configuration & $m^*$ & PRE & \texttt{pre\_mass\_off} & ACC & OBJ \\"]
cap_r = (r"One local step ($L=1$), no personalization, $R=50$, on the fresh block (seeds $20$--$29$): mean/worst client "
         r"ACC of the ungated PRE rule \texttt{pre\_none\_off} (PRE) and of \texttt{pre\_mass\_off}, and the effect of "
         r"the mass gate, \texttt{pre\_mass\_off} minus PRE, in mean client ACC and in OBJ \eqref{eq:protocol-obj} "
         r"(absolute differences; a positive OBJ difference is a higher, i.e.\ worse, objective; relative changes in "
         r"Figure~\ref{fig:tradeoff}). Bottom: counts of \texttt{pre\_mass\_off} against PRE in the three seed blocks, "
         r"unadjusted and, in parentheses, Holm-adjusted over the 14 configurations; " + SL + r": second look at seeds "
         r"$10$--$19$. C-FCM, $H/\ln c$ and PRE$-$POST on the fresh block: " + esm_ref("tab:L1-full") + r"; exploratory "
         r"seeds: " + esm_ref("tab:L1-expl") + ". " + MARKS_TXT + ". " + STATS + ".")
(OUT / "tab_L1.tex").write_text(table("tab:L1", cap_r, "@{}lccccc@{}", head_r, body_r), encoding="utf-8")

mdline("## 1. One local step (tab:L1)")
mdline()
for b in BLOCKS:
    mdline(f"- mass gate (pre_mass_off vs pre_none_off, L=1), {b}: ACC {uh(L1[('MG', b, 'mean_acc')])}; "
           f"OBJ {uh(L1[('MG', b, 'fcm_obj')])}")
    mdline(f"  - ACC better (unadj.): {cfg_list(L1[('MG', b, 'mean_acc')], 'a')}")
    mdline(f"  - ACC worse (unadj.): {cfg_list(L1[('MG', b, 'mean_acc')], 'b')}")
    mdline(f"  - OBJ better (lower; unadj.): {cfg_list(L1[('MG', b, 'fcm_obj')], 'a')}")
    ob = L1[('MG', b, 'fcm_obj')]
    mdline(f"  - OBJ mean difference range (raw, pre_mass_off - PRE): "
           f"{min(r['mean_diff_raw'] for r in ob if r['status'] == 'ok'):+.3f} to "
           f"{max(r['mean_diff_raw'] for r in ob if r['status'] == 'ok'):+.3f}; largest at "
           f"{PLAIN[max((r for r in ob if r['status'] == 'ok'), key=lambda r: r['mean_diff_raw'])['dataset']]}")
    mdline(f"- H4 (PRE vs POST, L=1), {b}: ACC {uh(L1[('H4', b, 'mean_acc')])}; OBJ {uh(L1[('H4', b, 'fcm_obj')])}; "
           f"ACC worse: {cfg_list(L1[('H4', b, 'mean_acc')], 'b')}")
# relative OBJ change of the mass gate (fresh), per configuration
rel = []
for c in CONFIGS:
    a_, b_ = mean_of("L1:pre_mass_off", c, F, "fcm_obj"), mean_of("L1:pre_none_off", c, F, "fcm_obj")
    rel.append((c, (a_ - b_) / b_))
mdline("- fresh block, relative OBJ change of the mass gate, (mean OBJ pre_mass_off - mean OBJ PRE)/mean OBJ PRE: "
       + "; ".join(f"{PLAIN[c]} {x:+.3f}" for c, x in rel))
mdline(f"- fresh block mean over configurations of H/ln c: PRE L=1 "
       f"{np.mean([mean_of('L1:pre_none_off', c, F, 'hnorm') for c in CONFIGS]):.3f}, pre_mass_off L=1 "
       f"{np.mean([mean_of('L1:pre_mass_off', c, F, 'hnorm') for c in CONFIGS]):.3f}, C-FCM "
       f"{np.mean([mean_of('CFCM', c, F, 'hnorm') for c in CONFIGS]):.3f}")
# PRE (L=1) and pre_mass_off (L=1) against C-FCM on seeds 0-19 (C-FCM on 20-29 is descriptive)
for arm, nm in (("L1:pre_none_off", "PRE"), ("L1:pre_mass_off", "pre_mass_off")):
    for b in ("s00-09", "s10-19"):
        rows = PRE_CF[b] if nm == "PRE" else T(pair(arm, "CFCM", f"{nm}-CFCM"), b)
        same3 = sum(1 for c in CONFIGS if f"{mean_of(arm, c, b):.3f}" == f"{mean_of('CFCM', c, b):.3f}")
        mdline(f"- {nm} (L=1) vs C-FCM, {b}: ACC {uh(rows)}; equal to three decimals on {same3} configurations; "
               f"better: {cfg_list(rows, 'a')}; worse: {cfg_list(rows, 'b')}")
# personalization at L=1 at equal budgets: GF (L=1, no early stop) vs post_footprint_off (L=1)
PL1 = pair("L1noES:post_footprint_on", "L1:post_footprint_off", "PERS-L1")
for b in BLOCKS:
    ra, ro = T(PL1, b, "mean_acc"), T(PL1, b, "fcm_obj")
    mdline(f"- personalization at L=1 (GF-PFedFCM L=1 no early stop vs post_footprint_off L=1), {b}: ACC {uh(ra)}; "
           f"OBJ {uh(ro)}")
# C-FCM iterations
for tag in ("cfcm_s00-19", "FRESH_cfcm_s20-29_descriptive"):
    rows = list(csv.DictReader((T10 / f"{tag}.csv").open()))
    capped = sum(1 for r in rows if int(r["n_iter"]) >= int(r["max_iter"]))
    mdline(f"- {tag}: {capped} of {len(rows)} C-FCM runs reached the {rows[0]['max_iter']}-iteration cap")
mdline()

# ================================================================================================ (2) tab:substitution
DOFF = pair("L5:pre_none_off", "L5:post_none_off", "DOFF")
DON = pair("L5:pre_footprint_off", "L5:post_footprint_off", "DON")
DINT = inter("L5:pre_none_off", "L5:post_none_off", "L5:pre_footprint_off", "L5:post_footprint_off", "D")
SUB = {(k, b): T(h, b) for b in BLOCKS for k, h in (("off", DOFF), ("on", DON), ("D", DINT))}
for b in BLOCKS:
    check(uh(SUB[("D", b)]) == frozen_uh(b, "H3", "mean_acc"), f"interaction D {b} equals the frozen H3 counts")
GAP = {c: mean_of("CFCM", c, "s00-09") - mean_of("L5:post_footprint_off", c, "s00-09") for c in CONFIGS}


def sub_cells(c):
    out = []
    for b in BLOCKS:
        out += [cell(SUB[("off", b)], c), cell(SUB[("on", b)], c), cell(SUB[("D", b)], c)]
    return out + [d3(GAP[c])]


# error scale -log(1 - ACC) for the interaction, on the configurations where it is finite for all four cells
def err_rows(block):
    arms = ["L5:pre_none_off", "L5:post_none_off", "L5:pre_footprint_off", "L5:post_footprint_off"]
    fam = []
    for c in CONFIGS:
        vals = [_num(TAB[(a, c, s)]["mean_acc"]) for a in arms for s in BLOCKS[block]]
        if all(v < 1.0 for v in vals):
            fam.append(c)
            for a in arms:
                for s in BLOCKS[block]:
                    TAB[(a, c, s)]["err_acc"] = repr(-math.log(1.0 - _num(TAB[(a, c, s)]["mean_acc"])))
    return fam, T(dict(DINT, id="D-err"), block, "err_acc", fam)


ERR = {b: err_rows(b) for b in BLOCKS}
# ceiling check (seeds 0-9): configurations whose gated POST cell is more than 0.03 below C-FCM
ceil_cfg = [c for c in CONFIGS if GAP[c] > 0.03]

body = config_block(12, sub_cells)
body.append(r"\midrule")
body.append(lab(2, r"Significantly positive/negative") + " & " +
            " & ".join(f"{counts(SUB[(k, b)])[0]}/{counts(SUB[(k, b)])[1]}" for b in BLOCKS
                       for k in ("off", "on", "D")) + r" & \\")
body.append(lab(2, r"Holm-adjusted (14 configurations)") + " & " +
            " & ".join(f"{counts(SUB[(k, b)], 'p_holm')[0]}/{counts(SUB[(k, b)], 'p_holm')[1]}" for b in BLOCKS
                       for k in ("off", "on", "D")) + r" & \\")
body.append(lab(2, r"Significant and $|\Delta|\ge0.005$") + " & " +
            " & ".join(f"{mrd(SUB[(k, b)])[0]}/{mrd(SUB[(k, b)])[1]}" for b in BLOCKS
                       for k in ("off", "on", "D")) + r" & \\")
head = [r" & & \multicolumn{3}{c}{Seeds $0$--$9$ (exploratory)} & \multicolumn{3}{c}{Seeds $10$--$19$} & "
        r"\multicolumn{3}{c}{Seeds $20$--$29$ (fresh)} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Configuration & $m^*$ & $\Delta^{\mathrm{off}}$ & $\Delta^{\mathrm{on}}$ & Shrink $D$ & "
        r"$\Delta^{\mathrm{off}}$ & $\Delta^{\mathrm{on}}$ & Shrink $D$ & $\Delta^{\mathrm{off}}$ & "
        r"$\Delta^{\mathrm{on}}$ & Shrink $D$ & Gap \\"]


def errtxt(b):
    fam, rows = ERR[b]
    u, h = counts(rows), counts(rows, "p_holm")
    return f"{u[0]}/{u[1]} ({h[0]}/{h[1]}) on seeds {BLK[b]} ({len(fam)} configurations)"


cap = (r"The substitution interaction with the footprint gate, decomposed (no personalization, $L=5$): PRE$-$POST "
       r"differences in mean client ACC without a gate ($\Delta^{\mathrm{off}}$, \texttt{pre\_none\_off} minus "
       r"\texttt{post\_none\_off}) and with the footprint gate ($\Delta^{\mathrm{on}}$, \texttt{pre\_footprint\_off} "
       r"minus \texttt{post\_footprint\_off}), and their difference, the interaction $D$ of "
       r"\eqref{eq:protocol-did} (Shrink: positive where the gate takes over part of the PRE$-$POST difference, read "
       r"so only where $\Delta^{\mathrm{off}}>0$). $D$ on seeds $20$--$29$ is the pre-registered H3 "
       r"(Table~\ref{tab:confirm}), on seeds $10$--$19$ a comparison of the first held-out look; the decompositions "
       r"$\Delta^{\mathrm{off}}$, $\Delta^{\mathrm{on}}$ are descriptive. $95\%$ $t$-intervals of $D$ on seeds "
       r"$20$--$29$: " + esm_ref("tab:fresh-detail-a") + r". The other gates: "
       r"Table~\ref{tab:gates-interaction}. Gap: C-FCM minus \texttt{post\_footprint\_off}, seeds $0$--$9$ "
       r"(descriptive). On the error scale $-\log(1-\mathrm{ACC})$, defined where no run of the four cells reaches "
       r"$\mathrm{ACC}=1$, the interaction is significantly positive/negative, unadjusted (Holm), "
       + "; ".join(errtxt(b) for b in BLOCKS) + ". " + MARKS_TXT + ". " + STATS + ".")
(OUT / "tab_substitution.tex").write_text(table("tab:substitution", cap, "@{}lccccccccccc@{}", head, body),
                                           encoding="utf-8")
mdline("## 2. Substitution interaction, footprint gate (tab:substitution)")
mdline()
for b in BLOCKS:
    for k, nm in (("off", "Delta_off"), ("on", "Delta_on"), ("D", "interaction D")):
        mdline(f"- {nm}, {b}: {uh(SUB[(k, b)])}; minimum reported difference (sig. and |d|>=0.005) "
               f"{mrd(SUB[(k, b)])[0]}/{mrd(SUB[(k, b)])[1]}; positive: {cfg_list(SUB[(k, b)], 'a')}; "
               f"negative: {cfg_list(SUB[(k, b)], 'b')}")
    fam, rows = ERR[b]
    mdline(f"- error scale, {b}: defined on {len(fam)} configurations ({', '.join(PLAIN[c] for c in fam)}); "
           f"D {uh(rows)}; significant positive: {', '.join(PLAIN[r['dataset']] for r in rows if r['status'] == 'ok' and r['p_exact'] < ALPHA and r['direction'] == 'a')}; "
           f"negative: {', '.join(PLAIN[r['dataset']] for r in rows if r['status'] == 'ok' and r['p_exact'] < ALPHA and r['direction'] == 'b') or 'none'}")
    acc_sig = {r["dataset"]: r["direction"] for r in SUB[("D", b)] if r["status"] == "ok" and r["p_exact"] < ALPHA
               and r["dataset"] in fam}
    err_sig = {r["dataset"]: r["direction"] for r in rows if r["status"] == "ok" and r["p_exact"] < ALPHA}
    mdline(f"  - on these configurations the significant interactions on the two scales are "
           f"{'the same, with the same signs' if acc_sig == err_sig else 'NOT the same: ACC ' + str(acc_sig) + ' vs error ' + str(err_sig)}")
mdline(f"- ceiling check, seeds 0-9: configurations with Gap > 0.03: {', '.join(PLAIN[c] for c in ceil_cfg)} "
       f"({len(ceil_cfg)}); D significantly positive (unadj.) on "
       f"{', '.join(PLAIN[r['dataset']] for r in SUB[('D', 's00-09')] if r['dataset'] in ceil_cfg and r['status'] == 'ok' and r['p_exact'] < ALPHA and r['direction'] == 'a')}")
mdline("- Gap (C-FCM - post_footprint_off, seeds 0-9): " + "; ".join(f"{PLAIN[c]} {GAP[c]:+.3f}" for c in CONFIGS))
for b in BLOCKS:
    mdline(f"- {b}: Delta_off - Delta_on per configuration (mean over seeds): " + "; ".join(
        f"{PLAIN[c]} {row_of(SUB[('off', b)], c).get('mean_diff_raw', 0.0):+.3f} -> "
        f"{row_of(SUB[('on', b)], c).get('mean_diff_raw', 0.0):+.3f}" for c in CONFIGS))
mdline()

# ================================================================================================ (3) tab:main
CELLS12 = ["post_none_off", "post_none_on", "post_footprint_off", "post_footprint_on", "post_mass_off",
           "post_mass_on", "pre_none_off", "pre_none_on", "pre_footprint_off", "pre_footprint_on", "pre_mass_off",
           "pre_mass_on"]
E = "s00-09"
BEST = {}
for c in CONFIGS:
    means = [(mean_of(f"L5:{cl}", c, E), -i, cl) for i, cl in enumerate(CELLS12)]
    BEST[c] = max(means)[2]
OLD_BEST = {"cluster_skew_hard": "post_footprint_off", "cluster_skew_overlap": "pre_mass_off",
            "dirichlet_0.03": "post_mass_on", "dirichlet_0.1": "post_footprint_off", "overlap_noise": "post_mass_on",
            "quantity_skew_extreme": "pre_footprint_on", "wine": "pre_footprint_off", "satimage": "post_none_off",
            "pendigits": "pre_none_off", "digits_pca16": "pre_footprint_off", "digits_pca32": "pre_footprint_off",
            "letter": "pre_mass_off", "mnist784_pca32 (20c)": "pre_footprint_off",
            "mnist784_pca32 (50c)": "pre_footprint_off"}   # as printed in the previous version of tab:main
_best_diff = [c for c in CONFIGS if BEST[c] != OLD_BEST[c]]
H1 = pair("L5:pre_mass_off", "L5:post_footprint_on", "H1")
PM1GF5 = pair("L1:pre_mass_off", "L5:post_footprint_on", "PM1-GF5")
PM5GF = T(H1, E)
PM1GF = T(PM1GF5, E)
BESTGF = [one_test(pair(f"L5:{BEST[c]}", "L5:post_footprint_on"), c, BLOCKS[E], "mean_acc") for c in CONFIGS]
_adj = R.holm([r["p_exact"] for r in BESTGF])
for r, a in zip(BESTGF, _adj):
    r["p_holm"], r["hyp"], r["block"] = a, "BEST-GF", E
ALL_ROWS.extend(BESTGF)
H1B = {b: T(H1, b) for b in BLOCKS}
for b in BLOCKS:
    check(uh(H1B[b]) == frozen_uh(b, "H1", "mean_acc"), f"H1 {b} equals the frozen summary")


def pm(r):
    if r["status"] != "ok" or r["p_exact"] >= ALPHA:
        return ""
    return "+" if r["direction"] == "a" else "-"


def mw(arm, c, b=E, sup=""):
    m, w = mean_of(arm, c, b), mean_of(arm, c, b, "worst_acc")
    return f"{m:.3f}/{w:.3f}" + (f"$^{{{sup}}}$" if sup else "")


def main_cells(c):
    return [mw("L5:post_none_off", c), mw("L5:post_footprint_on", c),
            mw("L5:pre_mass_off", c, sup=pm(row_of(PM5GF, c))), mw("L1:pre_mass_off", c, sup=pm(row_of(PM1GF, c))),
            r"\texttt{" + BEST[c].replace("_", r"\_") + "}", mw(f"L5:{BEST[c]}", c), cell(BESTGF, c)] + \
        [cell(H1B[b], c) for b in BLOCKS]


body = config_block(12, main_cells)
body += [r"\midrule",
         lab(4, r"Significantly better/worse than GF-PFedFCM") + f" & {uh(PM5GF)} & {uh(PM1GF)} & & & {uh(BESTGF)} & "
         + " & ".join(uh(H1B[b]) for b in BLOCKS) + r" \\"]
head = [r" & & \multicolumn{7}{c}{Seeds $0$--$9$ (exploratory), mean/worst client ACC} & "
        r"\multicolumn{3}{c}{\texttt{pre\_mass\_off}$-$GF-PFedFCM (H1)} \\",
        r"\cmidrule(lr){3-9}\cmidrule(lr){10-12}",
        r" & & & & \multicolumn{2}{c}{\texttt{pre\_mass\_off}} & \multicolumn{2}{c}{Best of 12 cells} & & & & \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Configuration & $m^*$ & F-FCM & GF-PFedFCM & $L=5$ & $L=1$ & Cell & ACC & Best$-$GF & $0$--$9$ & "
        r"$10$--$19$ & $20$--$29$ \\"]
# Online Resource 1 (moved out of the paper's Appendix A): references into the paper are typed
cap = (r"How the default cell was chosen: exploratory seeds $0$--$9$, $L=5$ unless stated. Mean/worst client ACC of "
       r"F-FCM, GF-PFedFCM, \texttt{pre\_mass\_off} and the best of the twelve cells, the row maximum of "
       r"Table~\ref{tab:fullcells}, which is selected and scored on the same seeds; $^{+}$ ($^{-}$): significantly "
       r"higher (lower) than GF-PFedFCM, unadjusted. The \texttt{pre\_mass\_off} ($L=1$) column is a cross-$L$ "
       r"comparison (" + paper_ref("sec:results-rule") + r"). Last three columns: \texttt{pre\_mass\_off} minus "
       r"GF-PFedFCM ($L=5$) in the three seed blocks, the pre-registered H1 on seeds $20$--$29$ ("
       + paper_ref("tab:confirm", "Table") + r"). Bottom: significantly better/worse, unadjusted (Holm over 14 "
       r"configurations). " + MARKS_TXT + ". " + STATS_ESM + ".")
(OUT / "tabS_main.tex").write_text(table("tab:main", cap, "@{}lcccccl ccccc@{}".replace(" ", ""), head, body),
                                   encoding="utf-8")
mdline("## 3. Exploratory selection (tab:main)")
mdline()
mdline(f"- best cell per configuration (argmax of the unrounded mean over seeds 0-9, first in the Table S1 order on "
       f"ties) equals the previous table on {14 - len(_best_diff)} of 14 configurations"
       + (f"; differs on {', '.join(PLAIN[c] for c in _best_diff)}" if _best_diff else ""))
mdline(f"- pre_mass_off (L=5) vs GF-PFedFCM (L=5), seeds 0-9: {uh(PM5GF)}; pre_mass_off (L=1) vs GF-PFedFCM (L=5), "
       f"seeds 0-9 (cross-L): {uh(PM1GF)}; best cell vs GF-PFedFCM, seeds 0-9: {uh(BESTGF)}")
n_pfo_best = sum(1 for c in CONFIGS if BEST[c] == "pre_footprint_off")
gap_pfo = max(max(mean_of(f"L5:{cl}", c, E) for cl in CELLS12) - mean_of("L5:pre_footprint_off", c, E)
              for c in CONFIGS)
mdline(f"- pre_footprint_off is the best cell on {n_pfo_best} configurations and within {gap_pfo:.3f} of the row "
       f"maximum everywhere (seeds 0-9)")
mdline()
check(not _best_diff, "the best cell per configuration equals the previous tab:main on all 14 configurations")

# ================================================================================================ (4) tab:footprintmass
FMP = pair("L5:post_footprint_off", "L5:post_mass_off", "FM-POST-off")
FMR = pair("L5:pre_footprint_off", "L5:pre_mass_off", "FM-PRE-off")
FMPon = pair("L5:post_footprint_on", "L5:post_mass_on", "FM-POST-on")
FMRon = pair("L5:pre_footprint_on", "L5:pre_mass_on", "FM-PRE-on")
FM = {}
for b in BLOCKS:
    FM[("POST", b)] = T(FMP, b)
    FM[("PRE", b)] = T(FMR, b)
FM[("POSTon", E)] = T(FMPon, E)
FM[("PREon", E)] = T(FMRon, E)
FMCOLS = [("POST", b) for b in BLOCKS] + [("PRE", b) for b in BLOCKS] + [("POSTon", E), ("PREon", E)]
body = config_block(10, lambda c: [cell(FM[k], c) for k in FMCOLS])
body += [r"\midrule",
         lab(2, "Significantly positive/negative") + " & " +
         " & ".join(f"{counts(FM[k])[0]}/{counts(FM[k])[1]}" for k in FMCOLS) + r" \\",
         lab(2, "Holm-adjusted (14 configurations)") + " & " +
         " & ".join(f"{counts(FM[k], 'p_holm')[0]}/{counts(FM[k], 'p_holm')[1]}" for k in FMCOLS) + r" \\",
         lab(2, r"Significant and $|\Delta|\ge0.005$") + " & " +
         " & ".join(f"{mrd(FM[k])[0]}/{mrd(FM[k])[1]}" for k in FMCOLS) + r" \\"]
head = [r" & & \multicolumn{3}{c}{POST, personalization off} & \multicolumn{3}{c}{PRE, personalization off} & "
        r"\multicolumn{2}{c}{Personalization on} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"Configuration & $m^*$ & $0$--$9$ & $10$--$19$ & $20$--$29$ & $0$--$9$ & $10$--$19$ & $20$--$29$ & "
        r"POST, $0$--$9$ & PRE, $0$--$9$ \\"]
# Online Resource 1 (moved out of the paper's Appendix A): references into the paper are typed
cap = (r"Footprint gate minus mass gate (\texttt{*\_footprint\_*} minus \texttt{*\_mass\_*}) in mean client ACC at "
       r"the same mass point and personalization setting, $L=5$; positive favours the footprint. Seeds $10$--$19$: the "
       r"PRE contrast was a comparison of the first held-out look, the POST contrast uses the re-runs of "
       + paper_ref("sec:protocol-metrics") + r" (exploratory); seeds $20$--$29$: post hoc, not pre-registered. "
       r"Personalized cells stop early as in the reference implementation (" + paper_ref("sec:protocol-runs") + "). "
       + MARKS_TXT + ". " + STATS_ESM + ".")
(OUT / "tabS_footprintmass.tex").write_text(table("tab:footprintmass", cap, "@{}lccccccccc@{}", head, body),
                                            encoding="utf-8")
mdline("## 4. Footprint minus mass gate (tab:footprintmass)")
mdline()
for k in FMCOLS:
    rows = FM[k]
    mdline(f"- {k[0]} {k[1]}: {uh(rows)}; min. reported difference {mrd(rows)[0]}/{mrd(rows)[1]}; footprint better: "
           f"{cfg_list(rows, 'a')}; mass better: {cfg_list(rows, 'b')}")
OLD_FM = {("POST", "s00-09"): ("3/0", "1/0"), ("POSTon", "s00-09"): ("4/0", "1/0"), ("PRE", "s00-09"): ("1/0", "0/0"),
          ("PREon", "s00-09"): ("2/0", "2/0"), ("PRE", "s10-19"): ("3/1", "0/0")}   # previous tab:footprintmass
for k, (u_old, h_old) in OLD_FM.items():
    u, h = counts(FM[k]), counts(FM[k], "p_holm")
    mdline(f"  - previous printed counts {k}: {u_old} (Holm {h_old}); exact test: {u[0]}/{u[1]} (Holm {h[0]}/{h[1]})"
           + ("" if (f"{u[0]}/{u[1]}", f"{h[0]}/{h[1]}") == (u_old, h_old) else "  <-- CHANGED"))
mdline()

# ================================================================================================ (5) tab:gfgain
GFF = pair("L5:post_footprint_on", "L5:post_none_off", "GF-FFCM")
GAF = pair("L5:post_footprint_off", "L5:post_none_off", "GATE-FFCM")
H9 = pair("L5:post_footprint_on", "L5:post_footprint_off", "H9")
GG = {}
for b in BLOCKS:
    GG[("GF", b)] = T(GFF, b)
    GG[("GATE", b)] = T(GAF, b)
    GG[("H9", b)] = T(H9, b)
    check(uh(GG[("H9", b)]) == frozen_uh(b, "H9", "mean_acc"), f"H9 {b} equals the frozen summary")
GGCOLS = [(k, b) for k in ("GF", "GATE", "H9") for b in BLOCKS]
PERS = [("pre_footprint", r"\texttt{pre\_footprint\_on}$-$\texttt{pre\_footprint\_off}"),
        ("pre_mass", r"\texttt{pre\_mass\_on}$-$\texttt{pre\_mass\_off}"),
        ("post_mass", r"\texttt{post\_mass\_on}$-$\texttt{post\_mass\_off}"),
        ("post_none", r"\texttt{post\_none\_on}$-$\texttt{post\_none\_off} (ungated)"),
        ("pre_none", r"\texttt{pre\_none\_on}$-$\texttt{pre\_none\_off} (ungated)")]
PR = {(k, b): T(pair(f"L5:{k}_on", f"L5:{k}_off", f"PERS-{k}"), b) for k, _ in PERS for b in BLOCKS}
PRO = {(k, b): T(pair(f"L5:{k}_on", f"L5:{k}_off", f"PERS-{k}"), b, "fcm_obj") for k, _ in PERS for b in BLOCKS}
body = config_block(11, lambda c: [cell(GG[k], c) for k in GGCOLS])
body += [r"\midrule",
         lab(2, "Significantly positive/negative") + " & " +
         " & ".join(f"{counts(GG[k])[0]}/{counts(GG[k])[1]}" for k in GGCOLS) + r" \\",
         lab(2, "Holm-adjusted (14 configurations)") + " & " +
         " & ".join(f"{counts(GG[k], 'p_holm')[0]}/{counts(GG[k], 'p_holm')[1]}" for k in GGCOLS) + r" \\",
         lab(2, r"Significant and $|\Delta|\ge0.005$") + " & " +
         " & ".join(f"{mrd(GG[k])[0]}/{mrd(GG[k])[1]}" for k in GGCOLS) + r" \\",
         r"\midrule",
         lab(8, r"\emph{Personalization effect (on$-$off), other cells: significantly positive/negative, unadjusted "
                r"(Holm)}") + r" & $0$--$9$ & $10$--$19$ & $20$--$29$ \\"]
for k, txt in PERS:
    body.append(lab(8, txt) + " & " + " & ".join(uh(PR[(k, b)]) for b in BLOCKS) + r" \\")
# rounds executed by the personalized cells (early stop), fresh and earlier blocks
NR = {}
for k in ("post_footprint", "pre_footprint", "pre_mass", "post_mass", "post_none", "pre_none"):
    for b in BLOCKS:
        v = [_num(TAB[(f"L5:{k}_on", c, s)]["n_rounds"]) for c in CONFIGS for s in BLOCKS[b]]
        NR[(k, b)] = (float(np.mean(v)), sum(1 for x in v if x < 50), len(v))
# counts that differ from the earlier analysis (SciPy's exact default test; previous version of the table)
OLD_GG = {("GF", "s00-09"): ("9/1", "7/1"), ("GF", "s10-19"): ("10/0", "8/0"), ("GATE", "s00-09"): ("10/1", "9/1"),
          ("GATE", "s10-19"): ("11/0", "11/0"), ("H9", "s00-09"): ("0/6", "0/3"), ("H9", "s10-19"): ("1/5", "0/5")}
GG_TXT = {"GF": "GF-PFedFCM minus F-FCM", "GATE": r"\texttt{post\_footprint\_off} minus F-FCM",
          "H9": r"GF-PFedFCM minus \texttt{post\_footprint\_off}"}
_chg = []
for (k, b), (u_old, h_old) in OLD_GG.items():
    u, h = counts(GG[(k, b)]), counts(GG[(k, b)], "p_holm")
    if f"{u[0]}/{u[1]}" != u_old:
        _chg.append(f"the unadjusted count of {GG_TXT[k]} on seeds {BLK[b]} is {u[0]}/{u[1]} ({u_old} in the earlier "
                    f"analysis)")
    if f"{h[0]}/{h[1]}" != h_old:
        _chg.append(f"the Holm count of {GG_TXT[k]} on seeds {BLK[b]} is {h[0]}/{h[1]} ({h_old} in the earlier "
                    f"analysis, whose test did not tie differences that are equal to twelve decimals)")
GG_CHANGE = ("Under the exact test of Section~\\ref{sec:protocol-stats}, " + "; ".join(_chg) + ". ") if _chg else ""
head = [r" & & \multicolumn{3}{c}{GF-PFedFCM$-$F-FCM} & \multicolumn{3}{c}{\texttt{post\_footprint\_off}$-$F-FCM} & "
        r"\multicolumn{3}{c}{GF-PFedFCM$-$\texttt{post\_footprint\_off}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Configuration & $m^*$ & $0$--$9$ & $10$--$19$ & $20$--$29$ & $0$--$9$ & $10$--$19$ & $20$--$29$ & "
        r"$0$--$9$ & $10$--$19$ & $20$--$29$ \\"]
cap = (r"Where the gains of GF-PFedFCM over F-FCM come from, $L=5$: the footprint gate alone "
       r"(\texttt{post\_footprint\_off} minus F-FCM) and the personalization effect (GF-PFedFCM minus "
       r"\texttt{post\_footprint\_off}; offsets and evaluation model change together), in the three seed blocks. "
       r"The personalization effect on seeds $20$--$29$ is the pre-registered H9; the other contrasts on seeds "
       r"$10$--$29$ are post hoc. Personalized cells stop once the global prototypes move by less than $10^{-5}$; "
       r"rerunning GF-PFedFCM, \texttt{pre\_footprint\_on} and \texttt{pre\_mass\_on} without this rule on seeds "
       r"$0$--$19$ (\texttt{post\_mass\_on} on $0$--$9$) left every mean client ACC unchanged "
       r"(Section~\ref{sec:protocol-runs}); on seeds $20$--$29$ GF-PFedFCM stopped after "
       rf"{NR[('post_footprint', 's20-29')][0]:.1f} rounds on average; a post hoc rerun without the rule "
       r"(Section~\ref{app:calibration}) left every accuracy unchanged. The other personalized cells were not rerun "
       r"on these seeds. "
       r"Bottom: the personalization effect of the other personalized cells. " + GG_CHANGE + MARKS_TXT + "." +
       # round 3: GG_CHANGE already names the test section; the repeated pointer is dropped so the float fits its page
       ("" if GG_CHANGE else " " + STATS + "."))
(OUT / "tab_gfgain.tex").write_text(table("tab:gfgain", cap, "@{}lcccccccccc@{}", head, body), encoding="utf-8")
mdline("## 5. Personalization and the gate (tab:gfgain)")
mdline()
for k in GGCOLS:
    rows = GG[k]
    mdline(f"- {k[0]} {k[1]}: {uh(rows)}; min. reported difference {mrd(rows)[0]}/{mrd(rows)[1]}; positive: "
           f"{cfg_list(rows, 'a')}; negative: {cfg_list(rows, 'b')}")
for (k, _), b in [(p, b) for p in PERS for b in BLOCKS]:
    mdline(f"- personalization effect {k}, {b}: ACC {uh(PR[(k, b)])}; OBJ {uh(PRO[(k, b)])}")
for (k, b), (mean_r, n_early, n_all) in NR.items():
    mdline(f"- rounds executed, {k}_on L=5, {b}: mean {mean_r:.1f}; {n_early} of {n_all} runs stopped before round 50")
# ungated personalization, seeds 0-9, detail (A1 / FACTS_R2 (b))
for k in ("post_none", "pre_none"):
    rows = PR[(k, "s00-09")]
    falls = sum(1 for r in rows if r["status"] == "ok" and r["mean_diff_raw"] < 0)
    rises = sum(1 for r in rows if r["status"] == "ok" and r["mean_diff_raw"] > 0)
    mdline(f"  - ungated personalization {k}, seeds 0-9: mean falls on {falls}, rises on {rises} of 14; "
           f"significant: {uh(rows)}; largest fall {min(r['mean_diff_raw'] for r in rows):+.3f} "
           f"({PLAIN[min(rows, key=lambda r: r['mean_diff_raw'])['dataset']]}); MNIST rows: " + "; ".join(
               f"{PLAIN[c]} {row_of(rows, c)['mean_diff_raw']:+.3f} (p={row_of(rows, c)['p_exact']:.4f})"
               for c in ("mnist784_pca32 (20c)", "mnist784_pca32 (50c)")))
OLD_GG = {("GF", "s00-09"): ("9/1", "7/1"), ("GF", "s10-19"): ("10/0", "8/0"), ("GATE", "s00-09"): ("10/1", "9/1"),
          ("GATE", "s10-19"): ("11/0", "11/0"), ("H9", "s00-09"): ("0/6", "0/3"), ("H9", "s10-19"): ("1/5", "0/5")}
for k, (u_old, h_old) in OLD_GG.items():
    u, h = counts(GG[k]), counts(GG[k], "p_holm")
    mdline(f"  - previous printed counts {k}: {u_old} (Holm {h_old}); exact test: {u[0]}/{u[1]} (Holm {h[0]}/{h[1]})"
           + ("" if (f"{u[0]}/{u[1]}", f"{h[0]}/{h[1]}") == (u_old, h_old) else "  <-- CHANGED"))
OLD_PERS = {("pre_footprint", "s00-09"): "0/4", ("pre_footprint", "s10-19"): "0/5", ("pre_mass", "s00-09"): "0/5",
            ("pre_mass", "s10-19"): "0/5", ("post_mass", "s00-09"): "0/6"}
for k, u_old in OLD_PERS.items():
    u = counts(PR[k])
    mdline(f"  - previous printed personalization count {k}: {u_old}; exact test: {u[0]}/{u[1]}"
           + ("" if f"{u[0]}/{u[1]}" == u_old else "  <-- CHANGED"))
mdline()

# ================================================================================================ (6) tab:fuzziness
FZ_ROWS = [("D", "Substitution interaction, positive/negative", None),
           ("MG", r"\texttt{pre\_mass\_off} vs \texttt{pre\_none\_off} (mass gate)", ("pre_mass_off", "pre_none_off")),
           ("PM-F", r"\texttt{pre\_mass\_off} vs F-FCM", ("pre_mass_off", "post_none_off")),
           ("PM-GF", r"\texttt{pre\_mass\_off} vs GF-PFedFCM", ("pre_mass_off", "post_footprint_on")),
           ("PERS", r"Personalization: GF-PFedFCM vs \texttt{post\_footprint\_off}",
            ("post_footprint_on", "post_footprint_off"))]
FZ = {}
for mval in MS:
    for key, _, ab in FZ_ROWS:
        if ab is None:
            h = inter(f"M{mval}:pre_none_off", f"M{mval}:post_none_off", f"M{mval}:pre_footprint_off",
                      f"M{mval}:post_footprint_off", f"FZ-D-{mval}")
            FZ[(key, mval, "mean_acc")] = T(h, E, "mean_acc", FUZZ_CONFIGS)
        else:
            h = pair(f"M{mval}:{ab[0]}", f"M{mval}:{ab[1]}", f"FZ-{key}-{mval}")
            FZ[(key, mval, "mean_acc")] = T(h, E, "mean_acc", FUZZ_CONFIGS)
            FZ[(key, mval, "fcm_obj")] = T(h, E, "fcm_obj", FUZZ_CONFIGS)
# m = 2.0 reruns reproduce the main runs
_n_same = 0
for c in FUZZ_CONFIGS:
    for cl in ("post_none_off", "post_footprint_off", "post_footprint_on", "pre_none_off", "pre_footprint_off",
               "pre_mass_off"):
        for s in BLOCKS[E]:
            _n_same += int(_num(TAB[(f"M2.0:{cl}", c, s)]["mean_acc"]) == _num(TAB[(f"L5:{cl}", c, s)]["mean_acc"]))
check(_n_same == 480, "the m = 2.0 fuzzifier reruns reproduce the mean client ACC of the main runs in all 480 runs")
body = []
for key, txt, ab in FZ_ROWS:
    cells_ = []
    for mval in MS:
        cells_.append(uh(FZ[(key, mval, "mean_acc")]))
        cells_.append("--" if ab is None else uh(FZ[(key, mval, "fcm_obj")]))
    body.append(txt + " & " + " & ".join(cells_) + r" \\")
head = [r" & \multicolumn{2}{c}{$m=1.3$} & \multicolumn{2}{c}{$m=1.5$} & \multicolumn{2}{c}{$m=2.0$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Comparison & ACC & OBJ & ACC & OBJ & ACC & OBJ \\"]
cap = (r"Fuzzifier sensitivity: the eight configurations non-degenerate at every $m\in\{1.3,1.5,2.0\}$, rerun at "
       r"each $m$ (exploratory seeds $0$--$9$, $L=5$; at $m=2.0$, their $m^*$, identical to the main runs in all $480$ "
       r"runs). Entries: configurations (of 8) on which the first method is significantly better/worse in mean client "
       r"ACC and in OBJ (better: lower), unadjusted and, in parentheses, Holm-adjusted over the 8 configurations; for "
       r"the interaction \eqref{eq:protocol-did} significantly positive/negative, on ACC only. Personalized cells: "
       r"equal round budgets not established at $m=1.3$ and $1.5$ (Section~\ref{sec:protocol-runs}). " + STATS + ".")
(OUT / "tab_fuzziness.tex").write_text(table("tab:fuzziness", cap, "@{}lcccccc@{}", head, body, sep="4pt"),
                                        encoding="utf-8")
mdline("## 6. Fuzzifier reruns (tab:fuzziness)")
mdline()
OLD_FZ = {("D", "1.3", "mean_acc"): "4/0", ("D", "1.5", "mean_acc"): "4/0", ("D", "2.0", "mean_acc"): "5/0",
          ("MG", "1.3", "mean_acc"): "2/0", ("MG", "1.3", "fcm_obj"): "0/5", ("MG", "1.5", "mean_acc"): "3/0",
          ("MG", "1.5", "fcm_obj"): "0/6", ("MG", "2.0", "mean_acc"): "1/1", ("MG", "2.0", "fcm_obj"): "0/6",
          ("PM-F", "1.3", "mean_acc"): "3/0", ("PM-F", "1.3", "fcm_obj"): "8/0", ("PM-F", "1.5", "mean_acc"): "5/1",
          ("PM-F", "1.5", "fcm_obj"): "8/0", ("PM-F", "2.0", "mean_acc"): "2/1", ("PM-F", "2.0", "fcm_obj"): "5/0",
          ("PM-GF", "1.3", "mean_acc"): "4/0", ("PM-GF", "1.3", "fcm_obj"): "0/3", ("PM-GF", "1.5", "mean_acc"): "3/0",
          ("PM-GF", "1.5", "fcm_obj"): "0/5", ("PM-GF", "2.0", "mean_acc"): "2/1", ("PM-GF", "2.0", "fcm_obj"): "0/7",
          ("PERS", "1.3", "mean_acc"): "0/3", ("PERS", "1.3", "fcm_obj"): "2/1", ("PERS", "1.5", "mean_acc"): "0/3",
          ("PERS", "1.5", "fcm_obj"): "2/2", ("PERS", "2.0", "mean_acc"): "0/3", ("PERS", "2.0", "fcm_obj"): "5/0"}
for k, u_old in OLD_FZ.items():
    u = counts(FZ[k])
    mdline(f"- {k[0]} m={k[1]} {k[2]}: {uh(FZ[k])}; previous printed {u_old}"
           + ("" if f"{u[0]}/{u[1]}" == u_old else "  <-- CHANGED"))
mdline()

# ================================================================================================ (7) other text numbers
mdline("## 7. Other numbers quoted in the text")
mdline()
# the second candidate cell against GF-PFedFCM (L=5) in the three blocks (s10-19: first held-out look; s20-29: post hoc)
PFGF = pair("L5:pre_footprint_off", "L5:post_footprint_on", "PF-GF")
for b in BLOCKS:
    rows = T(PFGF, b)
    mdline(f"- pre_footprint_off (L=5) vs GF-PFedFCM (L=5), {b}: {uh(rows)}; better: {cfg_list(rows, 'a')}; "
           f"worse: {cfg_list(rows, 'b')}")
PFPM = pair("L5:pre_footprint_off", "L5:pre_mass_off", "PF-PM")
# size-weighted averaging and avg2 (seeds 0-9, L=5; experiments/results/t6_baselines)
T6 = PROJECT / "experiments" / "results" / "t6_baselines" / "per_seed_metrics__mstar.csv"
for r in csv.DictReader(T6.open()):
    if r["method"] in ("fednova", "stallmann_avg2"):
        TAB[(f"B:{r['method']}", r["dataset"], int(r["seed"]))] = r
for arm, nm in (("B:fednova", "size-weighted averaging"), ("B:stallmann_avg2", "avg2")):
    rows = T(pair("L5:pre_mass_off", arm, f"PM5-{nm}"), "s00-09")
    mdline(f"- pre_mass_off (L=5) vs {nm} (L=5), seeds 0-9: {uh(rows)}; pre_mass_off better: {cfg_list(rows, 'a')}; "
           f"worse: {cfg_list(rows, 'b')}")
    mdline(f"  - {nm} mean client ACC (seeds 0-9): " +
           "; ".join(f"{PLAIN[c]} {mean_of(arm, c, 's00-09'):.3f}" for c in CONFIGS))
# pre_mass_off with L=1 against L=5 (a cross-L deployment comparison) at R=50
for b in BLOCKS:
    rows = T(pair("L1:pre_mass_off", "L5:pre_mass_off", "PM1-PM5"), b)
    mdline(f"- pre_mass_off L=1 vs L=5 (R=50), {b}: ACC {uh(rows)}; worse: {cfg_list(rows, 'b')}")
for b in BLOCKS:
    rows = T(pair("L1noES:post_footprint_on", "L5:post_footprint_on", "GF1-GF5"), b)
    mdline(f"- GF-PFedFCM L=1 (no early stop) vs L=5, {b}: ACC {uh(rows)}")
mdline()

# ================================================================================================ (8) round budgets
# Online Resource 1, Table S3 (tab:rounds) prints counts of the earlier analysis (SciPy's default test, no Holm);
# here the same contrasts under the frozen exact test with Holm over the nine configurations of the sweep.
for Rr in (5, 10, 20):
    for Lr in (1, 5):
        for r in csv.DictReader((T3D / f"per_seed_metrics__R{Rr}_L{Lr}.csv").open()):
            TAB[(f"R{Rr}L{Lr}:{r['cell']}", r["dataset"], int(r["seed"]))] = r
RB_CONFIGS = [c for c in CONFIGS if ("R5L1:pre_mass_off", c, 0) in TAB]
check(len(RB_CONFIGS) == 9, "the round-budget runs cover nine configurations")


def rb_arm(Rr, Lr, cl):
    """R < 50: the round-budget runs (T3's stopping rule); R = 50: the main and one-step runs of seeds 0-9."""
    return f"R{Rr}L{Lr}:{cl}" if Rr != 50 else f"L{Lr}:{cl}"


RB = {}
for Rr in (5, 10, 20, 50):
    RB[("PM", Rr)] = T(pair(rb_arm(Rr, 1, "pre_mass_off"), rb_arm(Rr, 5, "pre_mass_off"), f"RB-PM1-PM5-R{Rr}"),
                       E, "mean_acc", RB_CONFIGS)
    RB[("PMGF", Rr)] = T(pair(rb_arm(Rr, 1, "pre_mass_off"), rb_arm(Rr, 5, "post_footprint_on"),
                              f"RB-PM1-GF5-R{Rr}"), E, "mean_acc", RB_CONFIGS)
    RB[("GF", Rr)] = T(pair(rb_arm(Rr, 1, "post_footprint_on"), rb_arm(Rr, 5, "post_footprint_on"),
                            f"RB-GF1-GF5-R{Rr}"), E, "mean_acc", RB_CONFIGS)
    RB[("POST", Rr)] = T(pair(rb_arm(Rr, 1, "post_none_off"), rb_arm(Rr, 5, "post_none_off"),
                              f"RB-POST1-POST5-R{Rr}"), E, "mean_acc", RB_CONFIGS)
# counts printed in Online Resource 1, Table S3 (earlier analysis): pre_mass_off L1 vs L5 and vs GF (L5)
OLD_RB = {("PM", 5): "3/1", ("PM", 10): "5/0", ("PM", 20): "5/0", ("PM", 50): "6/0",
          ("PMGF", 5): "5/0", ("PMGF", 10): "6/0", ("PMGF", 20): "6/0", ("PMGF", 50): "8/0"}
mdline("## 8. Round budgets (Section 7.7; exploratory seeds 0-9, the nine configurations of the sweep)")
mdline()
mdline("Exact test, Holm over the 9 configurations of the sweep. R < 50: `per_seed_metrics__R{5,10,20}_L{1,5}.csv` "
       "(T3's stopping rule for GF-PFedFCM); R = 50: the T10 re-runs of the L=1 and L=5 cells, seeds 0-9.")
RB_TXT = {"PM": "pre_mass_off L=1 vs L=5", "PMGF": "pre_mass_off L=1 vs GF-PFedFCM L=5 (cross-L)",
          "GF": "GF-PFedFCM L=1 vs L=5", "POST": "post_none_off (ungated POST) L=1 vs L=5"}
for k in ("PM", "PMGF", "GF", "POST"):
    for Rr in (5, 10, 20, 50):
        rows = RB[(k, Rr)]
        u = counts(rows)
        extra = ""
        if (k, Rr) in OLD_RB:
            extra = f"; Table S3 prints {OLD_RB[(k, Rr)]}" + ("" if f"{u[0]}/{u[1]}" == OLD_RB[(k, Rr)]
                                                             else "  <-- DIFFERS")
        mdline(f"- {RB_TXT[k]}, R={Rr}: {uh(rows)}{extra}; L=1 better: {cfg_list(rows, 'a')}; "
               f"worse: {cfg_list(rows, 'b')}")
for k in ("PM", "GF", "POST"):
    cl = {"PM": "pre_mass_off", "GF": "post_footprint_on", "POST": "post_none_off"}[k]
    for c in RB_CONFIGS:
        vals = [f"R={Rr}: {mean_of(rb_arm(Rr, 1, cl), c, E):.3f} (L=1) / {mean_of(rb_arm(Rr, 5, cl), c, E):.3f} (L=5)"
                for Rr in (5, 10, 20, 50)]
        mdline(f"  - {cl}, {PLAIN[c]}, mean client ACC: " + "; ".join(vals))
n_higher = {k: sum(1 for c in RB_CONFIGS for Rr in (5, 10, 20, 50)
                   if mean_of(rb_arm(Rr, 1, cl), c, E) > mean_of(rb_arm(Rr, 5, cl), c, E))
            for k, cl in (("GF", "post_footprint_on"), ("PM", "pre_mass_off"))}
mdline(f"- (configuration, budget) pairs (of 36) with the higher mean at L=1: GF-PFedFCM {n_higher['GF']}, "
       f"pre_mass_off {n_higher['PM']}")
mdline()

# ================================================================================================ (9) L sweep
# Online Resource 1, Table S2 (tab:Lsweep) prints the interaction counts of the earlier analysis without Holm; here the
# same interaction under the frozen exact test with Holm over the eight configurations of the sweep.
for Lr in (1, 2, 10):
    for r in csv.DictReader((T3D / f"per_seed_metrics__L{Lr}.csv").open()):
        TAB[(f"LS{Lr}:{r['cell']}", r["dataset"], int(r["seed"]))] = r
LS_CONFIGS = [c for c in CONFIGS if ("LS1:pre_none_off", c, 0) in TAB]
check(len(LS_CONFIGS) == 8, "the L sweep covers eight configurations")


def ls_arm(Lr, cl):
    return f"LS{Lr}:{cl}" if Lr != 5 else f"L5:{cl}"


OLD_LS = {1: "6/1", 2: "6/1", 5: "6/0", 10: "6/0"}      # Online Resource 1, Table S2 (earlier analysis)
mdline("## 9. L sweep: substitution interaction with the footprint gate (Online Resource 1, Table S2)")
mdline()
mdline("Exact test, Holm over the 8 configurations of the sweep; exploratory seeds 0-9; L=5 = the main runs.")
for Lr in (1, 2, 5, 10):
    rows = T(inter(ls_arm(Lr, "pre_none_off"), ls_arm(Lr, "post_none_off"), ls_arm(Lr, "pre_footprint_off"),
                   ls_arm(Lr, "post_footprint_off"), f"LS-D-L{Lr}"), E, "mean_acc", LS_CONFIGS)
    u = counts(rows)
    mdline(f"- L={Lr}: interaction positive/negative {uh(rows)}; Table S2 prints {OLD_LS[Lr]}"
           + ("" if f"{u[0]}/{u[1]}" == OLD_LS[Lr] else "  <-- DIFFERS")
           + f"; positive: {cfg_list(rows, 'a')}; negative: {cfg_list(rows, 'b')}")
mdline()

# ================================================================================================ outputs
keys = []
for r in ALL_ROWS:
    for k in r:
        if k not in keys:
            keys.append(k)
with TESTS_CSV.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(ALL_ROWS)
mdline("## Cross-document references at this run")
mdline()
mdline("- " + ("; ".join(sorted(set(XREF_WARN))) + ": re-run after rebuilding main.tex and ESM_1.tex"
               if XREF_WARN else "all typed references resolved from main.aux and ESM_1.aux"))
mdline()
mdline("## Consistency checks passed")
mdline()
md.extend(f"- {c}" for c in CHECKS)
MD.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"tables written to {OUT}; {len(ALL_ROWS)} test rows; {len(CHECKS)} checks passed")
if XREF_WARN:
    print("unresolved cross-document references (re-run after rebuilding): " + "; ".join(sorted(set(XREF_WARN))))
