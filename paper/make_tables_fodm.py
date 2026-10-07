#!/usr/bin/env python3
"""make_tables_fodm.py -- generate the data tables of the FODM manuscript and of Online Resource 1.

Writes LaTeX table environments into paper2_fodm/tables_gen/ and the index tables_gen/TABLES_MAP.md.
Main-text tables are pulled into main.tex by flatten.py through a line ``%%INSERT tables_gen/<name>.tex``;
Online Resource 1 tables are pasted into ESM_1.tex (single file, no \\input).

Every number is read from, or computed from, the T10 outputs in experiments/results/t10/, the evidence CSVs in
paper2_fodm/_results/ (written by evidence_main.py / evidence_extra.py) and, for avg2 and size-weighted averaging of
the local centres (seeds 0-9 only), experiments/results/t6_baselines/per_seed_metrics__mstar.csv. Statistics are the
frozen ones of experiments/scripts/T10_review.py (sha256 checked against results/t10/ANALYSIS_FROZEN): exact two-sided
paired Wilcoxon signed-rank test (|d| <= 1e-12 dropped, mid-ranks, n' = effective n), Holm per comparison over its
configuration family (14; 12 with SC-FFCM at L=50). Counts read from the frozen summaries are cross-checked against a
recount from the per-configuration test rows, and the counts of EVIDENCE_T10_extra.md against this script's recount.

Run:  python3 make_tables_fodm.py        (exits non-zero if a consistency check fails)

Fix stage (2026-09-25): the former Appendix A tables (tab:substitution, tab:scffcm, tab:baselines, tab:gfgain,
tab:fuzziness) moved to Online Resource 1. tab:scffcm and tab:baselines are generated here in Online Resource 1 style
(typed references into the paper, resolved from main.aux); tab:substitution, tab:gfgain and tab:fuzziness, generated
by _results/results_stage_tables.py for the main text, are rewritten here for Online Resource 1 (references only; every
number is copied unchanged), and tab_L1 is copied with the three centralized budgets added to its caption.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
T10 = PROJECT / "experiments" / "results" / "t10"
T6 = PROJECT / "experiments" / "results" / "t6_baselines" / "per_seed_metrics__mstar.csv"
RES = HERE / "_results"
OUT = HERE / "tables_gen"
SCRIPT = PROJECT / "experiments" / "scripts" / "T10_review.py"
EXTRA_MD = HERE / "EVIDENCE_T10_extra.md"
MAIN_AUX = HERE / "main.aux"
ESM_AUX = HERE / "ESM_1.aux"

# ---------------------------------------------------------------------------------------------- frozen statistics
frozen = (T10 / "ANALYSIS_FROZEN").read_text().split()[0]
if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != frozen:
    sys.exit("T10_review.py differs from ANALYSIS_FROZEN; refusing to build tables")
_spec = importlib.util.spec_from_file_location("T10_review", SCRIPT)
R = importlib.util.module_from_spec(_spec)
sys.modules["T10_review"] = R
_spec.loader.exec_module(R)
CONFIGS, NON_MNIST, BLOCKS, ALPHA = R.CONFIGS, R.NON_MNIST, R.BLOCKS, R.ALPHA

CHECKS: list[str] = []
WARN: list[str] = []


def check(cond, msg):
    if not cond:
        sys.exit(f"CONSISTENCY CHECK FAILED: {msg}")
    CHECKS.append(msg)


# ---------------------------------------------------------------------------------------------- inputs
SUMM = pd.read_csv(T10 / "analysis_final_summary.csv")
TESTS = pd.read_csv(T10 / "analysis_final_tests.csv")
EA = pd.read_csv(T10 / "analysis_final_ea.csv")
EC = pd.read_csv(T10 / "analysis_final_ec.csv")
DEC = pd.read_csv(T10 / "analysis_final_decision.csv")
DEC19 = pd.read_csv(T10 / "analysis_s00-19_decision.csv")
CHG = pd.read_csv(T10 / "analysis_final_count_changes.csv")
XT = pd.read_csv(RES / "extra_tests_all.csv")
EBV = pd.read_csv(RES / "extra_A_eb_values.csv")
AUD2 = pd.read_csv(RES / "extra_D_audit_m2.csv")
LEDGER = pd.read_csv(RES / "extra_F_collapse_ledger.csv")
B4 = pd.read_csv(RES / "extra_B4_default_steps.csv")
MSTAR_T = pd.read_csv(T10 / "ef_audit_s00-04_mstar.csv")
AUD = pd.read_csv(T10 / "ef_audit_s00-04.csv")
EH = pd.read_csv(T10 / "theory" / "E_H_config.csv")
EDCAL = pd.read_csv(T10 / "ed_scffcm_cal_L1_R250.csv")
PEDCAL = {L: pd.read_csv(T10 / f"ee_pedrycz_cal_L{L}.csv") for L in (1, 5, 10)}
PEDRUN = {L: pd.read_csv(T10 / f"ee_pedrycz_L{L}_s00-19.csv") for L in (1, 5, 10)}
TAB, _missing = R.load_arms_t10(T10)
check(not _missing, "all T10 source CSVs used by the frozen analysis are present")

# ---------------------------------------------------------------------------------------------- configurations
NAME = {"cluster_skew_hard": "Cluster skew", "cluster_skew_overlap": "Cluster skew (overlap)",
        "dirichlet_0.03": "Dirichlet $0.03$", "dirichlet_0.1": "Dirichlet $0.1$", "overlap_noise": "Overlap/noise",
        "quantity_skew_extreme": "Quantity skew", "wine": "Wine", "satimage": "Satimage", "pendigits": "Pendigits",
        "digits_pca16": "Digits PCA-16", "digits_pca32": "Digits PCA-32", "letter": "Letter",
        "mnist784_pca32 (20c)": "MNIST PCA-32 (20 clients)", "mnist784_pca32 (50c)": "MNIST PCA-32 (50 clients)"}
SYNTH = ["cluster_skew_hard", "cluster_skew_overlap", "dirichlet_0.03", "dirichlet_0.1", "overlap_noise",
         "quantity_skew_extreme"]
_mp = MSTAR_T[MSTAR_T.init == "paper"].set_index("dataset")["paper_mstar"]
MSTAR = {c: float(_mp[c]) for c in CONFIGS}
GROUPS = [(r"Synthetic scenarios ($m^*=2$)", [c for c in CONFIGS if c in SYNTH]),
          (r"Real data, non-degenerate at $m=2$ ($m^*=2$)",
           [c for c in CONFIGS if c not in SYNTH and MSTAR[c] == 2.0]),
          (r"Real data, degenerate at $m=2$ (rescued, $m^*<2$)", [c for c in CONFIGS if MSTAR[c] < 2.0])]
RESCUED = GROUPS[2][1]
check(len(RESCUED) == 6, "six configurations are degenerate at m=2 (m* < 2)")
check(all(MSTAR[c] == 2.0 for c in SYNTH), "the synthetic scenarios have m* = 2")


# ---------------------------------------------------------------------------------------------- formatting
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def d3(x, marks="", sign=True):
    """A difference with three decimals, as the manuscript prints it: $+0.080$, $0.000$ for |x| < 0.0005."""
    x = num(x)
    if not math.isfinite(x):
        return "--"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    if s in ("+0.000", "-0.000"):
        s = "0.000"
    return f"${s}^{{{marks}}}$" if marks else f"${s}$"


def v3(x):
    x = num(x)
    return "--" if not math.isfinite(x) else f"{x:.3f}"


def pv(p):
    p = num(p)
    return "--" if not math.isfinite(p) else f"{p:.4f}"


def sci(x, digits=3):
    x = num(x)
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** e
    s = f"{mant:.{digits - 1}f}"
    if s.startswith("10"):
        e += 1
        s = f"{mant / 10:.{digits - 1}f}"
    return f"${s}\\times10^{{{e}}}$"


def big(x):
    """Validity indices: three decimals below 1000, else scientific with three significant digits."""
    x = num(x)
    if not math.isfinite(x):
        return "--"
    return v3(x) if abs(x) < 1000 else sci(x)


def cnt(a, b):
    return f"{int(a)}/{int(b)}"


def tt(cell: str) -> str:
    return r"\texttt{" + cell.replace("_", r"\_") + "}"


def marks_of(p, p_holm, n_eff=None, low_n=False):
    m = ""
    if num(p) < ALPHA:
        m += "*"
    if num(p_holm) < ALPHA:
        m += r"\dagger"
    if low_n and n_eff is not None and num(n_eff) <= 5:
        m += r"\circ"
    return m


def neff_note(n_eff, n_seeds=10):
    n = num(n_eff)
    return f"\\,({int(n)})" if math.isfinite(n) and n < n_seeds else ""


def aux_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", path.read_text(errors="ignore")):
        out[m.group(1)] = m.group(2)
    return out


MAINLAB = aux_labels(MAIN_AUX)
ESMLAB = aux_labels(ESM_AUX)


def paper_ref(label, kind="Section"):
    """Typed reference into the paper for Online Resource 1 (which cannot \\ref the paper)."""
    if label in MAINLAB:
        return f"{kind}~{MAINLAB[label]} of the paper"
    WARN.append(f"main.aux has no label {label}: re-run after rebuilding main.tex")
    return f"{kind}~?? of the paper"


def esm_ref(label):
    """Typed reference from the paper into Online Resource 1."""
    if label in ESMLAB:
        return f"Online Resource 1, Table~{ESMLAB[label]}"
    WARN.append(f"ESM_1.aux has no label {label}: re-run after inserting the Online Resource 1 tables and "
                "rebuilding ESM_1.tex")
    return "Online Resource 1, Table~S??"


STATS_MAIN = r"Tests and marks: Section~\ref{sec:protocol-stats}"


def stats_esm():
    return f"Tests and marks: {paper_ref('sec:protocol-stats')}"


MARKS_TXT = (r"$^{*}$: $p<0.05$ (exact test, unadjusted); $^{\dagger}$: Holm-adjusted $p<0.05$ over the "
             r"configurations of the comparison")


def table(label, caption, colspec, head, body, sep="3pt", place="htbp"):
    return "\n".join([
        rf"\begin{{table}}[{place}]", r"\centering", r"\tablefontsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\setlength{{\tabcolsep}}{{{sep}}}",
        r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}%", r"}\end{tabular}", r"\end{table}", ""])


def config_block(ncols, cells, configs=None, mstar=True):
    lines = []
    for title, members in GROUPS:
        members = [c for c in members if configs is None or c in configs]
        if not members:
            continue
        if lines:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for c in members:
            row = [NAME[c]] + ([f"{MSTAR[c]:.1f}"] if mstar else []) + list(cells(c))
            lines.append(" & ".join(row) + r" \\")
    return lines


def mc(n, text, align="c"):
    return rf"\multicolumn{{{n}}}{{{align}}}{{{text}}}"


def lab_row(ncols_label, text):
    return rf"\multicolumn{{{ncols_label}}}{{@{{}}l}}{{{text}}}"


GENERATED: list[dict] = []


def emit(name, tex, label, where, replaces, summary):
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    GENERATED.append(dict(name=name, label=label, where=where, replaces=replaces, summary=summary))


# ---------------------------------------------------------------------------------------------- count helpers
def srow(block, hyp, metric):
    s = SUMM[(SUMM.block == block) & (SUMM.hyp == hyp) & (SUMM.metric == metric)]
    return None if s.empty else s.iloc[0]


def recount(df, block, hyp, metric, col):
    t = df[(df.block == block) & (df.hyp == hyp) & (df.metric == metric) & (df.status == "ok")]
    return int(((t[col] < ALPHA) & (t.direction == "a")).sum()), int(((t[col] < ALPHA) & (t.direction == "b")).sum())


def mrd(df, block, hyp, metric):
    """Significant (unadjusted) and |mean difference| >= 0.005: the minimum reported difference convention."""
    t = df[(df.block == block) & (df.hyp == hyp) & (df.metric == metric) & (df.status == "ok")]
    s = t[(t.p_exact < ALPHA) & (t.mean_diff_raw.abs() >= 0.005)]
    return int((s.direction == "a").sum()), int((s.direction == "b").sum())


def tested(df, block, hyp, metric):
    t = df[(df.block == block) & (df.hyp == hyp) & (df.metric == metric) & (df.status == "ok")]
    return len(t), int((t.n_eff < 10).sum())


def ucount(block, hyp, metric, df=None):
    """'u (H)' from the frozen summary; recount from the per-configuration rows checked."""
    s = srow(block, hyp, metric)
    if s is None:
        return None
    src = TESTS if df is None else df
    if hyp.startswith("EA-"):
        src = EA
    if hyp.startswith("EC-"):
        src = EC
    ru, rh = recount(src, block, hyp, metric, "p_exact"), recount(src, block, hyp, metric, "p_holm")
    check(ru == (s.better_unadj, s.worse_unadj) and rh == (s.better_holm, s.worse_holm),
          f"recount of {hyp} {block} {metric} equals analysis_final_summary.csv")
    return f"{cnt(s.better_unadj, s.worse_unadj)} ({cnt(s.better_holm, s.worse_holm)})"


def xcount(block, hyp, metric):
    """'u (H)' recounted from _results/extra_tests_all.csv (comparisons outside the frozen analysis)."""
    t = XT[(XT.block == block) & (XT.hyp == hyp) & (XT.metric == metric)]
    if t.empty:
        return None
    u, h = recount(XT, block, hyp, metric, "p_exact"), recount(XT, block, hyp, metric, "p_holm")
    return u, h


def xc(block, hyp, metric):
    r = xcount(block, hyp, metric)
    return "n.r." if r is None else f"{cnt(*r[0])} ({cnt(*r[1])})"


# cross-check the recount of extra_tests_all.csv against the count tables of EVIDENCE_T10_extra.md
XLAB = {"CM-mass-SC5": "pre_mass_off L1 R250 vs SC-FFCM L5 R50", "CM-none-SC5": "ungated PRE L1 R250 vs SC-FFCM L5 R50",
        "CM-mass-PM5": "pre_mass_off L1 R250 vs pre_mass_off L5 R50",
        "CM-none-PM5": "ungated PRE L1 R250 vs pre_mass_off L5 R50", "CM-SC1R250-SC5": "SC-FFCM L1 R250 vs SC-FFCM L5 R50",
        "DEF-L1": "SC-FFCM default vs calibrated, L=1", "DEF-L5": "SC-FFCM default vs calibrated, L=5",
        "DEF-L50": "SC-FFCM default vs calibrated, L=50",
        "PED1-PM": "pre_mass_off L1 vs Pedrycz L1", "PED5-PM": "pre_mass_off L5 vs Pedrycz L5",
        "PED1-FFCM": "F-FCM (post_none_off) L1 vs Pedrycz L1", "PED5-FFCM": "F-FCM (post_none_off) L5 vs Pedrycz L5",
        "P05L1-PRE-POST": "P=0.5, L=1: PRE vs POST (ungated)", "P05L1-PM-GF": "P=0.5, L=1: pre_mass_off vs GF-PFedFCM",
        "P05L1-PM-SC": "P=0.5, L=1: pre_mass_off vs SC-FFCM", "P1L1-PRE-POST": "P=1, L=1: PRE vs POST (ungated)",
        "P1L1-PM-GF": "P=1, L=1: pre_mass_off vs GF-PFedFCM", "P1L1-PM-SC": "P=1, L=1: pre_mass_off vs SC-FFCM",
        "P05L5-PRE-POST": "P=0.5, L=5: PRE vs POST (ungated)", "P05L5-PM-GF": "P=0.5, L=5: pre_mass_off vs GF-PFedFCM",
        "P05L5-PM-SC": "P=0.5, L=5: pre_mass_off vs SC-FFCM", "P1L5-PRE-POST": "P=1, L=5: PRE vs POST (ungated)",
        "P1L5-PM-GF": "P=1, L=5: pre_mass_off vs GF-PFedFCM", "P1L5-PM-SC": "P=1, L=5: pre_mass_off vs SC-FFCM"}
for _L in (1, 5):
    for _cell in ("post_none_off", "pre_none_off", "pre_mass_off", "post_footprint_on"):
        XLAB[f"CH-L{_L}-{_cell}"] = f"L={_L} {_cell}: P=0.5 vs P=1"
    XLAB[f"CH-L{_L}-SC"] = f"L={_L} SC-FFCM: cfraction 0.5 vs 1"
_md_rows = {}
for line in EXTRA_MD.read_text().splitlines():
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) == 10 and parts[1].startswith("s") and "/" in parts[6]:
        _md_rows[(parts[0], parts[1], parts[2])] = (parts[6], parts[7])
_n_x = 0
for (hid, lab) in XLAB.items():
    for (blk, met) in {(b, m) for b, m in zip(XT[XT.hyp == hid].block, XT[XT.hyp == hid].metric)}:
        r = xcount(blk, hid, met)
        key = (lab, blk, met)
        check(key in _md_rows, f"EVIDENCE_T10_extra.md lists {key}")
        check(_md_rows[key] == (cnt(*r[0]), cnt(*r[1])),
              f"recount of {hid} {blk} {met} equals EVIDENCE_T10_extra.md ({_md_rows[key]})")
        _n_x += 1
CHECKS.append(f"{_n_x} count lines of EVIDENCE_T10_extra.md reproduced from extra_tests_all.csv")

# ================================================================================================ MAIN TEXT
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------------------------- (1) tab:confirm
HYP_ROWS = [
    ("H1", r"\texttt{pre\_mass\_off} vs GF-PFedFCM", "5", "first"),
    ("H2", r"\texttt{pre\_mass\_off} vs F-FCM", "5", "first"),
    ("H3", r"Interaction $D$ \eqref{eq:protocol-did}, footprint gate", "5", "positive"),
    ("H4", r"PRE vs POST, both ungated", "1", "first"),
    ("H5", r"\texttt{pre\_mass\_off} vs GF-PFedFCM", "1", "first"),
    ("H6", r"\texttt{pre\_mass\_off} vs SC-FFCM", "1 vs 1", "first"),
    ("H7", r"\texttt{pre\_mass\_off} vs SC-FFCM", "1 vs 5", "first"),
    ("H8", r"\texttt{pre\_mass\_off} vs SC-FFCM", "5 vs 5", "second"),
    ("H9", r"GF-PFedFCM vs \texttt{post\_footprint\_off}", "5", "no gain"),
]
SECOND_LOOK = {"H4", "H5", "H6", "H7", "H8", "H5-ES", "S-L50", "S-PRE-SC1", "S-PRE-SC5"}


def hcell(block, hyp, metric):
    u = ucount(block, hyp, metric)
    if u is None:
        return "n.r."
    if block == "s10-19" and hyp in SECOND_LOOK:
        u += r"$^{\S}$"
    return u


def hyp_rows(hid, desc, L, direc, rep, blocks=("s00-09", "s10-19", "s20-29")):
    n_t, n_low = tested(TESTS, "s20-29", hid, "mean_acc")
    s = srow("s20-29", hid, "mean_acc")
    check(n_t == s.n_tested, f"{hid}: tested count equals the summary")
    return [" & ".join([hid, desc, L, direc, "ACC"] + [hcell(b, hid, "mean_acc") for b in blocks]
                       + [f"{n_t} ({n_low})", rep]) + r" \\",
            " & ".join(["", "", "", "", "OBJ"] + [hcell(b, hid, "fcm_obj") for b in blocks] + ["", ""]) + r" \\"]


body, body_fresh = [], []
for i, (hid, desc, L, direc) in enumerate(HYP_ROWS):
    s = srow("s20-29", hid, "mean_acc")
    rep = ("cons." if hid == "H9" else "yes") if int(s.replicates) == 1 else "no"
    if i:
        body.append(r"\addlinespace[2pt]")
        body_fresh.append(r"\addlinespace[2pt]")
    body += hyp_rows(hid, desc, L, direc, rep)
    body_fresh += hyp_rows(hid, desc, L, direc, rep, blocks=("s20-29",))
check(all(int(srow("s20-29", h, "mean_acc").replicates) == 1 for h, *_ in HYP_ROWS),
      "all nine primary hypotheses carry the REPLICATES flag on the fresh block (H9 by its interpretation)")
body.append(r"\midrule")
body.append(lab_row(10, r"\emph{Secondary (declared before the fresh run; not used for replication)}") + r" \\")
body += hyp_rows("H5-ES", r"H5 with GF-PFedFCM's early stop", "1", "first", "--")
body_fresh.append(r"\midrule")
body_fresh.append(lab_row(8, r"\emph{Secondary (declared before the fresh run; not used for replication)}") + r" \\")
body_fresh += hyp_rows("H5-ES", r"H5 with GF-PFedFCM's early stop", "1", "first", "--", blocks=("s20-29",))
# family-wide sensitivity
hall = SUMM[SUMM.hyp.isin([f"H{i}" for i in range(1, 9)]) & (SUMM.metric == "mean_acc")]
check(int(hall.better_holm_all.sum() + hall.worse_holm_all.sum()) == 0,
      "family-wide Holm (K=112) counts are 0/0 for every H1-H8 in every block")
K_ALL = 8 * len(CONFIGS)
pmin = 2 / 2 ** 10
check(pmin > ALPHA / K_ALL, "1/512 exceeds 0.05/112: the family-wide Holm cannot reject with ten seeds")
check(pmin <= ALPHA / len(CONFIGS), "1/512 is below 0.05/14: the per-hypothesis Holm is attainable")
head = [
    r" & & & Stated & & \multicolumn{3}{c}{Seeds} & Tested & \\",
    r"\cmidrule(lr){6-8}",
    r"Hyp. & Comparison (first vs second) & $L$ & direction & Crit. & $0$--$9$ & $10$--$19$ & $20$--$29$ & "
    r"($n'<10$) & Rule met \\",
]
body_c = body
cap = (r"The pre-registered fresh block (seeds $20$--$29$, run once and analysed once; Section~\ref{sec:protocol-stages}) "
       r"and the same contrasts on the earlier blocks. Entries: configurations (of 14) on which the first method is "
       r"significantly better/the second is significantly better, unadjusted, and in parentheses after the Holm "
       r"adjustment over the 14 configurations of the hypothesis; for H3 the interaction is significantly "
       r"positive/negative, and for OBJ (second row of each hypothesis) ``better'' is lower. Seeds $0$--$9$: exploratory; seeds $10$--$19$: the first "
       r"held-out block for the $L=5$ cells (H1--H3, H9) and a second look ($^{\S}$) for every contrast with $L=1$ or "
       r"SC-FFCM, whose runs were analysed after the block had been unblinded. Tested: configurations with at least one "
       r"nonzero paired difference on seeds $20$--$29$ (mean client ACC) and, in parentheses, how many of them have an "
       r"effective sample size $n'<10$, which cannot pass the first Holm step. Rule met: the pre-registered replication rule (Holm count "
       r"in the stated direction at least one and not exceeded by the Holm count in the opposite direction); H9 has no "
       r"rule and is read as consistent (cons.) when no configuration shows a Holm-significant gain from "
       r"personalization, an interpretation fixed before the run. All nine primary hypotheses meet this rule "
       r"(H9 under its interpretation); the rule asks for a consistent direction, not for a majority of configurations. "
       r"H5 compares with GF-PFedFCM at $L=1$ run for $50$ rounds without early stopping. Holm over all "
       rf"H1--H8 $\times$ 14 configurations of a block ($K={K_ALL}$) gives $0/0$ for every hypothesis in every block, "
       rf"since $1/512>0.05/{K_ALL}$ (Section~\ref{{sec:results-confirm}}). Per-configuration differences and "
       rf"intervals of the fresh block: {esm_ref('tab:fresh-detail-a').replace('Table~', 'Tables~')}--"
       rf"{esm_ref('tab:fresh-detail-c').split('Table~')[-1]}. "
       + STATS_MAIN + ".")
# 2026-10-07 (compression): the paper keeps the fresh block only; the table with all three blocks moves to Online
# Resource 1 (tab:confirm-all, emitted after to_esm() is defined below).
CONFIRM_ALL_TEX = table("tab:confirm-all", cap.replace("Section~\\ref{sec:results-confirm}", "Section~\\ref{sec:results-confirm}"),
                        "@{}llccccccc c@{}".replace(" ", ""), head, body_c, sep="4pt")
head_fresh = [r" & & & Stated & & Seeds & Tested & \\",
              r"Hyp. & Comparison (first vs second) & $L$ & direction & Crit. & $20$--$29$ & ($n'<10$) & Rule met \\"]
cap_fresh = (r"The pre-registered fresh block (seeds $20$--$29$, run once and analysed once; Section~\ref{sec:protocol-stages}). "
             r"Entries: configurations (of 14) on which the first method is significantly better/worse, unadjusted and, in "
             r"parentheses, after the Holm adjustment over the 14 configurations of the hypothesis; for H3, significantly "
             r"positive/negative interactions; for OBJ (second row of each hypothesis) better is lower. Tested: "
             r"configurations with a nonzero paired difference in mean client ACC and, in parentheses, those with an "
             r"effective sample size $n'<10$, which cannot pass the first Holm step. Rule met: Holm count in the stated "
             r"direction at least one and not exceeded by the opposite Holm count; H9 is read as consistent (cons.) when no "
             r"configuration shows a Holm-significant gain from personalization. H5 compares with GF-PFedFCM at $L=1$ "
             r"without early stopping. The same contrasts on seeds $0$--$9$ and $10$--$19$: "
             + esm_ref("tab:confirm-all") + r"; per-configuration differences with $95\%$ $t$-intervals: "
             + esm_ref("tab:fresh-detail-a").replace("Table~", "Tables~") + r"--"
             + esm_ref("tab:fresh-detail-c").split("Table~")[-1] + ".")
emit("tab_confirm", table("tab:confirm", cap_fresh, "@{}llcccccc@{}", head_fresh, body_fresh, sep="4pt"),
     "tab:confirm", "Main text, Section 7 (first results table; the fresh block leads)",
     "REPLACES the inline tab:confirm of src/07_results.tex (held-out seeds 10-19 comparisons at L=5). Its "
     "per-configuration held-out columns and the counts it alone carried (pre_footprint_off vs GF-PFedFCM 8/2, Holm 8/1; "
     "footprint-mass at PRE 3/1, Holm 0/0) are NOT in the new table: keep them in the text only if re-sourced "
     "(source: experiments/results/t8_e7_confirm_analysis.txt) or drop them.",
     "H1-H9 and H5-ES: stated direction; unadjusted (Holm) counts on mean client ACC and on OBJ for seeds 0-9, "
     "10-19 (second look marked) and 20-29; tested and n'<10 on the fresh block; replication flag; family-wide Holm "
     "(K=112) note.")

# ---------------------------------------------------------------------------------------------- (2) tab:gates-interaction
GATES = ["footprint", "mass", "fpmask", "massmask"]
doff_h = [dict(id="DOFF", kind="pair", a="L5:pre_none_off", b="L5:post_none_off", expect=None)]
doff = pd.DataFrame(R.run_tests(TAB, doff_h, "s20-29", BLOCKS["s20-29"], "mean_acc"))


def ea_cell(gate, c, metric="mean_acc", block="s20-29"):
    r = EA[(EA.block == block) & (EA.hyp == f"EA-{gate}") & (EA.metric == metric) & (EA.dataset == c)].iloc[0]
    if r.status != "ok":
        return d3(0.0) + r"\,(0)"
    return d3(r.mean_diff_raw, marks_of(r.p_exact, r.p_holm)) + neff_note(r.n_eff)


def doff_cell(c):
    # Round 2 (P8): Delta_off carries the Holm mark as well (Holm over its 14 configurations, as in tab:substitution),
    # so that the Holm-significant Satimage loss is visible here too.
    r = doff[doff.dataset == c].iloc[0]
    if r.status != "ok":
        return d3(0.0) + r"\,(0)"
    return d3(r.mean_diff_raw, marks_of(r.p_exact, r.p_holm)) + neff_note(r.n_eff)


_doff_h = recount(doff, "s20-29", "DOFF", "mean_acc", "p_holm")
_sat = doff[doff.dataset == "satimage"].iloc[0]
check(_doff_h == (10, 1) and _sat.p_holm < ALPHA and _sat.mean_diff_raw < 0,
      "Delta_off (fresh block): Holm 10/1, the negative being Satimage (as in tab:substitution)")


body = config_block(7, lambda c: [doff_cell(c)] + [ea_cell(g, c) for g in GATES])
body.append(r"\midrule")
body.append(lab_row(7, r"\emph{Significantly positive/negative, unadjusted (Holm)}") + r" \\")
BLK_TXT = {"s00-09": r"seeds $0$--$9$", "s10-19": r"seeds $10$--$19$", "s20-29": r"seeds $20$--$29$"}
for metric, mt in (("mean_acc", "ACC"), ("fcm_obj", "OBJ")):
    for b in ("s00-09", "s10-19", "s20-29"):
        body.append(lab_row(3, f"{mt}, {BLK_TXT[b]}") + " & " +
                    " & ".join(ucount(b, f"EA-{g}", metric) for g in GATES) + r" \\")
for b in ("s00-09", "s10-19", "s20-29"):
    for metric in ("mean_acc", "fcm_obj"):
        a, bb = ucount(b, "EA-footprint", metric), ucount(b, "H3", metric)
        check(a == bb, f"EA-footprint equals H3 on {b} {metric}")
head = [r" & & PRE$-$POST & \multicolumn{4}{c}{Interaction $D$ with the gate} \\",
        r"\cmidrule(lr){4-7}",
        r"Configuration & $m^*$ & no gate & footprint & mass & \texttt{fpmask} & \texttt{massmask} \\"]
cap = (r"Substitution interaction \eqref{eq:protocol-did} of the PRE mass point with each of four gates, no "
       r"personalization, $L=5$: per configuration on the fresh block (seeds $20$--$29$), mean over seeds of $D_s$ in mean "
       r"client ACC, where the footprint gate in \eqref{eq:protocol-did} is replaced by the gate of the column; positive: "
       r"the gate takes over part of the PRE$-$POST difference. The footprint and mass gates carry the pre-adaptation "
       r"mass; the mask gates \texttt{fpmask} and \texttt{massmask} keep only which pairs are clipped "
       r"(Section~\ref{sec:protocol-additional}). PRE$-$POST, no gate: $\Delta^{\mathrm{off}}$, not a pre-registered "
       r"test, with its unadjusted and Holm marks (Holm over the 14 configurations; significantly positive/negative "
       rf"{_doff_h[0]}/{_doff_h[1]} after Holm); $D_s$ is read as substitution only where $\Delta^{{\mathrm{{off}}}}>0$. "
       r"In parentheses: the effective "
       r"sample size $n'$ where it is below $10$. Bottom: counts over the 14 configurations for each block, on mean "
       r"client ACC and on OBJ (positive: the interaction lowers OBJ, the orientation of the ACC rows). Only the "
       r"footprint column on ACC, seeds $20$--$29$, is a pre-registered primary test (H3; the footprint rows "
       r"equal the H3 counts in every block); the other gates are secondary analyses on seeds $20$--$29$ and "
       r"exploratory on seeds $0$--$19$. The identity of Corollary~\ref{cor:same-tilt} fixes the form of the interaction, not its sign "
       r"or size. " + MARKS_TXT + ". " + STATS_MAIN + ".")
# 2026-10-07 (compression): Figure 6 of the paper shows these values; the table moves to Online Resource 1.
GATES_TEX = table("tab:gates-interaction", cap, "@{}lcccccc@{}", head, body)

# ---------------------------------------------------------------------------------------------- (3) tab:decision
ACT = [a for a, _ in R.DECISION_ACTIONS]
ACT_TEX = {"F-FCM (post_none_off, L=5)": r"F-FCM ($L=5$)", "GF-PFedFCM (L=5)": r"GF-PFedFCM ($L=5$)",
           "ungated PRE (L=1)": r"PRE, ungated ($L=1$)", "pre_mass_off (L=1)": r"\texttt{pre\_mass\_off} ($L=1$)",
           "pre_mass_off (L=5)": r"\texttt{pre\_mass\_off} ($L=5$)", "SC-FFCM (L=5)": r"SC-FFCM ($L=5$)",
           "SC-FFCM (L=1, R=50)": r"SC-FFCM ($L=1$)"}
ACT_AB = {"F-FCM (post_none_off, L=5)": "F5", "GF-PFedFCM (L=5)": "GF5", "ungated PRE (L=1)": "PRE1",
          "pre_mass_off (L=1)": "PM1", "pre_mass_off (L=5)": "PM5", "SC-FFCM (L=5)": "SC5", "SC-FFCM (L=1, R=50)": "SC1"}
CRIT = list(R.DECISION_CRITERIA)
# recompute: seeds 10-29 must reproduce the frozen CSV, seeds 10-19 the frozen s00-19 CSV; 20-29 is derived
rows29, _ = R.decision_analysis(TAB, tuple(range(10, 30)))
rows19, _ = R.decision_analysis(TAB, BLOCKS["s10-19"])
rows2029, _ = R.decision_analysis(TAB, BLOCKS["s20-29"])


def dec_same(rows, df):
    a = pd.DataFrame(rows)
    for _, r in df.iterrows():
        q = a[(a.criterion == r.criterion) & (a.action == r.action)].iloc[0]
        if abs(q.mean_regret - r.mean_regret) > 1e-12 or abs(q.max_regret - r.max_regret) > 1e-12 \
                or q.rec_mean_regret != r.rec_mean_regret or q.rec_max_regret != r.rec_max_regret:
            return False
    return True


check(dec_same(rows29, DEC), "frozen decision_analysis on seeds 10-29 reproduces analysis_final_decision.csv")
check(dec_same(rows19, DEC19), "frozen decision_analysis on seeds 10-19 reproduces analysis_s00-19_decision.csv")
check(int(DEC.n_substituted.sum()) == 0, "decision analysis: no substituted runs")
D29, D2029 = pd.DataFrame(rows29), pd.DataFrame(rows2029)
D19 = DEC19


def recs(df, crit):
    s = df[df.criterion == crit]
    return (" / ".join(ACT_AB[a] for a in s[s.rec_mean_regret == 1].action),
            " / ".join(ACT_AB[a] for a in s[s.rec_max_regret == 1].action))


# Round 2 (P3): regrets to FOUR decimals. At three decimals SC-FFCM with L=5 and L=1 tie on two recommended entries
# (mean gACC regret, maximum OBJ regret); at four decimals every recommended (smallest) value of a column is unique.
DEC_DIG = 4
TIES3 = []   # (criterion, column, printed three-decimal value, actions tied with the recommended one)
for crit in CRIT:
    s = DEC[DEC.criterion == crit]
    for col, rec in (("mean_regret", "rec_mean_regret"), ("max_regret", "rec_max_regret")):
        best = s[s[rec] == 1]
        check(len(best) == 1, f"decision table: one recommended action for {crit} {col}")
        b4 = f"{best[col].iloc[0]:.{DEC_DIG}f}"
        check(int((s[col].map(lambda x: f"{x:.{DEC_DIG}f}") == b4).sum()) == 1,
              f"decision table: the recommended {crit} {col} ({b4}) is unique at {DEC_DIG} decimals")
        b3 = f"{best[col].iloc[0]:.3f}"
        tied = s[s[col].map(lambda x: f"{x:.3f}") == b3].action.tolist()
        if len(tied) > 1:
            TIES3.append((crit, col, b3, tied))
check([(c, k) for c, k, _, _ in TIES3] == [("gacc", "mean_regret"), ("fcm_obj", "max_regret")],
      "decision table: at three decimals the recommended value ties only on mean gACC regret and maximum OBJ regret")
check(all(sorted(t) == sorted(["SC-FFCM (L=5)", "SC-FFCM (L=1, R=50)"]) for _, _, _, t in TIES3),
      "decision table: both three-decimal ties are between SC-FFCM with L=5 and with L=1")
body = []
for a in ACT:
    cells = [ACT_TEX[a]]
    for crit in CRIT:
        r = DEC[(DEC.criterion == crit) & (DEC.action == a)].iloc[0]
        mr, xr = f"{r.mean_regret:.{DEC_DIG}f}", f"{r.max_regret:.{DEC_DIG}f}"
        cells += [rf"$\mathbf{{{mr}}}$" if r.rec_mean_regret else f"${mr}$",
                  rf"$\mathbf{{{xr}}}$" if r.rec_max_regret else f"${xr}$"]
    body.append(" & ".join(cells) + r" \\")
body.append(r"\midrule")
body.append(lab_row(11, r"\emph{Recommended action per seed block (in each pair of columns: min.\ mean regret, "
                         r"min.\ max regret)}") + r" \\")
for lab, df in ((r"Seeds $10$--$19$", D19), (r"Seeds $20$--$29$ (derived)", D2029)):
    cells = [lab]
    for crit in CRIT:
        bm, bx = recs(df, crit)
        cells += [bm, bx]
    body.append(" & ".join(cells) + r" \\")
pm = DEC[(DEC.criterion == "mean_acc") & (DEC.action == "pre_mass_off (L=1)")].iloc[0]
check(pm.argmax_state == "quantity_skew_extreme", "pre_mass_off (L=1): max regret on mean ACC at Quantity skew")
check(pm.rec_mean_regret == 1, "pre_mass_off (L=1) has the smallest mean regret on mean ACC (seeds 10-29)")
for crit in ("worst_acc", "gacc", "fcm_obj"):
    w = DEC[(DEC.criterion == crit) & (DEC.rec_max_regret == 1)].action.tolist()
    check(len(w) == 1 and w[0].startswith("SC-FFCM"), f"minimax-regret action on {crit} is SC-FFCM (seeds 10-29)")
up = DEC[(DEC.criterion == "upload_numbers") & (DEC.rec_mean_regret == 1) & (DEC.rec_max_regret == 1)].action.tolist()
check(up == ["GF-PFedFCM (L=5)"], "uploaded numbers: GF-PFedFCM is recommended under both rules")
head = [r" & \multicolumn{2}{c}{Mean client ACC} & \multicolumn{2}{c}{Worst client ACC} & \multicolumn{2}{c}{gACC} & "
        r"\multicolumn{2}{c}{OBJ (relative)} & \multicolumn{2}{c}{Uploads (relative)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"Action & mean & max & mean & max & mean & max & mean & max & mean & max \\"]
qs = NAME[pm.argmax_state]
CRIT_TX = {"mean_acc": "mean client ACC", "worst_acc": "worst client ACC", "gacc": "gACC", "fcm_obj": "OBJ",
           "upload_numbers": "uploads"}
tie_txt = " and on ".join(("the mean" if col == "mean_regret" else "the maximum") + f" {CRIT_TX[crit]} regret (${v}$)"
                          for crit, col, v, _ in TIES3)
cap = (r"Decision analysis (pre-registration addendum; Section~\ref{sec:discussion-guidance}): regret of seven candidate "
       r"rules (actions) under five criteria, with the 14 configurations as states. State value: mean over the $20$ "
       r"held-out seeds $10$--$29$ (no run substituted), of which seeds $10$--$19$ are a second look for the one-step and "
       r"SC-FFCM actions (Section~\ref{sec:protocol-stages}); regret: best value minus the action's value for the ACC-type "
       r"criteria, and (value$-$best)/best for OBJ and uploaded numbers, whose scale differs across configurations. "
       r"Uploads are counted per run and therefore also reflect the stopping rules: GF-PFedFCM and SC-FFCM stop once the "
       r"prototypes move by less than $10^{-5}$, the other actions run all $50$ rounds; per client and round the "
       r"mass-weighted actions upload $c(d+1)$ numbers and SC-FFCM $2cd$. "
       r"Mean: mean regret over the states (uniform prior); max: maximum regret. Bold: the recommended action, by the "
       r"smallest mean regret and by the smallest maximum regret (minimax regret). Regrets are printed to four decimals, "
       r"at which the smallest value of every column is unique; at three decimals SC-FFCM with $L=5$ and with $L=1$ "
       rf"would tie on {tie_txt}. SC-FFCM: steps calibrated on seed "
       r"index $100$, $R=50$; the other actions keep the constants of the earlier GF-PFedFCM paper. Bottom: the recommended "
       r"actions when each seed block is analysed alone (F5, GF5: F-FCM, GF-PFedFCM with $L=5$; PRE1: ungated PRE, "
       r"$L=1$; PM1, PM5: \texttt{pre\_mass\_off} with $L=1$, $5$; SC1, SC5: SC-FFCM with $L=1$, $5$); seeds "
       r"$10$--$19$ were analysed and frozen before the fresh block; the seeds $20$--$29$ row is derived with the same "
       r"frozen code after the fact. Per-configuration values: "
       + esm_ref("tab:eb-acc").replace("Table~", "Tables~") + r"--" + esm_ref("tab:eb-upload").split("Table~")[-1]
       + r", panel (a) for seeds $20$--$29$ and panel (b) for seeds $10$--$19$; the state value is the mean of the two "
         r"panels.")
# 2026-10-07 (compression): the decision table as pre-specified moves to Online Resource 1 (emitted below, after
# to_esm); the paper shows the post hoc analysis with a common stopping rule and calibrated steps in Figure 9.
DECISION_TEX = table("tab:decision", cap.replace("Decision analysis (pre-registration addendum;",
                                                 "Decision analysis as pre-specified (pre-registration addendum;"),
                     "@{}lcccccccccc@{}", head, body)
DEC_HEAD = head

# ---------------------------------------------------------------------------------------------- (4) tab:scffcm
SC_COLS = [  # (source, hyp, block, header group)
    ("T", "H6", "s00-09"), ("T", "H6", "s10-19"), ("T", "H6", "s20-29"),
    ("T", "H7", "s00-09"), ("T", "H7", "s10-19"), ("T", "H7", "s20-29"),
    ("T", "H8", "s00-09"), ("T", "H8", "s10-19"), ("T", "H8", "s20-29"),
    ("T", "S-L50", "s10-19"), ("T", "S-L50", "s20-29"),   # round 2 (P6): the seeds 10-19 column quoted in Section 7.4
    ("X", "CM-mass-SC5", "s10-19"), ("X", "CM-mass-SC5", "s20-29"),
    ("T", "ED-mass", "s10-19"), ("T", "ED-mass", "s20-29"),
    ("T", "S-PRE-SC1", "s20-29"), ("T", "S-PRE-SC5", "s20-29")]


def src_df(s):
    return TESTS if s == "T" else XT


def sc_cell(src, hyp, block, c):
    df = src_df(src)
    t = df[(df.block == block) & (df.hyp == hyp) & (df.metric == "mean_acc") & (df.dataset == c)]
    if t.empty:
        return "n.r."
    r = t.iloc[0]
    if r.status != "ok":
        return d3(0.0, r"\circ")
    return d3(r.mean_diff_raw, marks_of(r.p_exact, r.p_holm, r.n_eff, low_n=True))


def sc_counts(src, hyp, block, kind):
    df = src_df(src)
    if kind == "u":
        return cnt(*recount(df, block, hyp, "mean_acc", "p_exact"))
    if kind == "h":
        return cnt(*recount(df, block, hyp, "mean_acc", "p_holm"))
    if kind == "m":
        return cnt(*mrd(df, block, hyp, "mean_acc"))
    n_t, n_low = tested(df, block, hyp, "mean_acc")
    return f"{n_t} ({n_low})"


for src, hyp, block in SC_COLS:
    if src == "T":
        s = srow(block, hyp, "mean_acc")
        check(sc_counts(src, hyp, block, "u") == cnt(s.better_unadj, s.worse_unadj)
              and sc_counts(src, hyp, block, "h") == cnt(s.better_holm, s.worse_holm),
              f"tab:scffcm column {hyp} {block} equals analysis_final_summary.csv")
SL = r"$^{\S}$"
SC_A, SC_B = SC_COLS[:9], SC_COLS[9:]


def sc_panel(cols):
    n = 2 + len(cols)
    holm_lab = ("Holm-adjusted (14 configurations; 12 for $L=50$)" if any(h == "S-L50" for _, h, _ in cols)
                else "Holm-adjusted (14 configurations)")
    b = config_block(n, lambda c: [sc_cell(src, h, bl, c) for src, h, bl in cols])
    b.append(r"\midrule")
    for kind, lab in (("u", "Significantly positive/negative"), ("h", holm_lab),
                      ("m", r"Significant and $|\Delta|\ge0.005$"), ("t", r"Tested ($n'<10$)")):
        b.append(lab_row(2, lab) + " & " + " & ".join(sc_counts(s_, h, bl, kind) for s_, h, bl in cols) + r" \\")
    return b


body_a, body_b = sc_panel(SC_A), sc_panel(SC_B)
head_a = [r"\multicolumn{11}{@{}l}{\emph{(a) Communication-matched: $R=50$ rounds for both}} \\", r"\midrule",
          r" & & \multicolumn{3}{c}{H6: $L=1$ vs $L=1$} & \multicolumn{3}{c}{H7: $L=1$ vs $L=5$} & "
          r"\multicolumn{3}{c}{H8: $L=5$ vs $L=5$} \\",
          r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
          rf"Configuration & $m^*$ & $0$--$9$ & $10$--$19${SL} & $20$--$29$ & $0$--$9$ & $10$--$19${SL} & $20$--$29$ & "
          rf"$0$--$9$ & $10$--$19${SL} & $20$--$29$ \\"]
check(SC_B[0] == ("T", "S-L50", "s10-19") and SC_B[1] == ("T", "S-L50", "s20-29"),
      "tab:scffcm panel (b) starts with the L=50 columns of seeds 10-19 and 20-29")
_l50 = srow("s10-19", "S-L50", "mean_acc")
check(cnt(_l50.better_unadj, _l50.worse_unadj) == "4/2" and cnt(_l50.better_holm, _l50.worse_holm) == "0/1",
      "S-L50 on seeds 10-19 (second look): 4/2 (0/1), the count quoted in Section 7.4")
head_b = [r"\multicolumn{10}{@{}l}{\emph{(b) Released default $L=50$; compute-matched ($250$ local passes each); "
          r"ungated PRE}} \\",
          r"\midrule",
          r" & & \multicolumn{6}{c}{\texttt{pre\_mass\_off} vs SC-FFCM} & \multicolumn{2}{c}{Ungated PRE vs SC-FFCM} \\",
          r"\cmidrule(lr){3-8}\cmidrule(lr){9-10}",
          r" & & \multicolumn{2}{c}{$L=1$ vs $L=50$} & \multicolumn{2}{c}{$L{=}1$, $R{=}250$ vs $L{=}5$, $R{=}50$} & "
          r"\multicolumn{2}{c}{both $L{=}1$, $R{=}250$} & $L=1$ vs $L=1$ & $L=1$ vs $L=5$ \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}",
          rf"Configuration & $m^*$ & $10$--$19${SL} & $20$--$29$ & $10$--$19${SL} & $20$--$29$ & $10$--$19${SL} & $20$--$29$ "
          r"& $20$--$29$ & $20$--$29$ \\"]
fr = {h: srow("s20-29", h, "mean_acc") for h in ("S-PRE-SC1", "S-PRE-SC5", "ED-none")}
cm_none = xcount("s20-29", "CM-none-SC5", "mean_acc")
cap = (r"Mean client ACC of \texttt{pre\_mass\_off} minus that of SC-FFCM (last two columns of panel (b): the ungated PRE "
       r"rule with $L=1$ minus SC-FFCM, a secondary comparison declared before the fresh run); local steps of the two "
       r"methods in the headers. SC-FFCM steps are calibrated per configuration and $L$ on seed index $100$ "
       rf"({paper_ref('sec:protocol-baselines', 'Sections').replace(' of the paper', '')} and "
       rf"{MAINLAB.get('sec:protocol-additional', '??')} of the paper); \texttt{{pre\_mass\_off}} keeps the earlier GF-PFedFCM paper's "
       r"constants. Seeds $20$--$29$: H6--H8 are pre-registered primary hypotheses (all replicate; "
       rf"{paper_ref('tab:confirm', 'Table')}), $L=1$ vs $L=50$ and the ungated PRE columns secondary ones; $^{{\S}}$: "
       r"second look. Compute-matched columns are exploratory; rounds and uploads are not matched. $^{\circ}$: "
       r"$n'\le5$, so $p<0.05$ is unattainable. $p$, Holm $p$, $n'$: Tables~\ref{tab:scffcm-p-a}--\ref{tab:scffcm-p-b}; "
       r"other criteria: Table~\ref{tab:ed-counts}. " + MARKS_TXT + ". " + stats_esm() + ".")
if "sec:protocol-additional" not in MAINLAB:
    WARN.append("main.aux has no label sec:protocol-additional")
tex = "\n".join([
    r"\begin{table}[htbp]", r"\centering", r"\tablefontsize", rf"\caption{{{cap}}}", r"\label{tab:scffcm}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
    r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule", *head_a, r"\midrule", *body_a, r"\bottomrule",
    r"\end{tabular}%", r"}\end{tabular}\par\vspace{2pt}",
    r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
    r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule", *head_b, r"\midrule", *body_b, r"\bottomrule",
    r"\end{tabular}%", r"}\end{tabular}", r"\end{table}", ""])
emit("tabS_scffcm", tex, "tab:scffcm",
     "Online Resource 1, Section S14 (moved from the paper's Appendix A in the fix stage), cited from Sections 7 and 8",
     "REPLACES the inline tab:scffcm of src/11_appendixA_tables.tex. The old ungated-PRE columns move to the caption "
     "(fresh-block counts) and to Online Resource 1 (tab:ed-counts); the old 'each at its better L' columns are the H7 "
     "columns here.",
     "pre_mass_off minus SC-FFCM per configuration: H6/H7/H8 on seeds 0-9, 10-19 (second look), 20-29; L=1 vs L=50 "
     "(10-19 second look, 20-29); compute-matched L=1 R=250 vs SC L=5 R=50 and both at R=250 (10-19, 20-29); ungated PRE (L=1) vs SC-FFCM "
     "L=1 and L=5 (fresh, secondary); rows unadjusted, Holm, minimum reported difference, tested (n'<10).")

# ---------------------------------------------------------------------------------------------- (5) tab:baselines
t6 = pd.read_csv(T6)
BT = dict(TAB)
for _, r in t6.iterrows():
    if r.method in ("fednova", "stallmann_avg2"):
        BT[(f"B:{r.method}", r.dataset, int(r.seed))] = {"mean_acc": str(r.mean_acc)}
for tag in ("cfcm_s00-19", "FRESH_cfcm_s20-29_descriptive"):
    for r in csv.DictReader((T10 / f"{tag}.csv").open()):
        BT[("CFCM", r["dataset"], int(r["seed"]))] = r
for L, df in PEDRUN.items():
    for _, r in df.iterrows():
        BT[(f"PED:L{L}", r.dataset, int(r.seed))] = {"mean_acc": str(r.mean_acc)}
# Pedrycz: the L with the highest mean client ACC over the configurations with a calibrated step at every L
cal_all = [c for c in CONFIGS if all(np.isfinite(num(PEDCAL[L].set_index("dataset").loc[c, "alpha"])) for L in PEDCAL)]
ped_mean = {L: float(np.nanmean([num(BT[(f"PED:L{L}", c, s)]["mean_acc"]) for c in cal_all for s in range(20)]))
            for L in PEDCAL}
L_PED = max(ped_mean, key=ped_mean.get)
check(L_PED == 1, "Pedrycz's rule has its highest mean client ACC at L=1")
ped_runs = pd.read_csv(RES / "extra_C_pedrycz_runs.csv").set_index("L")
check(all(abs(ped_mean[L] - ped_runs.loc[L, "mean_acc_11"]) < 1e-12 for L in PEDCAL),
      "Pedrycz mean ACC per L equals extra_C_pedrycz_runs.csv")


def bmean(arm, c, block):
    v = [num(BT[(arm, c, s)]["mean_acc"]) for s in BLOCKS[block] if (arm, c, s) in BT]
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return "n.s." if arm.startswith("PED") else "n.r."
    return v3(np.mean(v))


PA = [("L1:pre_mass_off", "s00-09"), ("L5:pre_mass_off", "s00-09"), ("B:fednova", "s00-09"),
      ("B:stallmann_avg2", "s00-09"), ("SC:L5", "s00-09"), ("SC:L1", "s00-09"), (f"PED:L{L_PED}", "s00-09"),
      ("CFCM", "s00-09")]
PB = [("L1:pre_mass_off", "s10-19"), ("L5:pre_mass_off", "s10-19"), ("SC:L5", "s10-19"), ("SC:L1", "s10-19"),
      (f"PED:L{L_PED}", "s10-19"), ("CFCM", "s10-19"),
      ("L1:pre_mass_off", "s20-29"), ("L5:pre_mass_off", "s20-29"), ("SC:L5", "s20-29"), ("SC:L1", "s20-29"),
      ("CFCM", "s20-29")]
# counts: pre_mass_off at the column's L against the column's method
bl_h = [dict(id="BL-SW", kind="pair", a="L5:pre_mass_off", b="B:fednova", expect=None),
        dict(id="BL-AVG2", kind="pair", a="L5:pre_mass_off", b="B:stallmann_avg2", expect=None)]
blt = pd.DataFrame(R.run_tests(BT, bl_h, "s00-09", BLOCKS["s00-09"], "mean_acc"))
check((blt.n == 10).all(), "size-weighted averaging and avg2 pair with pre_mass_off on all 10 seeds, 14 configurations")
# Round 2 (P7): pre_mass_off (L=1) against C-FCM, the counts quoted in Section 7.5 (frozen test, Holm over 14). On
# seeds 20-29 C-FCM is a descriptive reference, outside the pre-registered analysis, and is not tested.
bl_cf = [dict(id="BL-CFCM", kind="pair", a="L1:pre_mass_off", b="CFCM", expect=None)]
blc = pd.DataFrame([r for b in ("s00-09", "s10-19") for r in R.run_tests(BT, bl_cf, b, BLOCKS[b], "mean_acc")])
check((blc.n == 10).all(), "C-FCM pairs with pre_mass_off (L=1) on all ten seeds of both blocks, 14 configurations")
_cf_txt = {b: f"{cnt(*recount(blc, b, 'BL-CFCM', 'mean_acc', 'p_exact'))} "
              f"({cnt(*recount(blc, b, 'BL-CFCM', 'mean_acc', 'p_holm'))})" for b in ("s00-09", "s10-19")}
check(_cf_txt == {"s00-09": "4/2 (2/1)", "s10-19": "5/1 (1/1)"},
      "pre_mass_off (L=1) vs C-FCM: 4/2 (2/1) on seeds 0-9 and 5/1 (1/1) on seeds 10-19, as quoted in Section 7.5")
_cf_holm = {b: sorted((r.dataset, r.direction) for r in blc[(blc.block == b) & (blc.p_holm < ALPHA)].itertuples())
            for b in ("s00-09", "s10-19")}
check(_cf_holm == {"s00-09": [("mnist784_pca32 (20c)", "a"), ("mnist784_pca32 (50c)", "a"), ("satimage", "b")],
                   "s10-19": [("mnist784_pca32 (50c)", "a"), ("satimage", "b")]},
      "pre_mass_off (L=1) vs C-FCM: Holm-significant gains only on MNIST, the loss on Satimage (Section 7.5)")


def bl_count(arm, block):
    if arm == "B:fednova":
        return f"{cnt(*recount(blt, block, 'BL-SW', 'mean_acc', 'p_exact'))} " \
               f"({cnt(*recount(blt, block, 'BL-SW', 'mean_acc', 'p_holm'))})"
    if arm == "B:stallmann_avg2":
        return f"{cnt(*recount(blt, block, 'BL-AVG2', 'mean_acc', 'p_exact'))} " \
               f"({cnt(*recount(blt, block, 'BL-AVG2', 'mean_acc', 'p_holm'))})"
    if arm == "SC:L5":
        return ucount(block, "H8", "mean_acc") + (SL if block == "s10-19" else "")
    if arm == "SC:L1":
        return ucount(block, "H6", "mean_acc") + (SL if block == "s10-19" else "")
    if arm.startswith("PED"):
        return xc(block, f"PED{L_PED}-PM", "mean_acc") + (SL if block == "s10-19" and L_PED == 1 else "")
    if arm == "CFCM" and block in _cf_txt:
        return _cf_txt[block] + (SL if block == "s10-19" else "")
    return "--"


def panel(cols, ncol_lab):
    b = config_block(ncol_lab + len(cols) - 1 + 1, lambda c: [bmean(a, c, blk) for a, blk in cols])
    b.append(r"\midrule")
    b.append(lab_row(2, r"\texttt{pre\_mass\_off} vs column") + " & " +
             " & ".join(bl_count(a, blk) for a, blk in cols) + r" \\")
    return b


METHOD = {"L1:pre_mass_off": r"\texttt{pre\_mass\_off}", "L5:pre_mass_off": r"\texttt{pre\_mass\_off}",
          "B:fednova": r"Size-wt.\ avg.", "B:stallmann_avg2": "avg2", "SC:L5": "SC-FFCM", "SC:L1": "SC-FFCM",
          f"PED:L{L_PED}": "Pedrycz", "CFCM": "C-FCM"}
LTXT = {"L1:pre_mass_off": "1", "L5:pre_mass_off": "5", "B:fednova": "5", "B:stallmann_avg2": "5", "SC:L5": "5",
        "SC:L1": "1", f"PED:L{L_PED}": str(L_PED), "CFCM": ""}


def method_head(cols, first_col=3):
    """Two header rows: method names (merged over adjacent columns of the same method) and L."""
    cells, rules, i = [], [], 0
    while i < len(cols):
        j = i
        while j + 1 < len(cols) and METHOD[cols[j + 1][0]] == METHOD[cols[i][0]] and cols[j + 1][1] == cols[i][1]:
            j += 1
        k = j - i + 1
        cells.append(mc(k, METHOD[cols[i][0]]) if k > 1 else METHOD[cols[i][0]])
        if k > 1:
            rules.append(rf"\cmidrule(lr){{{first_col + i}-{first_col + j}}}")
        i = j + 1
    row1 = " & & " + " & ".join(cells) + r" \\"
    row2 = "Configuration & $m^*$ & " + " & ".join(rf"$L{{=}}{LTXT[a]}$" if LTXT[a] else "" for a, _ in cols) + r" \\"
    return [row1, "".join(rules), row2] if rules else [row1, row2]


body_a = panel(PA, 2)
body_b = panel(PB, 2)
head_a = [r"\multicolumn{10}{@{}l}{\emph{(a) Exploratory seeds $0$--$9$}} \\", r"\midrule"] + method_head(PA)
head_b = [r" & & \multicolumn{6}{c}{Seeds $10$--$19$} & \multicolumn{5}{c}{Seeds $20$--$29$ (fresh)} \\",
          r"\cmidrule(lr){3-8}\cmidrule(lr){9-13}"] + method_head(PB)
_pc = PEDCAL[L_PED].set_index("dataset")
no_alpha = [NAME[c] for c in CONFIGS if not np.isfinite(num(_pc.loc[c, "alpha"]))]
cap = (r"Mean client ACC of \texttt{pre\_mass\_off} ($L=1$ and $L=5$) and of the other families and C-FCM, $R=50$. "
       r"Size-wt.\ avg.: size-weighted averaging of the local centres, whose construction is adapted from FedNova's "
       rf"normalized averaging ({paper_ref('sec:design-families')}), server step $\eta_s=1.0$, the exact size-weighted "
       r"average, fixed as canonical since all five steps checked on an IID scenario tied with centralized FCM on the "
       r"label-using criterion (Section~\ref{app:calibration}); it and avg2 "
       r"were run with $L=5$ on seeds $0$--$9$ only. SC-FFCM and Pedrycz's gradient rule have step sizes calibrated per "
       r"configuration and $L$ on seed index $100$; Pedrycz's rule is shown at "
       rf"$L={L_PED}$, the value of $L\in\{{1,5,10\}}$ with the highest mean client ACC over the configurations with a stable step at "
       r"every $L$ and seeds $0$--$19$ (Table~\ref{tab:ee-pedrycz}), was run on seeds $0$--$19$ only (exploratory), and has no "
       rf"stable step on {', '.join(no_alpha)} (n.s.). C-FCM: centralized FCM at $m^*$ from the same prototypes, at "
       r"most $150$ iterations, a reference (descriptive on seeds $20$--$29$, not pre-registered). Last row of each "
       r"panel: configurations on which \texttt{pre\_mass\_off} at the column's $L$ (for C-FCM: $L=1$) is significantly "
       r"better/worse than the column's method, unadjusted (Holm over 14 configurations); against SC-FFCM these are H8 "
       r"($L=5$) and H6 ($L=1$); against C-FCM they are not tested on seeds $20$--$29$ (--), where C-FCM is a "
       r"descriptive reference; $^{\S}$: second look. The two \texttt{pre\_mass\_off} columns differ in $L$, a deployment comparison, "
       r"not a mass-point or gate effect. Normalized membership entropy $H/\ln c$ next to these accuracies: "
       r"Table~\ref{tab:eb-gacc} for \texttt{pre\_mass\_off}, SC-FFCM and C-FCM on the fresh block; it was not recorded "
       r"for size-weighted averaging and avg2. " + stats_esm() + ".")
tex = "\n".join([
    r"\begin{table}[htbp]", r"\centering", r"\tablefontsize", rf"\caption{{{cap}}}", r"\label{tab:baselines}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
    r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule", *head_a, r"\midrule", *body_a, r"\bottomrule",
    r"\end{tabular}%", r"}\end{tabular}", "", r"\medskip", "",
    r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
    r"\begin{tabular}{@{}lcccccccccccc@{}}", r"\toprule",
    r"\multicolumn{13}{@{}l}{\emph{(b) Held-out seeds: $10$--$19$ (second look for $L=1$ and SC-FFCM) and the "
    r"fresh block $20$--$29$}} \\", r"\midrule", *head_b, r"\midrule", *body_b, r"\bottomrule",
    r"\end{tabular}%", r"}\end{tabular}", r"\end{table}", ""])
emit("tabS_baselines", tex, "tab:baselines", "Online Resource 1, Section S14 (moved from the paper's Appendix A in the "
     "fix stage), cited from Sections 7.4 and 7.5",
     "REPLACES the inline tab:baselines of src/11_appendixA_tables.tex (best cell selected on the same seeds; "
     "'FedNova' column). The old SC-FFCM-minus-best-cell column is dropped (H8 counts are in the last row).",
     "Panel (a) seeds 0-9: pre_mass_off L=1/L=5, size-weighted averaging, avg2, SC-FFCM L=5/L=1, Pedrycz at L=1 "
     "(its best calibrated L), C-FCM; panel (b) seeds 10-19 and 20-29 for the methods run there; last rows: "
     "pre_mass_off vs column, unadjusted (Holm).")

# ---------------------------------------------------------------------------------------------- (6) tab:audit
INITS = ["paper", "kmeanspp_best10", "oracle", "fed_lossless_pre1"]
INIT_TX = {"paper": "paper", "kmeanspp_best10": r"$k$-means++", "oracle": "class means",
           "fed_lossless_pre1": "federated"}
INIT_AB = {"paper": "P", "kmeanspp_best10": "K", "oracle": "C", "fed_lossless_pre1": "F"}
ms = MSTAR_T.set_index(["dataset", "init"])
cmap = AUD2[AUD2.init == "paper"].set_index("dataset")["c"]


def mstar_cells(c):
    out = [str(int(cmap[c])), f"{MSTAR[c]:.1f}"]
    for i in INITS:
        v = float(ms.loc[(c, i), "mstar"])
        out.append(f"{v:.1f}" if v == MSTAR[c] else rf"$\mathbf{{{v:.1f}}}$")
    return out


dis = [(c, i) for c in CONFIGS for i in INITS if float(ms.loc[(c, i), "mstar"]) != MSTAR[c]]
check(dis == [("letter", "oracle")], "the only m* disagreement is Letter under the class-mean (oracle) init")
fed = AUD[AUD.init == "fed_lossless_pre1"]
max_gap = float(fed.maxgap_fed_vs_cen_iter.max())
eq_cen = int((fed.distinct == fed.distinct_cen_iter).sum())
paper = AUD[AUD.init == "paper"].set_index(["dataset", "m", "seed"])
eq_pap = int(sum(int(r.distinct) == int(paper.loc[(r.dataset, r.m, r.seed), "distinct"]) for r in fed.itertuples()))
check(len(fed) == eq_cen == eq_pap, "federated audit: distinct prototypes equal to the centralized iteration and to the "
                                     "paper-init FCM on every run")
a2 = AUD2.set_index(["dataset", "init"])
kpp = AUD[(AUD.init == "kmeanspp_best10") & (AUD.m == 2.0)].groupby("dataset")


def m2_cells(c):
    out = [str(int(cmap[c])), f"{MSTAR[c]:.1f}"]
    for i in INITS:
        r = a2.loc[(c, i)]
        rng = f"{int(r.distinct_min)}" if r.distinct_min == r.distinct_max else f"{int(r.distinct_min)}--{int(r.distinct_max)}"
        out.append(f"{r.distinct_mean:.1f} ({rng})")
    for i in INITS:
        out.append(v3(a2.loc[(c, i), "hnorm_mean"]))
    g = kpp.get_group(c)
    lo, hi = int(g.kpp_distinct_min.min()), int(g.kpp_distinct_max.max())
    out.append(f"{lo}" if lo == hi else f"{lo}--{hi}")
    return out


for c in RESCUED:
    check(all(a2.loc[(c, i), "distinct_mean"] < a2.loc[(c, i), "c"] for i in INITS),
          f"{c}: at m=2 fewer than c distinct prototypes under every init")
lt = AUD[(AUD.dataset == "letter") & (AUD.m == 1.5)]
lt_or = lt[lt.init == "oracle"].sort_values("seed").distinct.astype(int).tolist()
lt_pa = lt[lt.init == "paper"].sort_values("seed").distinct.astype(int).tolist()
check(all(v == int(cmap["letter"]) for v in lt_or), "Letter, m=1.5, class-mean init: all prototypes distinct")
n_seeds_aud = int(AUD.seed.nunique())
body_a = config_block(7, lambda c: mstar_cells(c)[:1] + mstar_cells(c)[2:], mstar=True)
head_a = [r"\multicolumn{7}{@{}l}{\emph{(a) $m^*$ by initialization of the audit}} \\", r"\midrule",
          r" & & & \multicolumn{4}{c}{Initialization} \\", r"\cmidrule(lr){4-7}",
          r"Configuration & $m^*$ & $c$ & " + " & ".join(INIT_TX[i] for i in INITS) + r" \\"]
body_b = config_block(12, lambda c: m2_cells(c)[:1] + m2_cells(c)[2:], configs=RESCUED)
head_b = [r"\multicolumn{12}{@{}l}{\emph{(b) At $m=2$: distinct prototypes and $H/\ln c$ on the six configurations "
          r"with $m^*<2$}} \\", r"\midrule",
          r" & & & \multicolumn{4}{c}{Distinct prototypes, mean (min--max)} & \multicolumn{4}{c}{$H/\ln c$} & "
          r"$k$-means++ \\", r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
          r"Configuration & $m^*$ & $c$ & " + " & ".join(INIT_AB[i] for i in INITS) + " & " +
          " & ".join(INIT_AB[i] for i in INITS) + r" & restarts \\"]
cap = (rf"The fuzzifier audit under other initializations and in federated execution ({paper_ref('sec:pitfall-rule')}; "
       rf"exploratory, seeds $0$--${n_seeds_aud - 1}$, run on the delegated workstation ws67, "
       rf"{paper_ref('sec:protocol-compute')}). Initializations: "
       r"paper (P), $c$ random pooled data points (as in all runs); $k$-means++ (K), the best of ten $k$-means++ restarts "
       r"by the FCM objective; class means (C), an oracle that uses the labels; federated (F), the unweighted, undamped "
       rf"PRE one-step exchange run from the paper initialization. (a) $m^*$ by rule ({MAINLAB.get('eq:mstar', '??')}) of the paper "
       r"under each initialization; bold: differs from the $m^*$ used in the paper. The only difference is Letter under "
       rf"class-mean initialization, which keeps all {int(cmap['letter'])} prototypes distinct at $m=1.5$ on every seed "
       rf"({', '.join(map(str, lt_or))}; paper initialization {', '.join(map(str, lt_pa))}). (b) Centralized FCM at "
       r"$m=2$ collapses on the six configurations under every initialization; last column: range of distinct "
       r"prototypes over the ten $k$-means++ restarts. The federated execution reproduces the centralized iteration: "
       rf"the largest prototype deviation over all {len(fed)} runs and iterations is {sci(max_gap)}, and the number of "
       rf"distinct prototypes agrees on {eq_cen} of {len(fed)} runs. These $m=2$ runs audit the centralized FCM "
       r"objective; they are not evidence about aggregation.")
tex = "\n".join([
    r"\begin{table}[htbp]", r"\centering", r"\tablefontsize", rf"\caption{{{cap}}}", r"\label{tab:audit}",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
    r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", *head_a, r"\midrule", *body_a, r"\bottomrule",
    r"\end{tabular}%", r"}\end{tabular}", "", r"\medskip", "",
    r"\begin{tabular}{@{}c@{}}\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
    r"\begin{tabular}{@{}lccccccccccc@{}}", r"\toprule", *head_b, r"\midrule", *body_b, r"\bottomrule",
    r"\end{tabular}%", r"}\end{tabular}", r"\end{table}", ""])
emit("tabS_audit", tex, "tab:audit", "Online Resource 1, Section S18 (moved from Section 3.4 of the paper in the fix "
     "stage)", "none (new)",
     "E-F: (a) m* for 14 configurations under paper, k-means++ best-of-10, class-mean and federated execution; "
     "(b) distinct prototypes and H/ln c at m=2 per init on the six rescued configurations, k-means++ restart range.")

# ================================================================================================ ONLINE RESOURCE 1
ESM_ACT = [a for a, _ in R.DECISION_ACTIONS] + ["C-FCM (descriptive)"]
ESM_ACT_TX = {**{a: ACT_AB[a] for a in ACT}, "C-FCM (descriptive)": "C-FCM"}
# Round 2 (P3): Tables S13-S16 carry both held-out blocks, (a) the fresh block and (b) seeds 10-19, so that Table 7
# (state value = mean over seeds 10-29 = mean of the two block means, checked below) can be recomputed from them.
EB_BLOCKS = ("s20-29", "s10-19")
ev = {b: EBV[EBV.block == b].set_index(["dataset", "action"]) for b in EB_BLOCKS}
EB_PANEL = {"s20-29": r"(a) Fresh block, seeds $20$--$29$",
            "s10-19": r"(b) Seeds $10$--$19$ (a second look for the one-step and SC-FFCM actions)"}
ACT_LEGEND = (r"F5, GF5: F-FCM and GF-PFedFCM with $L=5$; PRE1: ungated PRE, $L=1$; PM1, PM5: \texttt{pre\_mass\_off} "
              r"with $L=1$, $5$; SC5, SC1: SC-FFCM with $L=5$, $1$ ($R=50$, calibrated steps); C-FCM: centralized FCM, "
              r"a reference and not an action (descriptive on seeds $20$--$29$, not pre-registered there)")
# state values of the decision analysis (seeds 10-29) = mean of the two block means
_dv = pd.read_csv(T10 / "analysis_final_decision_values.csv")
_e2 = EBV[EBV.action != "C-FCM (descriptive)"]
check(set(_e2.n_seeds) == {10} and len(_e2) == 2 * 14 * 7, "E-B values: ten seeds per block for every action")
_dev = 0.0
for _crit in CRIT:
    _v = _dv[_dv.criterion == _crit].set_index(["dataset", "action"])["value"]
    _p = _e2.pivot_table(index=["dataset", "action"], columns="block", values=_crit)
    _j = pd.concat([_v, (_p["s10-19"] + _p["s20-29"]) / 2], axis=1, join="inner")
    check(len(_j) == 98, f"decision state values of {_crit}: 98 (configuration, action) pairs")
    _dev = max(_dev, float(((_j.iloc[:, 0] - _j.iloc[:, 1]).abs() / _j.iloc[:, 0].abs().clip(lower=1.0)).max()))
check(_dev < 1e-12, "Table 7 state values (seeds 10-29) equal the mean of the two block means of Tables S13-S16")


def eb_table(label, what, f1, f2, fmt1, fmt2, extra=""):
    body = []
    for i, b in enumerate(EB_BLOCKS):
        if i:
            body.append(r"\midrule")
        body.append(lab_row(10, rf"\emph{{{EB_PANEL[b]}}}") + r" \\")
        body.append(r"\midrule")
        body += config_block(10, lambda c, b=b: [f"{fmt1(ev[b].loc[(c, a), f1])}/{fmt2(ev[b].loc[(c, a), f2])}"
                                                 for a in ESM_ACT])
    head = [r"Configuration & $m^*$ & " + " & ".join(ESM_ACT_TX[a] for a in ESM_ACT) + r" \\"]
    # 2026-10-07: the decision table moved into Online Resource 1 (tab:decision; post hoc variants tab:decision-U/-C).
    cap = (f"{what} of the decision actions per configuration, (a) on the fresh block (seeds $20$--$29$) and (b) on "
           rf"seeds $10$--$19$; means over the ten seeds of each block. {ACT_LEGEND}. {extra}Table~\ref{{tab:decision}} "
           r"takes the mean of the two panels (twenty seeds) as the state value of each criterion it uses, so its regrets "
           r"can be recomputed from these tables up to rounding.")
    return table(label, cap, "@{}lccccccccc@{}", head, body)


# Round 2 (P9): the Xie-Beni sentence of tab:eb-obj, from the data (the old caption claimed values of order 1e3 and above
# in a table whose largest entry was 173.401).
_xb = {b: EBV[EBV.block == b].sort_values("xb", ascending=False) for b in EB_BLOCKS}
_big = EBV[EBV.xb >= 1000]
check(len(_big) == 1 and _big.iloc[0].block == "s10-19", "one Xie-Beni mean >= 1e3 in the two panels, on seeds 10-19")
_bt = _big.iloc[0]
_merged = [(s_, num(TAB[(_bt.arm, _bt.dataset, s_)]["min_dist"])) for s_ in BLOCKS["s10-19"]
           if num(TAB[(_bt.arm, _bt.dataset, s_)]["min_dist"]) < 1e-3]
check(len(_merged) == 1, "the Xie-Beni mean >= 1e3 comes from one run whose prototypes merged (distance < 1e-3)")
_top20 = _xb["s20-29"].iloc[0]
check(num(_top20.xb) < 1000, "the fresh block has no Xie-Beni mean >= 1e3")
XB_TXT = (rf"Xie--Beni means of order $10^{{3}}$ and above occur only in panel (b): {ESM_ACT_TX[_bt.action]} on "
          rf"{NAME[_bt.dataset]}, {sci(_bt.xb)}, where one run (seed {_merged[0][0]}) merged two prototypes (minimum "
          rf"distance {sci(_merged[0][1])}, below the collapse threshold $10^{{-3}}$); the largest mean in panel (a) is "
          rf"${v3(_top20.xb)}$ ({ESM_ACT_TX[_top20.action]} on {NAME[_top20.dataset]}). ")


def upl(x):
    x = num(x)
    return "--" if not math.isfinite(x) else f"{x:.0f}"


def rnd(x):
    x = num(x)
    return "--" if not math.isfinite(x) else f"{x:.1f}"


check(all(len(ev[b].loc[(c, a)]) > 0 for b in EB_BLOCKS for c in CONFIGS for a in ESM_ACT),
      "E-B values exist for every block/action/configuration")
check(all(set(EBV[EBV.block == b].n_seeds) == {10} for b in EB_BLOCKS), "E-B values of both blocks average ten seeds")
eb = "\n".join([
    eb_table("tab:eb-acc", "Mean/worst client ACC", "mean_acc", "worst_acc", v3, v3),
    eb_table("tab:eb-gacc", r"Global-alignment accuracy gACC/normalized membership entropy $H/\ln c$", "gacc", "hnorm",
             v3, v3),
    eb_table("tab:eb-obj", r"FCM objective OBJ \eqref{eq:protocol-obj}/Xie--Beni index (both lower is better)",
             "fcm_obj", "xb", v3, big, extra=XB_TXT),
    eb_table("tab:eb-upload", "Uploaded numbers per run/rounds executed", "upload_numbers", "n_rounds", upl, rnd,
             extra=r"C-FCM: iterations, no upload. GF-PFedFCM and SC-FFCM stop early; the other cells run all $50$ "
                   r"rounds. ")]).replace(r"\eqref{eq:protocol-obj}", "(" + MAINLAB.get("eq:protocol-obj", "??") + ")")
emit("tabS_eb", eb, "tab:eb-acc, tab:eb-gacc, tab:eb-obj, tab:eb-upload", "Online Resource 1 (new section on E-B)",
     "none (new)", "E-B on both held-out blocks, (a) seeds 20-29 and (b) seeds 10-19: 8 actions (7 decision actions + "
                   "C-FCM) x 14 configurations; four floats: mean/worst ACC, gACC/H over ln c, OBJ/XB, uploads/rounds.")

# ---------------------------------------------------------------------------------------------- E-C split sample
EC_M = [("test_fcm_obj", "OBJ"), ("test_gacc_trmap", "gACC"), ("test_mean_acc_trmap", "mean ACC"),
        ("test_worst_acc_trmap", "worst ACC")]
ECH = [("EC-GF", "GF-PFedFCM$-$\\texttt{post\\_footprint\\_off}"), ("EC-PREMASS", "\\texttt{pre\\_mass\\_on}$-$"
                                                                              "\\texttt{pre\\_mass\\_off}")]
unrel = set(EC[EC.test_acc_unreliable == 1].dataset)
check(unrel == {"wine", "quantity_skew_extreme"}, "held-out ACC flagged unreliable exactly on Wine and Quantity skew")


def ec_cell(h, m, c):
    r = EC[(EC.hyp == h) & (EC.metric == m) & (EC.dataset == c)].iloc[0]
    if r.status != "ok":
        return d3(0.0) + r"\,(0)"
    return d3(r.mean_diff_raw, marks_of(r.p_exact, r.p_holm)) + neff_note(r.n_eff, 20)


def ec_name_cells(c):
    return [ec_cell(h, m, c) for h, _ in ECH for m, _ in EC_M]


body = []
for title, members in GROUPS:
    if body:
        body.append(r"\midrule")
    body.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
    for c in members:
        nm = NAME[c] + (r"$^{\ddagger}$" if c in unrel else "")
        body.append(" & ".join([nm, f"{MSTAR[c]:.1f}"] + ec_name_cells(c)) + r" \\")
body.append(r"\midrule")
for kind, lab in (("u", "Personalization better/worse"), ("h", "Holm-adjusted (14 configurations)")):
    cells = []
    for h, _ in ECH:
        for m, _ in EC_M:
            s = srow("s00-19", h, m)
            ucount("s00-19", h, m)  # recount check
            cells.append(cnt(s.better_unadj, s.worse_unadj) if kind == "u" else cnt(s.better_holm, s.worse_holm))
    body.append(lab_row(2, lab) + " & " + " & ".join(cells) + r" \\")
head = [r" & & \multicolumn{4}{c}{" + ECH[0][1] + r"} & \multicolumn{4}{c}{" + ECH[1][1] + r"} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"Configuration & $m^*$ & " + " & ".join(t for _ in ECH for _, t in EC_M) + r" \\"]
cap = (r"Split-sample check of personalization (exploratory; $L=5$, seeds $0$--$19$, twenty paired seeds): each "
       r"client's data split $80/20$, the federated run on the training parts, the criteria on the held-out $20\%$ "
       r"(" + paper_ref("sec:protocol-metrics") + r"). Entries: mean over seeds of "
       r"personalization on minus off. OBJ: held-out FCM objective (negative favours personalization); gACC and "
       r"mean/worst client ACC with label maps fitted on training predictions only. $^{\ddagger}$: some clients have at "
       r"most five test points, so held-out ACC is unreliable there. In parentheses: $n'$ where below $20$. Counts: "
       r"configurations on which personalization is significantly better/worse (OBJ: lower/higher). " + MARKS_TXT +
       ". " + stats_esm() + ".")
emit("tabS_ec", table("tab:ec-split", cap, "@{}lccccccccc@{}", head, body), "tab:ec-split",
     "Online Resource 1 (E-C)", "none (new)",
     "E-C: per configuration, on-minus-off for GF-PFedFCM and pre_mass_on/off on held-out OBJ, gACC, mean ACC, worst "
     "ACC (train-fitted label maps), marks and n'; unreliable configurations flagged; count rows unadjusted/Holm.")

# ---------------------------------------------------------------------------------------------- E-D
cal = EDCAL.set_index("dataset")


def edge(r):
    parts = []
    if r.edge_eta_l != "interior":
        parts.append(rf"$\eta_l$ {r.edge_eta_l}")
    if r.edge_eta_g != "interior":
        parts.append(rf"$\eta_g$ {r.edge_eta_g}")
    return ", ".join(parts) if parts else "interior"


def fnum(x):
    x = num(x)
    return f"{x:g}"


body = config_block(7, lambda c: [fnum(cal.loc[c, "eta_l"]), fnum(cal.loc[c, "eta_g"]),
                                  sci(cal.loc[c, "rel_obj_gap"]), f"{int(cal.loc[c, 'n_stable'])}/"
                                                                  f"{int(cal.loc[c, 'n_grid'])}", edge(cal.loc[c])])
n_edge = int((cal.on_edge == 1).sum())
head = [r"Configuration & $m^*$ & $\eta_l$ & $\eta_g$ & Gap & Stable & Edge \\"]
ngrid = int(cal.n_grid.iloc[0])
cap = (r"Calibration of SC-FFCM with $L=1$ and $R=250$ rounds for the compute-matched comparison (seed index $100$, "
       rf"the rule of {paper_ref('sec:protocol-baselines')} on the extended grid of {ngrid} settings, $\eta_l\le16$, "
       r"$\eta_g\le32$). Gap: relative objective gap $|J-J_c|/J_c$ of the chosen setting; Stable: stable settings of the "
       rf"grid; Edge: the chosen setting lies on the low or high edge of the grid, which holds on {n_edge} of 14 "
       r"configurations. Every configuration has a stable setting.")
ed_cal = table("tab:ed-cal", cap, "@{}lcccccc@{}", head, body)

ED_ROWS = [("T", "ED-mass", r"\texttt{pre\_mass\_off} vs SC-FFCM, both $L=1$, $R=250$"),
           ("T", "ED-none", r"PRE vs SC-FFCM, both $L=1$, $R=250$"),
           ("X", "CM-mass-SC5", r"\texttt{pre\_mass\_off} ($L{=}1$, $R{=}250$) vs SC-FFCM ($L{=}5$, $R{=}50$)"),
           ("X", "CM-none-SC5", r"PRE ($L{=}1$, $R{=}250$) vs SC-FFCM ($L{=}5$, $R{=}50$)"),
           ("X", "CM-mass-PM5", r"\texttt{pre\_mass\_off} ($L{=}1$, $R{=}250$) vs \texttt{pre\_mass\_off} ($L{=}5$, "
                                r"$R{=}50$)"),
           ("X", "CM-none-PM5", r"PRE ($L{=}1$, $R{=}250$) vs \texttt{pre\_mass\_off} ($L{=}5$, $R{=}50$)"),
           ("X", "CM-SC1R250-SC5", r"SC-FFCM ($L{=}1$, $R{=}250$) vs SC-FFCM ($L{=}5$, $R{=}50$)")]
ED_MET = ["mean_acc", "worst_acc", "gacc", "fcm_obj"]


def any_count(src, hyp, block, metric):
    if src == "T":
        return ucount(block, hyp, metric) or "n.r."
    return xc(block, hyp, metric)


body = []
for src, hyp, lab in ED_ROWS:
    body.append(" & ".join([lab] + [any_count(src, hyp, b, m) for m in ED_MET for b in ("s10-19", "s20-29")]) + r" \\")
head = [r" & \multicolumn{2}{c}{Mean client ACC} & \multicolumn{2}{c}{Worst client ACC} & \multicolumn{2}{c}{gACC} & "
        r"\multicolumn{2}{c}{OBJ} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        rf"Comparison (first vs second) & $10$--$19${SL} & $20$--$29$ & $10$--$19${SL} & $20$--$29$ & $10$--$19${SL} & "
        rf"$20$--$29$ & $10$--$19${SL} & $20$--$29$ \\"]
upl_ed = {}
for b in ("s10-19", "s20-29"):
    for arm in ("L1R250:pre_mass_off", "SC:L1R250", "SC:L5", "L5:pre_mass_off"):
        v = [num(TAB[(arm, c, s)]["upload_numbers"]) for c in CONFIGS for s in BLOCKS[b] if (arm, c, s) in TAB]
        upl_ed[(arm, b)] = float(np.mean(v))
cap = (r"Compute-matched and $R=250$ comparisons (exploratory; seeds $10$--$19$ were already unblinded, and on seeds "
       r"$20$--$29$ these are not pre-registered tests; the two pairs with both methods at $R=250$ were coded in the "
       r"frozen analysis before the fresh run, the others are post hoc; $^{\S}$: second look). Entries: configurations on which the first "
       r"method is significantly better/worse (OBJ: lower/higher), unadjusted (Holm over 14 configurations). "
       r"Compute-matched pairs take $250$ local passes per client on both sides; rounds and uploads are not matched: mean "
       r"uploaded numbers per run over the 14 configurations on seeds $20$--$29$ are "
       rf"{upl_ed[('L1R250:pre_mass_off', 's20-29')]:.0f} for \texttt{{pre\_mass\_off}} ($L=1$, $R=250$), "
       rf"{upl_ed[('SC:L1R250', 's20-29')]:.0f} for SC-FFCM ($L=1$, $R=250$), {upl_ed[('SC:L5', 's20-29')]:.0f} for "
       rf"SC-FFCM ($L=5$, $R=50$) and {upl_ed[('L5:pre_mass_off', 's20-29')]:.0f} for \texttt{{pre\_mass\_off}} "
       r"($L=5$, $R=50$). " + stats_esm() + ".")
ed_cnt = table("tab:ed-counts", cap, "@{}lcccccccc@{}", head, body)

DEF_ROWS = [("DEF-L1", 1), ("DEF-L5", 5), ("DEF-L50", 50)]
b4 = B4.set_index("L")
DEF_MEAN = {}
for _hid, L in DEF_ROWS:
    _cf = CONFIGS if L != 50 else NON_MNIST
    _dd = pd.read_csv(T10 / f"ed_scffcm_default_L{L}_s00-19.csv")
    _dd = _dd[_dd.dataset.isin(_cf) & _dd.seed.between(0, 19)]
    _cal = [num(TAB[(f"SC:L{L}", c, s_)]["mean_acc"]) for c in _cf for s_ in range(20) if (f"SC:L{L}", c, s_) in TAB]
    check(len(_dd) == len(_cf) * 20 and len(_cal) == len(_cf) * 20, f"SC-FFCM L={L}: default and calibrated runs complete")
    DEF_MEAN[L] = (float(_dd.mean_acc.astype(float).mean()), float(np.mean(_cal)))
    _m = re.search(rf"- L={L}: mean_acc over {len(_cf)} configurations x seeds 0-19: default ([0-9.]+), calibrated ([0-9.]+)",
                   EXTRA_MD.read_text())
    check(_m is not None and (f"{DEF_MEAN[L][0]:.3f}", f"{DEF_MEAN[L][1]:.3f}") == (_m.group(1), _m.group(2)),
          f"SC-FFCM L={L}: default/calibrated mean client ACC equals EVIDENCE_T10_extra.md")
body = []
for hid, L in DEF_ROWS:
    r = b4.loc[L]
    body.append(" & ".join([rf"$L={L}$", xc("s00-09", hid, "mean_acc"), xc("s10-19", hid, "mean_acc"),
                            xc("s00-09", hid, "fcm_obj"), xc("s10-19", hid, "fcm_obj"),
                            v3(DEF_MEAN[L][0]), v3(DEF_MEAN[L][1]), str(int(r.runs)),
                            str(int(r.non_finite_or_error)), str(int(r["merges_min_dist_lt_1e-3"])),
                            str(int(r.fcm_obj_gt_5x_cfcm)), str(int(r.n_rounds_lt_50))]) + r" \\")
head = [r" & \multicolumn{4}{c}{Default vs calibrated steps} & \multicolumn{2}{c}{Mean client ACC} & "
        r"\multicolumn{5}{c}{Runs at the default steps, seeds $0$--$19$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-12}",
        r" & \multicolumn{2}{c}{Mean client ACC} & \multicolumn{2}{c}{OBJ} & \multicolumn{2}{c}{seeds $0$--$19$} & & Non- & "
        r"& OBJ${}>5\times$ & Stopped \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        rf"SC-FFCM & $0$--$9$ & $10$--$19${SL} & $0$--$9$ & $10$--$19${SL} & default & calibrated & Runs & finite & Merged & "
        r"C-FCM & early \\"]
cap = (r"SC-FFCM at the released default steps ($\eta_l=0.2$, $\eta_g=0.5$), $R=50$, all 14 configurations (12 "
       r"non-MNIST for $L=50$). Left: configurations on which the default steps are significantly better/worse than "
       r"the calibrated steps at the same $L$, unadjusted (Holm); $^{\S}$: second look. Middle: mean client ACC at the "
       r"default and at the calibrated steps over the configurations compared in the row (twelve non-MNIST for $L=50$) "
       r"and seeds $0$--$19$. Right: runs with "
       r"non-finite prototypes, merged "
       r"prototypes (minimum distance below $10^{-3}$), a final OBJ above five times that of C-FCM on the same seed (a "
       r"proxy of the calibration rule's stability criterion) and runs that stopped before $R=50$. " + stats_esm() + ".")
ed_def = table("tab:ed-default", cap, "@{}lccccccccccc@{}", head, body)
emit("tabS_ed", "\n".join([ed_cal, ed_cnt, ed_def]), "tab:ed-cal, tab:ed-counts, tab:ed-default",
     "Online Resource 1 (E-D; the R=250 calibration can sit next to the existing tab:scffcmcal)", "none (new)",
     "E-D: R=250 calibration of SC-FFCM at L=1 (steps, gap, stable/63, edge); counts of the R=250 and compute-matched "
     "pairs on four criteria, seeds 10-19 and 20-29, with mean uploads; released-default SC-FFCM vs calibrated counts "
     "and run diagnostics.")

# ---------------------------------------------------------------------------------------------- E-E Pedrycz
pc = {L: PEDCAL[L].set_index("dataset") for L in PEDCAL}
PM1 = {c: bmean("L1:pre_mass_off", c, "s00-09") for c in CONFIGS}


def mean20(arm, c):
    v = [num(BT[(arm, c, s)]["mean_acc"]) for s in range(20) if (arm, c, s) in BT]
    v = [x for x in v if np.isfinite(x)]
    return v3(np.mean(v)) if v else ("n.s." if arm.startswith("PED") else "n.r.")


def alpha_cell(L, c):
    r = pc[L].loc[c]
    if not np.isfinite(num(r.alpha)):
        return "none"
    return fnum(r.alpha) + (r"$^{e}$" if int(num(r.on_edge)) == 1 else "")


def ped_cells(c):
    return ([alpha_cell(L, c) for L in (1, 5, 10)] + [str(int(pc[L].loc[c, "n_stable"])) for L in (1, 5, 10)]
            + [mean20(f"PED:L{L}", c) for L in (1, 5, 10)]
            + [mean20("L1:pre_mass_off", c), mean20("L1:post_none_off", c), mean20("L5:pre_mass_off", c),
               mean20("L5:post_none_off", c)])


body = config_block(15, ped_cells)
head = [r" & & \multicolumn{3}{c}{Calibrated $\alpha$} & \multicolumn{3}{c}{Stable (of " +
        str(int(pc[1].n_grid.iloc[0])) + r")} & \multicolumn{7}{c}{Mean client ACC, seeds $0$--$19$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-15}",
        r" & & & & & & & & \multicolumn{3}{c}{Pedrycz} & \multicolumn{2}{c}{$L=1$} & \multicolumn{2}{c}{$L=5$} \\",
        r"\cmidrule(lr){9-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}",
        r"Configuration & $m^*$ & $L{=}1$ & $L{=}5$ & $L{=}10$ & $L{=}1$ & $L{=}5$ & $L{=}10$ & $L{=}1$ & $L{=}5$ & "
        r"$L{=}10$ & \texttt{pre\_mass\_off} & \texttt{post\_none\_off} & \texttt{pre\_mass\_off} & "
        r"\texttt{post\_none\_off} \\"]
cap = (r"Pedrycz's gradient-based federated FCM (exploratory; seeds $0$--$19$; the rule of "
       rf"{paper_ref('sec:protocol-additional')}, run on the delegated workstation, {paper_ref('sec:protocol-compute')}). Step $\alpha$ calibrated per configuration and "
       rf"$L$ on seed index $100$ by the rule of {paper_ref('sec:protocol-baselines')} over "
       rf"{int(pc[1].n_grid.iloc[0])} values from $0.0005$ to $0.5$; $^{{e}}$: on the grid's lower edge; none: no stable "
       r"value (n.s.: not run there; such configurations stay in the Holm families with $p=1$). Mean client ACC over the "
       r"twenty seeds, with \texttt{pre\_mass\_off} and the ungated POST cell at the same $L$ for comparison. Counts: "
       r"Table~\ref{tab:ee-counts}.")
ped_tab = table("tab:ee-pedrycz", cap, "@{}l" + "c" * 14 + "@{}", head, body, sep="2pt")
PED_ROWS = [("PED1-PM", r"\texttt{pre\_mass\_off} vs Pedrycz, both $L=1$"),
            ("PED1-FFCM", r"\texttt{post\_none\_off} vs Pedrycz, both $L=1$"),
            ("PED5-PM", r"\texttt{pre\_mass\_off} vs Pedrycz, both $L=5$"),
            ("PED5-FFCM", r"\texttt{post\_none\_off} vs Pedrycz, both $L=5$")]
body = [" & ".join([lab] + [xc(b, h, m) + (SL if b == "s10-19" and h.startswith("PED1") else "")
                            for m in ("mean_acc", "fcm_obj") for b in ("s00-09", "s10-19")]) + r" \\"
        for h, lab in PED_ROWS]
head = [r" & \multicolumn{2}{c}{Mean client ACC} & \multicolumn{2}{c}{OBJ} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Comparison (first vs second) & $0$--$9$ & $10$--$19$ & $0$--$9$ & $10$--$19$ \\"]
cap = (r"Pedrycz's gradient rule against the mass-weighted cells at the same $L$ (exploratory; all runs of Pedrycz's rule "
       r"are exploratory, and $^{\S}$ marks the one-step comparisons on seeds $10$--$19$, a second look): configurations on "
       r"which the first method is significantly better/worse (OBJ: lower/higher), unadjusted (Holm over 14 "
       r"configurations; those without a stable $\alpha$ count with $p=1$). " + stats_esm() + ".")
ped_cnt = table("tab:ee-counts", cap, "@{}lcccc@{}", head, body)
emit("tabS_ee", "\n".join([ped_tab, ped_cnt]), "tab:ee-pedrycz, tab:ee-counts", "Online Resource 1 (E-E)",
     "none (new)", "E-E: Pedrycz alpha per L with stable counts and edge marks, mean ACC seeds 0-19 of Pedrycz L=1/5/10 "
                   "and of pre_mass_off/post_none_off at L=1/5; count table vs pre_mass_off and post_none_off.")

# ---------------------------------------------------------------------------------------------- E-G partial participation
EG_ROWS = [("PRE-POST", r"PRE vs POST, ungated"), ("PM-GF", r"\texttt{pre\_mass\_off} vs GF-PFedFCM"),
           ("PM-SC", r"\texttt{pre\_mass\_off} vs SC-FFCM")]
body = []
for L in (1, 5):
    body.append(lab_row(5, rf"\emph{{$L={L}$ for both methods}}") + r" \\")
    for k, lab in EG_ROWS:
        body.append(" & ".join([lab] + [xc("s00-09", f"P{p}L{L}-{k}", m) for m in ("mean_acc", "fcm_obj")
                                        for p in ("05", "1")]) + r" \\")
led = LEDGER[LEDGER.experiment == "E-G partial participation"]
CH_ARMS = [("post_none_off", r"\texttt{post\_none\_off}"), ("pre_none_off", r"\texttt{pre\_none\_off}"),
           ("pre_mass_off", r"\texttt{pre\_mass\_off}"), ("post_footprint_on", "GF-PFedFCM"), ("SC", "SC-FFCM")]


def merges(L, arm):
    f = led[led.file.str.contains(f"_L{L}_")]
    f = f[f.cell_or_method == ("scffcm" if arm == "SC" else arm)] if arm != "SC" else f[f.file.str.contains("scffcm")]
    if f.empty:
        return "n.r."
    return f"{int(f.merges.sum())}/{int(f.runs.sum())}"


body2 = []
for L in (1, 5):
    body2.append(lab_row(4, rf"\emph{{$L={L}$}}") + r" \\")
    for arm, lab in CH_ARMS:
        body2.append(" & ".join([lab, xc("s00-09", f"CH-L{L}-{arm}", "mean_acc"), xc("s00-09", f"CH-L{L}-{arm}", "fcm_obj"),
                                 merges(L, arm)]) + r" \\")
cap = (r"Partial participation (exploratory; one rate: half of the clients per round, drawn without replacement; "
       r"seeds $0$--$9$; SC-FFCM with participation fraction $0.5$ at its calibrated steps; GF-PFedFCM with early stop). "
       r"(a) The contrasts under partial ($P=0.5$) and full ($P=1$) participation on the same seeds. (b) Each method at "
       r"$P=0.5$ against itself at $P=1$, and the runs at $P=0.5$ with merged prototypes (minimum distance below "
       r"$10^{-3}$). Entries: configurations on which the first method (in (b): $P=0.5$) is significantly better/worse "
       r"(OBJ: lower/higher), unadjusted (Holm over 14 configurations). Other rates and client sampling schemes were "
       r"not studied. " + stats_esm() + ".")
tex = "\n".join([
    r"\begin{table}[htbp]", r"\centering", r"\tablefontsize", rf"\caption{{{cap}}}", r"\label{tab:eg-partial}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"\multicolumn{5}{@{}l}{\emph{(a) Contrasts}} \\", r"\midrule",
    r" & \multicolumn{2}{c}{Mean client ACC} & \multicolumn{2}{c}{OBJ} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"Comparison & $P=0.5$ & $P=1$ & $P=0.5$ & $P=1$ \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
    "", r"\medskip", "",
    r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"\multicolumn{4}{@{}l}{\emph{(b) $P=0.5$ vs $P=1$, same method}} \\",
    r"\midrule", r"Method & Mean client ACC & OBJ & Merged at $P=0.5$ \\", r"\midrule", *body2, r"\bottomrule",
    r"\end{tabular}", r"\end{table}", ""])
emit("tabS_eg", tex, "tab:eg-partial", "Online Resource 1 (E-G)", "none (new)",
     "E-G: (a) PRE vs POST, pre_mass_off vs GF-PFedFCM and vs SC-FFCM at P=0.5 and P=1, L=1 and 5, ACC and OBJ counts; "
     "(b) each method at P=0.5 vs P=1 with merges at P=0.5.")

# ---------------------------------------------------------------------------------------------- E-H tilt diagnostics
TRAJ = [("post_footprint_off_L5", r"\texttt{post\_footprint\_off}, $L=5$"),
        ("pre_footprint_off_L5", r"\texttt{pre\_footprint\_off}, $L=5$"),
        ("pre_footprint_off_L1", r"\texttt{pre\_footprint\_off}, $L=1$")]
eh = EH.set_index(["trajectory", "dataset"])
extra_txt = EXTRA_MD.read_text()
RHO = {}
for tr, _ in TRAJ:
    sub = EH[EH.trajectory == tr].set_index("dataset").loc[CONFIGS]
    for col, key in (("tilt_cv_median", "weighted tilt CV (all pairs)"), ("rel_median", "disp/spread"),
                     ("cv_unclipped_median", "weighted tilt CV (unclipped pairs)"),
                     ("cv_clipped_median", "weighted tilt CV (clipped pairs)"),
                     ("share_r_unclipped_mean", "server-weight share of unclipped pairs")):
        rho, p = spearmanr(sub[col], sub.acc_gap_fp_vs_mass)
        m = re.search(rf"\[{tr}\] {re.escape(key)}: rho = ([+-][0-9.]+), p = ([0-9.]+)", extra_txt)
        check(m is not None and f"{rho:+.3f}" == m.group(1) and f"{p:.4f}" == m.group(2),
              f"Spearman {tr} {col} reproduces theory/SUMMARY.md ({m.group(1) if m else '?'})")
        RHO[(tr, col)] = (rho, p)
check(RHO[("post_footprint_off_L5", "tilt_cv_median")][1] < ALPHA
      and RHO[("pre_footprint_off_L5", "tilt_cv_median")][1] >= ALPHA
      and RHO[("pre_footprint_off_L1", "tilt_cv_median")][1] >= ALPHA,
      "the tilt-CV association is significant along POST only, not along either PRE trajectory")
check(set(EH.n_seeds) == {5}, "E-H diagnostics over five seeds")


def eh_cells(c):
    out = []
    for tr, _ in TRAJ:
        r = eh.loc[(tr, c)]
        out += [v3(r.tilt_cv_median), v3(r.rel_median), v3(r.acc_gap_fp_vs_mass)]
    return out


body = config_block(11, eh_cells)
body.append(r"\midrule")
body.append(lab_row(11, r"\emph{Spearman correlation with the accuracy gap over the 14 configurations: $\rho$ ($p$)}")
            + r" \\")
for col, lab in (("tilt_cv_median", "Tilt CV, all pairs"), ("cv_unclipped_median", "Tilt CV, unclipped pairs"),
                 ("cv_clipped_median", "Tilt CV, clipped pairs"), ("rel_median", "Displacement/spread"),
                 ("share_r_unclipped_mean", "Server-weight share, unclipped")):
    cells = []
    for tr, _ in TRAJ:
        rho, p = RHO[(tr, col)]
        cells.append(mc(3, f"${rho:+.3f}$ ({p:.4f})"))
    body.append(lab_row(2, lab) + " & " + " & ".join(cells) + r" \\")
head = [r" & & " + " & ".join(mc(3, t) for _, t in TRAJ) + r" \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Configuration & $m^*$ & " + " & ".join([r"CV & Disp. & Gap"] * 3) + r" \\"]
gap_src = sorted(set(EH.gap_source))
cap = (r"Tilt diagnostics along three trajectories (seeds $0$--$4$): CV, median over rounds and prototypes of the "
       r"coefficient of variation of the tilt $r_{pj}/q_{pj}$ under the mass-gated server weights; Disp., median "
       r"displacement of the one-round server target by the footprint relative to the weighted spread of the local "
       r"centres; Gap, $|\mathrm{ACC}(\text{footprint gate})-\mathrm{ACC}(\text{mass gate})|$ of the two gated cells at "
       r"that mass point and $L$ (seeds $0$--$9$). Bottom: rank correlations with the Gap, also for the clipped and "
       r"unclipped client--prototype pairs separately. The tilt CV is associated with the gap along the POST trajectory "
       r"and not along either PRE trajectory, so it does not predict the gate differences at PRE. Descriptive "
       r"($n=14$ configurations, unadjusted).")
emit("tabS_eh", table("tab:eh-tilt", cap, "@{}lcccccccccc@{}", head, body), "tab:eh-tilt", "Online Resource 1 (E-H)",
     "none (new; extends tab:theorylink of Appendix B along PRE)",
     "E-H: per configuration tilt CV, displacement/spread and footprint-vs-mass gap along post_footprint_off L=5, "
     "pre_footprint_off L=5 and L=1; Spearman rows (all/unclipped/clipped CV, displacement, weight share).")

# ---------------------------------------------------------------------------------------------- exact-test count changes
HYP_TX = {"S-PRE-SC1": r"PRE ($L=1$) vs SC-FFCM ($L=1$)", "S-PRE-SC5": r"PRE ($L=1$) vs SC-FFCM ($L=5$)",
          "H4": r"H4: PRE vs POST ($L=1$)", "H9": r"H9: GF-PFedFCM vs \texttt{post\_footprint\_off}"}
MET_TX = {"mean_acc": "mean client ACC", "gacc": "gACC", "worst_acc": "worst-client ACC", "fcm_obj": "OBJ"}
check(set(CHG.metric) == {"fcm_obj"} and set(CHG.hyp) <= {"S-PRE-SC1", "S-PRE-SC5"},
      "legacy vs exact count changes involve only fcm_obj of the secondary PRE vs SC-FFCM contrasts")


def _cnt_dir(t, col):
    sig = t[col] < ALPHA
    return int((sig & (t.direction == "a")).sum()), int((sig & (t.direction == "b")).sum())


EXC = []   # every (block, hyp, metric) family whose unadjusted or Holm count differs between the two tests
for (blk, hyp, met), t in TESTS.groupby(["block", "hyp", "metric"], sort=False):
    t = t.copy()
    t["p_leg"] = pd.to_numeric(t.p_legacy, errors="coerce").fillna(1.0).astype(float)
    t["p_leg_holm"] = R.holm(t.p_leg.tolist())
    lu, lh = _cnt_dir(t, "p_leg"), _cnt_dir(t, "p_leg_holm")
    eu, eh = _cnt_dir(t, "p_exact"), _cnt_dir(t, "p_holm")
    s_ = srow(blk, hyp, met)
    check(lu == (s_.better_legacy, s_.worse_legacy) and eu == (s_.better_unadj, s_.worse_unadj)
          and eh == (s_.better_holm, s_.worse_holm), f"legacy/exact recount of {hyp} {blk} {met} equals the summary")
    if lu != eu or lh != eh:
        EXC.append((blk, hyp, met, lu, lh, eu, eh))
_unadj = {(b, h, m) for b, h, m, lu, lh, eu, eh in EXC if lu != eu}
check(_unadj == {(r.block, r.hyp, r.metric) for r in CHG.itertuples()},
      "families with a changed unadjusted count equal analysis_final_count_changes.csv")
check(not any(b == "s20-29" and h.startswith("H") for b, h, *_ in EXC),
      "no count of the fresh block's hypotheses changes between the legacy and the exact test")
check(not any(m == "mean_acc" and lu != eu for b, h, m, lu, lh, eu, eh in EXC),
      "no unadjusted count on mean client ACC changes between the legacy and the exact test")
check(set(h for _, h, *_ in EXC) <= set(HYP_TX), "every changed family has a label")
BLK_ORDER = {"s00-09": 0, "s10-19": 1, "s20-29": 2}
SECOND_LOOK_HYP = {"S-PRE-SC1", "S-PRE-SC5", "H4", "H5", "H6", "H7", "H8"}   # L=1 or SC-FFCM contrasts


def _blk_cell(b, h):
    """Seeds cell; seeds 10-19 of an L=1 or SC-FFCM contrast are a second look (section sign)."""
    txt = BLK_TXT[b].replace("seeds ", "")
    return txt + (SL if b == "s10-19" and h in SECOND_LOOK_HYP else "")


body = [" & ".join([_blk_cell(b, h), HYP_TX[h], MET_TX[m],
                    f"{cnt(*lu)} ({cnt(*lh)})", f"{cnt(*eu)} ({cnt(*eh)})"]) + r" \\"
        for b, h, m, lu, lh, eu, eh in sorted(EXC, key=lambda x: (BLK_ORDER[x[0]], x[1], x[2]))]
head = [r"Seeds & Comparison (first vs second) & Criterion & SciPy default & Exact test \\"]
cap = (r"Counts that change when the earlier analyses' test (the default of \texttt{scipy.stats.wilcoxon}, SciPy 1.17, "
       r"exact for these sample sizes, zeros and ties by floating-point equality, OBJ differences below $10^{-10}$ "
       r"treated as ties) is replaced by the exact test with the fixed zero tolerance $10^{-12}$ and tie rounding to "
       rf"twelve decimals ({paper_ref('sec:protocol-stats')}), over every hypothesis and secondary pair of the frozen "
       rf"analysis in every block and on every criterion ({len(TESTS.groupby(['block', 'hyp', 'metric']))} families; the "
       r"mask-gate interactions and the split-sample check, new in this study, were analysed with the exact test only). "
       r"Entries: configurations on "
       r"which the first method is significantly better/worse (OBJ: lower/higher), unadjusted and, in parentheses, "
       r"Holm-adjusted over the configurations of the family, with Holm applied in the same way to the $p$-values of "
       rf"each test. Of these families, {len(_unadj)} change their unadjusted count and "
       rf"{sum(1 for b, h, m, lu, lh, eu, eh in EXC if lh != eh)} their Holm count; no count of the fresh block's "
       r"hypotheses and no unadjusted count on mean client ACC changes. $^{\S}$: second look (seeds $10$--$19$ of a "
       r"contrast with $L=1$ or SC-FFCM).")
emit("tabS_exact", table("tab:exact-changes", cap, "@{}lllcc@{}", head, body, sep="4pt"), "tab:exact-changes",
     "Online Resource 1 (statistics section)", "none (new)",
     "Exact-test count changes, unadjusted and Holm: every family of the frozen analysis whose unadjusted or Holm count "
     "differs between SciPy's default and the exact test with fixed zero tolerance (Holm applied to each test's p).")

# ---------------------------------------------------------------------------------------------- p-values for tab:scffcm


def p_cell(src, hyp, block, c):
    df = src_df(src)
    t = df[(df.block == block) & (df.hyp == hyp) & (df.metric == "mean_acc") & (df.dataset == c)]
    if t.empty:
        return "n.r."
    r = t.iloc[0]
    if r.status != "ok":
        return r"-- (0)"
    return f"{pv(r.p_exact)}/{pv(r.p_holm)}" + (f" ({int(r.n_eff)})" if r.n_eff < 10 else "")


grpA = SC_COLS[:9]
grpB = SC_COLS[9:]
body = config_block(11, lambda c: [p_cell(s, h, b, c) for s, h, b in grpA])
head = [r" & & \multicolumn{3}{c}{H6: $L=1$ vs $L=1$} & \multicolumn{3}{c}{H7: $L=1$ vs $L=5$} & "
        r"\multicolumn{3}{c}{H8: $L=5$ vs $L=5$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Configuration & $m^*$ & " + " & ".join([rf"$0$--$9$ & $10$--$19${SL} & $20$--$29$"] * 3) + r" \\"]
tabref = r"\ref{tab:scffcm}"
cap = (rf"Exact $p$/Holm-adjusted $p$ (and $n'$ where below $10$) of the entries of Table~{tabref}, "
       r"communication-matched columns (\texttt{pre\_mass\_off} vs SC-FFCM, $R=50$). $^{\S}$: second look; -- (0): all "
       r"paired differences zero. With $n'=10$ the smallest attainable $p$ is $0.0020$. " + stats_esm() + ".")
p_a = table("tab:scffcm-p-a", cap, "@{}lcccccccccc@{}", head, body)
body = config_block(10, lambda c: [p_cell(s, h, b, c) for s, h, b in grpB])
head = [r" & & \multicolumn{2}{c}{$L=1$ vs $50$} & \multicolumn{2}{c}{$L{=}1,R{=}250$ vs $L{=}5,R{=}50$} & "
        r"\multicolumn{2}{c}{both $L{=}1$, $R{=}250$} & \multicolumn{2}{c}{ungated PRE vs SC-FFCM} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        rf"Configuration & $m^*$ & $10$--$19${SL} & $20$--$29$ & $10$--$19${SL} & $20$--$29$ & $10$--$19${SL} & $20$--$29$ & "
        r"$L{=}1$ vs $1$ & $L{=}1$ vs $5$ \\"]
cap = (rf"As Table~\ref{{tab:scffcm-p-a}} for the remaining columns of Table~{tabref}: SC-FFCM with its "
       r"released default of $L=50$ local steps (12 non-MNIST configurations; Holm over 12), the compute-matched and "
       r"$R=250$ comparisons (exploratory; $^{\S}$: second look) and the ungated PRE rule ($L=1$) against SC-FFCM on the "
       r"fresh block (secondary). " + stats_esm() + ".")
p_b = table("tab:scffcm-p-b", cap, "@{}lccccccccc@{}", head, body)
emit("tabS_scffcm_p", "\n".join([p_a, p_b]), "tab:scffcm-p-a, tab:scffcm-p-b",
     "Online Resource 1 (next to tab:scffcmcal)", "none (new)",
     "Exact p / Holm p (n' when below 10) for every entry of tab:scffcm.")

# ---------------------------------------------------------------------------------------------- fresh-block detail
def fd_cells(h, c):
    r = TESTS[(TESTS.block == "s20-29") & (TESTS.hyp == h) & (TESTS.metric == "mean_acc") & (TESTS.dataset == c)].iloc[0]
    if r.status != "ok":
        return [r"$0.000$\,(0)", "--"]
    return [d3(r.mean_diff_raw, marks_of(r.p_exact, r.p_holm)) + neff_note(r.n_eff),
            "[" + d3(r.ci95_lo) + ", " + d3(r.ci95_hi) + "]"]


FD_LABELS = {"a": ["H1", "H2", "H3"], "b": ["H4", "H5", "H6"], "c": ["H7", "H8", "H9"]}
FD_NOTE = {"a": "H3: the interaction \\eqref{eq:protocol-did}. ",
           "b": "", "c": "H8 is stated in favour of the second method (SC-FFCM); H9 is personalization on minus off. "}


def fd_table(key):
    hyps = FD_LABELS[key]
    body = config_block(2 + 2 * len(hyps), lambda c: [x for h in hyps for x in fd_cells(h, c)])
    head = [r" & & " + " & ".join(mc(2, h) for h in hyps) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(hyps))),
            r"Configuration & $m^*$ & " + " & ".join([r"$\Delta$ & $95\%$ interval"] * len(hyps)) + r" \\"]
    cap = (r"Fresh block (seeds $20$--$29$, pre-registered), mean client ACC: paired mean difference $\Delta$, first "
           rf"minus second method of {', '.join(hyps)}, with its $95\%$ $t$-interval over the ten seeds; in parentheses "
           rf"$n'$ where below $10$. {FD_NOTE[key]}Counts and replication: Table~{MAINLAB.get('tab:confirm', '??')} of "
           r"the paper. " + MARKS_TXT + ". " + stats_esm() + ".")
    cap = cap.replace(r"\eqref{eq:protocol-did}", "(" + MAINLAB.get("eq:protocol-did", "??") + ")")
    return table(f"tab:fresh-detail-{key}", cap, "@{}l" + "c" * (1 + 2 * len(hyps)) + "@{}", head, body)


fd = "\n".join(fd_table(k) for k in "abc")
emit("tabS_fresh_detail", fd, "tab:fresh-detail-a, tab:fresh-detail-b, tab:fresh-detail-c",
     "Online Resource 1 (fresh block)", "none (new)",
     "Per-configuration fresh-block mean differences with 95% t-intervals, marks and n' for H1-H9 (mean client ACC), "
     "three floats (H1-H3, H4-H6, H7-H9).")

# ================================================================================================ RESULTS-STAGE TABLES
# Tables written by _results/results_stage_tables.py for the main text. The three former Appendix A tables among them
# moved to Online Resource 1 in the fix stage: their references are rewritten for a document compiled on its own
# (typed "Section~6.9 of the paper", equation numbers from main.aux, "Online Resource 1, Table~Sn" -> \ref); the
# tabular bodies are copied byte for byte (checked). tab_L1 stays in the main text; its caption gains the three
# centralized budgets and resolves its Online Resource 1 reference from ESM_1.aux.
RS = RES / "tables"
MOVED = {"tab:gates-interaction", "tab:L1-full", "tab:confirm-all", "tab:decision", "tab:L1",
         "tab:substitution", "tab:scffcm", "tab:baselines", "tab:gfgain", "tab:fuzziness", "tab:lossless",
         "tab:theorylink", "tab:reconcile"}
ESM_INV = {v: k for k, v in ESMLAB.items() if k.startswith("tab:")}


def rs_read(name):
    f = RS / f"{name}.tex"
    check(f.exists(), f"_results/tables/{name}.tex exists (written by _results/results_stage_tables.py)")
    return f.read_text(encoding="utf-8")


def tabular_part(tex):
    return tex[tex.index(r"\begin{tabular}"):tex.rindex(r"\end{tabular}")]


def to_esm(tex):
    def sec(m):
        kind, lab = m.group(1), m.group(2)
        if lab in MOVED or (lab in ESMLAB and lab not in MAINLAB):
            return f"{kind}~\\ref{{{lab}}}"
        if lab not in MAINLAB:
            WARN.append(f"main.aux has no label {lab}: re-run after rebuilding main.tex")
        return f"{kind}~{MAINLAB.get(lab, '??')} of the paper"

    def osr(m):
        lab = ESM_INV.get(m.group(2))
        if lab is None:
            WARN.append(f"ESM_1.aux has no table {m.group(2)}")
            return m.group(0)
        return f"{m.group(1)}~\\ref{{{lab}}}"

    tex = re.sub(r"(Section|Sections|Table|Tables|Figure|Appendix|Corollary|Proposition|Lemma|Remark|Example|Definition)~\\ref\{([^}]*)\}", sec, tex)
    tex = re.sub(r"Online Resource 1, (Tables?)~(S\d+)", osr, tex)
    tex = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({MAINLAB.get(m.group(1), '??')})", tex)
    check("\\cite" not in tex, "no citation left in a table moved to Online Resource 1")
    left = [x for x in re.findall(r"\\ref\{([^}]*)\}", tex) if x not in MOVED and x not in ESMLAB]
    check(not left or not ESMLAB, f"moved table: every remaining \\ref is internal to Online Resource 1 ({left})")
    return tex


emit("tabS_gates_interaction", to_esm(GATES_TEX), "tab:gates-interaction",
     "Online Resource 1 (2026-10-07: moved from the paper's Table 6; Figure 6 of the paper shows the values)",
     "tab_gates_interaction of the paper", "E-A: fresh-block interaction D for four gates; count rows.")
emit("tabS_decision", to_esm(DECISION_TEX), "tab:decision",
     "Online Resource 1 (2026-10-07: moved from the paper's Table 8)", "tab_decision of the paper",
     "Decision analysis as pre-specified, seeds 10-29: 7 actions x 5 criteria (mean | max regret), recommended in bold; "
     "per-block recommendations.")

# ---------------------------------------------------------------------------------------------- post hoc (T11) tables
T11 = T10.parent / "t11_posthoc"
PH = pd.read_csv(T11 / "decision_all.csv")
check(dec_same(PH[(PH.variant == "T10") & (PH.block == "s10-29")].to_dict("records"), DEC),
      "T11 analyze: variant T10 (as run) reproduces the frozen decision analysis on seeds 10-29")
check(dec_same(PH[(PH.variant == "T10") & (PH.block == "s10-19")].to_dict("records"), DEC19),
      "T11 analyze: variant T10 on seeds 10-19 reproduces the frozen s00-19 decision analysis")
check(int(PH[PH.status == "ok"].n_substituted.sum()) == 0, "T11 analyses: no substituted runs")


def _digits(D):
    for dig in (4, 5, 6):
        ok = True
        for crit in CRIT:
            s_ = D[D.criterion == crit]
            for col, rec in (("mean_regret", "rec_mean_regret"), ("max_regret", "rec_max_regret")):
                b = f"{s_[s_[rec] == 1][col].iloc[0]:.{dig}f}"
                ok &= int((s_[col].map(lambda x: f"{x:.{dig}f}") == b).sum()) == 1
        if ok:
            return dig
    raise SystemExit("post hoc decision table: a recommended value is not unique at 6 decimals")


def ph_table(variant, label, title):
    d = PH[PH.variant == variant]
    D = d[d.block == "s10-29"]
    for crit in CRIT:
        for rec in ("rec_mean_regret", "rec_max_regret"):
            check(int(D[D.criterion == crit][rec].sum()) == 1, f"post hoc {variant}: one recommended action, {crit} {rec}")
    dig = _digits(D)
    body_ = []
    for a in ACT:
        cells = [ACT_TEX[a]]
        for crit in CRIT:
            r = D[(D.criterion == crit) & (D.action == a)].iloc[0]
            mr, xr = f"{r.mean_regret:.{dig}f}", f"{r.max_regret:.{dig}f}"
            cells += [rf"$\mathbf{{{mr}}}$" if r.rec_mean_regret else f"${mr}$",
                      rf"$\mathbf{{{xr}}}$" if r.rec_max_regret else f"${xr}$"]
        body_.append(" & ".join(cells) + r" \\")
    body_.append(r"\midrule")
    body_.append(lab_row(11, r"\emph{Recommended action per seed block (in each pair of columns: min.\ mean regret, "
                              r"min.\ max regret)}") + r" \\")
    for lab_, blk in ((r"Seeds $10$--$19$", "s10-19"), (r"Seeds $20$--$29$", "s20-29")):
        cells = [lab_]
        for crit in CRIT:
            bm, bx = recs(d[d.block == blk], crit)
            cells += [bm, bx]
        body_.append(" & ".join(cells) + r" \\")
    cap_ = (title + r" Layout, regrets and abbreviations as in Table~\ref{tab:decision}; regrets to " + str(dig)
            + r" decimals, at which the smallest value of every column is unique. Post hoc: not part of the "
              r"pre-registration (Section~\ref{app:posthoc}).")
    return table(label, cap_, "@{}lcccccccccc@{}", DEC_HEAD, body_)


emit("tabS_decision_U", ph_table("U", "tab:decision-U",
     r"Decision analysis with a common stopping rule (variant U): as Table~\ref{tab:decision}, with the four "
     r"non-personalized mass-weighted actions re-run on seeds $10$--$29$ with the stopping rule that GF-PFedFCM and "
     r"SC-FFCM already had (prototype shift below $10^{-5}$), $\alpha_s=0.75$."), "tab:decision-U",
     "Online Resource 1, Section S21 (post hoc)", "none (new)", "Decision analysis, variant U (common stopping rule).")
emit("tabS_decision_C", ph_table("C", "tab:decision-C",
     r"Decision analysis with a common stopping rule and calibrated server steps (variant C), the analysis of "
     + paper_ref("fig:regret", "Figure") + r": as Table~\ref{tab:decision-U}, with $\alpha_s$ of every mass-weighted action "
     r"calibrated per configuration on seed index $100$ by the rule used for SC-FFCM (Table~\ref{tab:alpha-cal})."),
     "tab:decision-C", "Online Resource 1, Section S21 (post hoc)", "none (new)",
     "Decision analysis, variant C (common stopping rule, calibrated alpha_s); the data of Figure 9.")

CAL = pd.concat([pd.read_csv(T11 / f"alpha_cal_L{L}.csv") for L in (5, 1)])
MW = ["F-FCM (post_none_off, L=5)", "GF-PFedFCM (L=5)", "ungated PRE (L=1)", "pre_mass_off (L=1)", "pre_mass_off (L=5)"]
NO_STABLE = []


def acells(c):
    out = []
    for a in MW:
        s_ = CAL[(CAL.action == a) & (CAL.dataset == c)]
        check(len(s_) == 4, f"alpha calibration: four grid points for {a} on {c}")
        ch = s_[s_.chosen == 1]
        if len(ch):
            out.append(f"{ch.alpha_s.iloc[0]:.2f}")
        else:
            NO_STABLE.append((a, c))
            out.append(r"--")
    return out


body_a = config_block(7, acells)
check(sorted(NO_STABLE) == sorted([("F-FCM (post_none_off, L=5)", c) for c in
                                   ("cluster_skew_hard", "dirichlet_0.03", "dirichlet_0.1")]),
      "alpha calibration: no stable value only for F-FCM on Cluster skew and both Dirichlet scenarios")
head_a = [r" & & F-FCM & GF-PFedFCM & PRE & \texttt{pre\_mass\_off} & \texttt{pre\_mass\_off} \\",
          r"Configuration & $m^*$ & $L=5$ & $L=5$ & $L=1$ & $L=1$ & $L=5$ \\"]
cap_a = (r"Server step $\alpha_s$ chosen for the five mass-weighted actions of the decision analysis by the label-free "
         r"rule used for SC-FFCM (" + paper_ref("sec:protocol-baselines") + r"): on seed index $100$, with the "
         r"common stopping rule, among $\alpha_s\in\{0.25,0.5,0.75,1\}$ with finite prototypes and final pooled FCM "
         r"objective $J\le5J_c$, the value with the smallest $|J-J_c|/J_c$, $J_c$ the objective of centralized FCM from "
         r"the same prototypes (at most $300$ iterations). --: no stable value; the action keeps $\alpha_s=0.75$ there. "
         r"PRE: the ungated PRE rule \texttt{pre\_none\_off}. Post hoc (Section~\ref{app:posthoc}).")
emit("tabS_alpha_cal", table("tab:alpha-cal", cap_a, "@{}lcccccc@{}", head_a, body_a), "tab:alpha-cal",
     "Online Resource 1, Section S21 (post hoc)", "none (new)", "chosen alpha_s per configuration and action.")

emit("tabS_confirm_all", to_esm(CONFIRM_ALL_TEX), "tab:confirm-all",
     "Online Resource 1 (2026-10-07: moved from the paper's Table 5, which keeps the fresh block)",
     "the three-block version of tab:confirm", "H1-H9 and H5-ES on seeds 0-9, 10-19 and 20-29.")

for _name, _label, _what in (("tab_substitution", "tab:substitution",
                              "substitution interaction with the footprint gate, decomposed, three seed blocks"),
                             ("tab_gfgain", "tab:gfgain", "where GF-PFedFCM's gains come from; personalization effects"),
                             ("tab_fuzziness", "tab:fuzziness", "fuzzifier sensitivity at m in {1.3, 1.5, 2.0}")):
    _src = rs_read(_name)
    _out = to_esm(_src)
    check(tabular_part(_src) == tabular_part(_out), f"{_name}: tabular body copied unchanged to Online Resource 1")
    emit("tabS_" + _name[4:], _out, _label, "Online Resource 1, Section S14 (moved from the paper's Appendix A)",
         f"the main-text version _results/tables/{_name}.tex (no longer inserted in main.tex)", _what)

_src = rs_read("tab_L1")
check(_src.count(r"\label{tab:L1}") == 1 and "C-FCM" in _src, "tab_L1 (reduced) read")
emit("tab_L1", _src, "tab:L1", "Main text, Section 7.3",
     "_results/tables/tab_L1.tex (2026-10-07: reduced to the columns of Section 7.3; the full table is tab:L1-full)",
     "one local step on the fresh block: PRE and pre_mass_off mean/worst ACC, mass-gate ACC and OBJ differences; "
     "counts of the mass gate in the three blocks.")
_src = rs_read("tab_L1_full")
_old_cfcm = r"a descriptive reference on these seeds (not pre-registered, not tested)."
check(_src.count(_old_cfcm) == 1, "tab_L1_full caption: C-FCM sentence found")
_out = _src.replace(_old_cfcm, r"a descriptive reference on these seeds (not pre-registered, not tested); the exactness "
                    r"check (" + esm_ref("tab:lossless") + r") uses exactly $50$ centralized iterations, and the "
                    r"calibration target $J_c$ of SC-FFCM and the $m^*$ audit at most $300$ "
                    r"(Section~\ref{sec:protocol-baselines}).")
# M7/minor 13 of the pre-submission review: the OBJ column is an absolute difference; the relative change is in Fig. 8.
_old_mg = r"in OBJ \eqref{eq:protocol-obj}, where a positive OBJ difference"
check(_out.count(_old_mg) == 1, "tab_L1_full caption: mass-gate OBJ sentence found")
_out = _out.replace(_old_mg, r"in OBJ \eqref{eq:protocol-obj} (absolute differences; relative changes in "
                    r"Figure~\ref{fig:tradeoff}), where a positive OBJ difference")
# twelve columns do not fit the text width at \tablefontsize; set the table sideways instead of shrinking it.
check(_out.count(r"\begin{table}[htbp]") == 1 and _out.count(r"\end{table}") == 1, "tab_L1_full: one table environment")
_out = _out.replace(r"\begin{table}[htbp]", r"\begin{sidewaystable}").replace(r"\end{table}", r"\end{sidewaystable}")
_out_esm = to_esm(_out)
check(tabular_part(_src) == tabular_part(_out_esm), "tab_L1_full: tabular body copied unchanged")
emit("tabS_L1_full", _out_esm, "tab:L1-full", "Online Resource 1 (2026-10-07: moved from the paper's Table 7)",
     "_results/tables/tab_L1_full.tex", "one local step on the fresh block, all columns (PRE, pre_mass_off, C-FCM, "
     "H/ln c, mass-gate and PRE-POST differences; counts in the three blocks).")

# Online Resource 1 tables written by the results stage carry typed references into the paper ("Table~5 of the paper").
# They are re-resolved here from main.aux through the labels they denote, so that renumbering the paper cannot leave a
# stale number; every typed reference of each file must be listed (checked).
TYPED = {"tabS_main": [("Section~7.3 of the paper", "sec:results-rule", "Section"),
                       ("Table~5 of the paper", "tab:confirm", "Table"),
                       ("Section~6.9 of the paper", "sec:protocol-stats", "Section")],
         "tabS_footprintmass": [("Section~6.8 of the paper", "sec:protocol-metrics", "Section"),
                                ("Section~6.3 of the paper", "sec:protocol-runs", "Section"),
                                ("Section~6.9 of the paper", "sec:protocol-stats", "Section")],
         "tabS_L1_expl": [
                          ("Section~6.6 of the paper", "sec:protocol-baselines", "Section"),
                          ("Section~6.9 of the paper", "sec:protocol-stats", "Section")]}
_TYPED_LABEL = {"tabS_main": "tab:main", "tabS_footprintmass": "tab:footprintmass", "tabS_L1_expl": "tab:L1-expl"}
# Round 3: the typed numbers are those results_stage_tables.py resolved from main.aux, so they are recomputed here from
# the same main.aux (the literals above only record the labels; a stale literal number no longer breaks the check,
# while a results-stage file typed against an older main.aux still fails it).
TYPED = {_n: [(f"{_kind}~{MAINLAB.get(_lab, '??')} of the paper", _lab, _kind) for _t, _lab, _kind in _r]
         for _n, _r in TYPED.items()}
for _name, _refs in TYPED.items():
    _src = rs_read(_name)
    _found = set(re.findall(r"(?:Section|Table)~[0-9.]+ of the paper", _src))
    check(_found == {t for t, _, _ in _refs}, f"{_name}: every typed reference into the paper is mapped to a label")
    _out = _src
    for _k, (_t, _lab, _kind) in enumerate(_refs):
        _out = _out.replace(_t, f"@@TYPED{_k}@@")
    for _k, (_t, _lab, _kind) in enumerate(_refs):
        _out = _out.replace(f"@@TYPED{_k}@@", paper_ref(_lab, _kind))
    check(tabular_part(_src) == tabular_part(_out), f"{_name}: tabular body copied unchanged")
    emit(_name, _out, _TYPED_LABEL[_name], "Online Resource 1, Section S13",
         f"_results/tables/{_name}.tex (same table; typed references into the paper re-resolved from main.aux)",
         "results-stage table of Online Resource 1")

# ================================================================================================ TABLES_MAP.md
lines = ["# TABLES_MAP: generated tables (make_tables_fodm.py)", "",
         "Generated by `paper2_fodm/make_tables_fodm.py` from the T10 outputs (`experiments/results/t10/`), "
         "`paper2_fodm/_results/*.csv` and, for avg2 and size-weighted averaging (seeds 0-9), "
         "`experiments/results/t6_baselines/per_seed_metrics__mstar.csv`, with the frozen statistics of "
         "`experiments/scripts/T10_review.py` (sha256 checked against `ANALYSIS_FROZEN`). Do not edit the .tex files "
         "by hand; re-run the script.", "",
         "Insertion. Main text: put a line `%%INSERT tables_gen/<name>.tex` in the src chunk where the table belongs "
         "(flatten.py replaces it), and delete the inline table it replaces. Online Resource 1: paste the file content "
         "into ESM_1.tex (single file; do not run esm/build_esm.py).", "",
         "Marks used by the generated tables (state them once in Section 6.9 'Table conventions'): `*` p<0.05 exact "
         "test unadjusted; dagger = Holm-adjusted p<0.05 over the comparison's configuration family; `(n)` after a "
         "value = effective sample size n' when below the number of seeds; circle (tab:scffcm) = n'<=5, test cannot "
         "reach p<0.05; section sign = second look at seeds 10-19 (L=1 or SC-FFCM contrasts); counts `u (H)` = "
         "unadjusted (Holm).", "",
         "| file | label(s) | where | replaces | content |", "|---|---|---|---|---|"]
for g in GENERATED:
    lines.append(f"| `tables_gen/{g['name']}.tex` | `{g['label']}` | {g['where']} | {g['replaces']} | {g['summary']} |")
lines += ["", "## Notes for the writing stage", "",
          "- Main text: tab:audit (Sec. 3), tab:confirm (Sec. 7.1), tab:gates-interaction (Sec. 7.2), tab_L1 (Sec. 7.3), "
          "tab:decision (Sec. 8.1). Online Resource 1, Section S14 (the former Appendix A): tabS_substitution, tabS_scffcm, "
          "tabS_baselines, tabS_gfgain, tabS_fuzziness.",
          "- tab:confirm replaces the old held-out table; text in Sec. 7 that cites the old tab:confirm columns "
          "(pre_footprint_off vs GF-PFedFCM, footprint-mass at PRE) must be re-sourced or removed.",
          "- tab:substitution stays (seeds 0-19 detail); its legacy p-values give the same counts as the exact test "
          "(EVIDENCE_T10 E3: no mean-ACC count changes).",
          "- The size-weighted averaging column is the renamed 'FedNova' baseline (06_protocol, sec:protocol-baselines).",
          "- Online Resource 1 captions carry typed references (\"Section 6.9 of the paper\") resolved from main.aux, "
          "and main-text captions carry \"Online Resource 1, Table Sn\" resolved from ESM_1.aux. After inserting the "
          "tables and rebuilding both documents, re-run `python3 make_tables_fodm.py` and rebuild so that these "
          "numbers are current.", ""]
if WARN:
    lines += ["## Unresolved cross-document references at the last run", ""]
    lines += [f"- {w}" for w in sorted(set(WARN))]
    lines += [""]
n_re = sum(1 for c in CHECKS if c.startswith("recount of "))
n_md = sum(1 for c in CHECKS if c.startswith("EVIDENCE_T10_extra.md lists"))
other = [c for c in CHECKS if not c.startswith("recount of ") and not c.startswith("EVIDENCE_T10_extra.md lists")]
lines += ["## Compile test", "",
          "The tables are inserted by flatten.py (main.tex) and flatten_esm.py (ESM_1.tex) through %%INSERT lines; "
          "both documents are built with latexmk (see _results/ESM_XREF.md for the build order).",
          "", "## Consistency checks passed at the last run", "",
          f"- {n_re} recounts of unadjusted and Holm counts from the per-configuration test rows (frozen summaries and "
          f"EVIDENCE_T10_extra.md), over {n_md} count lines of EVIDENCE_T10_extra.md"] + [f"- {c}" for c in other] + [""]
(OUT / "TABLES_MAP.md").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(GENERATED)} files written to {OUT}; {len(CHECKS)} checks passed; {len(set(WARN))} unresolved references")
for w in sorted(set(WARN)):
    print("  WARN", w)
