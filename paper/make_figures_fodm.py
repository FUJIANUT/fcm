"""Data-generated figures of the FODM manuscript (vector PDF into figures/). Never edit the PDFs by hand; re-run:
    python3 make_figures_fodm.py            # all figures
    python3 make_figures_fodm.py motivation # one figure
Sources (read only): the frozen T10 analysis outputs (experiments/results/t10/analysis_final_*.csv), the results-stage test
table (_results/results_stage_tests.csv, written by _results/results_stage_tables.py with the frozen test of
scripts/T10_review.py), the theory checks (results/t10/theory/E_I_descent.csv) and, for the motivation figure only, a
re-run of the design-space harness (scripts/T3_design_space.py, unchanged) on one seed of one synthetic configuration;
every re-run is checked bit for bit against the stored per-seed CSV before it is drawn.
Palette (validated with the dataviz validator, all-pairs, light surface): PRE / ours-in-focus = blue #2a78d6, POST =
orange #eb6834, SC-FFCM = aqua #1baf7a; everything else neutral gray. Every series also differs by marker or line style."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
T10 = EXP / "results" / "t10"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

import matplotlib as mpl  # noqa: E402

mpl.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

TW = 372.0 / 72.27  # \textwidth of sn-jnl in inches
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, LIGHT, MID = "#e4e3df", "#d9d8d3", "#b4b3ad"
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

NAME = {"cluster_skew_hard": "Cluster skew", "cluster_skew_overlap": "Cluster skew (overlap)",
        "dirichlet_0.03": "Dirichlet 0.03", "dirichlet_0.1": "Dirichlet 0.1", "overlap_noise": "Overlap/noise",
        "quantity_skew_extreme": "Quantity skew", "wine": "Wine", "satimage": "Satimage", "pendigits": "Pendigits",
        "digits_pca16": "Digits PCA-16", "digits_pca32": "Digits PCA-32", "letter": "Letter",
        "mnist784_pca32 (20c)": "MNIST PCA-32 (20 cl.)", "mnist784_pca32 (50c)": "MNIST PCA-32 (50 cl.)"}
CONFIGS = list(NAME)
CHECKS: list[str] = []


def check(ok: bool, what: str) -> None:
    CHECKS.append(("PASS " if ok else "FAIL ") + what)
    if not ok:
        raise SystemExit("FAILED check: " + what)


def recessive_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")



# ============================================================================================ 0 degeneracy (Fig. 2)
DEG_ORDER = ["wine", "satimage", "pendigits", "digits_pca16", "letter", "digits_pca32", "mnist784_pca32 (20c)",
             "mnist784_pca32 (50c)"]
DEG_MSTAR = {"wine": 2.0, "satimage": 2.0, "pendigits": 1.5, "digits_pca16": 1.3, "letter": 1.3, "digits_pca32": 1.1,
             "mnist784_pca32 (20c)": 1.1, "mnist784_pca32 (50c)": 1.1}


def fig_degeneracy():
    """Centralized FCM on the pooled data of each real configuration over the audit grid (5 audit seeds);
    m* recomputed by the paper's rule and checked against Table 2 (scripts/T3_design_space.py MSTAR)."""
    d = pd.read_csv(HERE.parent / "paper2" / "figures" / "degeneracy_data.csv", float_precision="round_trip")
    grid = [2.0, 1.5, 1.3, 1.2, 1.1]
    fig, axes = plt.subplots(2, 4, figsize=(TW, 0.60 * TW), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.62, wspace=0.12, left=0.085, right=0.995, top=0.86, bottom=0.12))
    x = np.arange(len(grid))
    out = {}
    for ax, ds in zip(axes.ravel(), DEG_ORDER):
        sub = d[d.dataset == ds]
        c = int(sub.c.iloc[0])
        ok = [m for m in grid if (sub[sub.m == m].distinct == c).all() and len(sub[sub.m == m]) == 5]
        mstar = max(ok)
        check(mstar == DEG_MSTAR[ds], f"degeneracy: m* of {ds} recomputed = {mstar}")
        fr = [sub[sub.m == m].distinct / c for m in grid]
        en = [sub[sub.m == m].entropy_norm for m in grid]
        ax.axvspan(grid.index(mstar) - 0.38, grid.index(mstar) + 0.38, color=GRID, zorder=0, linewidth=0)
        ax.plot(x, [v.mean() for v in fr], color=BLUE, marker="o", ms=3.4, lw=1.1, zorder=3)
        ax.vlines(x, [v.min() for v in fr], [v.max() for v in fr], color=BLUE, lw=0.7, zorder=2)
        ax.plot(x, [v.mean() for v in en], color=ORANGE, marker="s", ms=3.0, lw=1.0, ls="--", zorder=3)
        head, _, tail = NAME[ds].partition(" (")
        ax.set_title(head + "\n" + (f"({tail}, " if tail else "") + rf"$c={c}$", fontsize=7.0, loc="left",
                     linespacing=1.1)
        ax.set_xticks(x, [f"{m:.1f}" for m in grid])
        ax.set_ylim(-0.04, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        recessive_grid(ax, "y")
        out[ds] = (mstar, round(float(fr[0].mean()), 3), round(float(en[0].mean()), 3))
    for ax in axes[1]:
        ax.set_xlabel(r"fuzzifier $m$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"normalized value", fontsize=7)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.4, lw=1.1, label=r"distinct prototypes$/c$ (mean, range)"),
                        Line2D([], [], color=ORANGE, marker="s", ms=3.0, lw=1.0, ls="--", label=r"$H/\ln c$ (mean)"),
                        Patch(color=GRID, label=r"chosen $m^*$")],
               loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=6.8)
    save(fig, "fig_degeneracy")
    return out


# ============================================================================================ 1 motivation (Fig. 1)
def fig_motivation():
    """Running example: Cluster skew (overlap), m = 2, L = 5. (a) one client's round from prototypes at the four
    generating centres; (b) PRE versus POST fuzzy mass of that client per prototype; (c, d) global prototypes over
    50 rounds of the POST and the PRE rule on the seed whose PRE-POST gain in mean client ACC is closest to the median
    gain over seeds 0-9 (ties: lower seed index)."""
    os.environ["FFCM_MSTAR"] = "1"
    for k in ("FFCM_L", "FFCM_ROUNDS", "FFCM_SEED_START", "FFCM_M", "FFCM_NO_EARLYSTOP"):
        os.environ.pop(k, None)
    sys.path.insert(0, str(EXP / "scripts"))
    sys.path.insert(0, str(EXP))
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data, predict_membership, run_local_fcm_steps

    label = "cluster_skew_overlap"
    T.M = T.fuzzifier_for(label)
    check(T.M == 2.0, "motivation: m* = 2 for Cluster skew (overlap)")
    stored = pd.read_csv(T10 / "cells16_L5_s00-09.csv", float_precision="round_trip")
    st = stored[stored.dataset == label].pivot_table(index="seed", columns="cell", values="mean_acc")
    gain = (st["pre_none_off"] - st["post_none_off"])
    med = float(np.median(gain.values))
    seed = int(sorted(gain.index, key=lambda s: (abs(gain[s] - med), s))[0])
    spec = ("synthetic", label)
    clients, k = T.build_clients(spec, seed)
    X = np.vstack([c.x for c in clients])
    Y = np.concatenate([c.y for c in clients])
    init = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    traj = {}
    for mp in ("post", "pre"):
        Vs = [T.run_cell(clients, init, mp, "none", "off", rounds=r, local_steps=5)[0] for r in range(0, 51)]
        acc = T.client_accs(clients, [Vs[-1]] * len(clients))[0]
        check(acc == float(st.loc[seed, f"{mp}_none_off"]),
              f"motivation: re-run {mp}_none_off seed {seed} equals stored mean_acc bit for bit")
        traj[mp] = (np.array(Vs), acc)
    centres = np.array([[-2.6, -2.2], [-0.9, 2.3], [2.3, -1.8], [2.8, 2.3]])  # generator of this scenario
    # (a, b): client 0 from prototypes at the generating centres
    p = 0
    xp, yp = clients[p].x, clients[p].y
    ubar = predict_membership(xp, centres, m=T.M)
    pre_mass = (ubar ** T.M).sum(0) / len(xp)
    loc = run_local_fcm_steps(xp, centres, steps=5, m=T.M)
    post_mass = (loc.membership ** T.M).sum(0) / len(xp)
    share = np.bincount(yp, minlength=k) / len(yp)

    fig = plt.figure(figsize=(TW, 0.95 * TW))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.42)
    box = dict(frameon=True, framealpha=0.9, facecolor="white", edgecolor="none")
    lim = (-7.2, 7.2)
    # (a)
    ax = fig.add_subplot(gs[0, 0])
    others = np.vstack([c.x for i, c in enumerate(clients) if i != p])
    ax.scatter(others[:, 0], others[:, 1], s=1.2, color=LIGHT, linewidths=0, rasterized=False, zorder=1)
    ax.scatter(xp[:, 0], xp[:, 1], s=2.5, color=INK2, linewidths=0, zorder=2)
    for j in range(k):
        ax.annotate("", xy=loc.centers[j], xytext=centres[j],
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, shrinkA=2.5, shrinkB=2.5), zorder=4)
    ax.scatter(centres[:, 0], centres[:, 1], s=34, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=5,
               label=r"broadcast $v_j$")
    ax.scatter(loc.centers[:, 0], loc.centers[:, 1], s=26, marker="^", color=ORANGE, edgecolor="white",
               linewidths=0.5, zorder=6, label=r"local centre $c_{pj}$ after $L=5$")
    for j in range(k):
        ax.text(centres[j, 0] + 0.35, centres[j, 1] + 0.45, rf"$v_{j + 1}$", fontsize=7, color=INK, zorder=7)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xticks([-6, 0, 6]); ax.set_yticks([-6, 0, 6])
    ax.set_title(r"(a) One client, one round ($L=5$)", loc="left", fontsize=7.8)
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2, fontsize=6.3, **box)
    # (b)
    ax = fig.add_subplot(gs[0, 1])
    idx = np.arange(k)
    w = 0.36
    b1 = ax.bar(idx - w / 2, pre_mass, width=w, color=BLUE, label=r"PRE, $\bar M_{pj}/N_p$")
    b2 = ax.bar(idx + w / 2, post_mass, width=w, color=ORANGE, hatch="////", edgecolor="white", linewidth=0,
                label=r"POST, $M^{L}_{pj}/N_p$")
    ax.set_xticks(idx, [rf"$v_{j + 1}$ ({100 * share[j]:.0f}\%)" for j in range(k)])
    ax.set_xlabel(r"prototype (share of the client's points)", fontsize=7)
    ax.set_ylim(0, 0.70)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel(r"fuzzy mass per point")
    recessive_grid(ax, "y")
    ax.set_title(r"(b) Fuzzy mass of this client", loc="left", fontsize=7.8)
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4, **box)
    # (c, d)
    for col, (mp, colr, mk, title) in enumerate([("post", ORANGE, "^", "(c) POST mass"),
                                                 ("pre", BLUE, "o", "(d) PRE mass")]):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(X[:, 0], X[:, 1], s=1.0, color=LIGHT, linewidths=0, zorder=1)
        Vs, acc = traj[mp]
        for j in range(k):
            ax.plot(Vs[:, j, 0], Vs[:, j, 1], color=colr, lw=0.8, alpha=0.9, zorder=3)
        ax.scatter(Vs[0, :, 0], Vs[0, :, 1], s=14, facecolor="white", edgecolor=MUTED, linewidths=0.8, zorder=4,
                   label="initial prototypes")
        ax.scatter(Vs[-1, :, 0], Vs[-1, :, 1], s=30, marker=mk, color=colr, edgecolor="white", linewidths=0.5,
                   zorder=5, label="after 50 rounds")
        ax.scatter(centres[:, 0], centres[:, 1], s=36, marker="+", color=INK, linewidths=1.0, zorder=6,
                   label="generating centres")
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xticks([-6, 0, 6]); ax.set_yticks([-6, 0, 6])
        ax.set_title(rf"{title} after 50 rounds" + "\n" + rf"mean client accuracy ${acc:.3f}$", loc="left",
                     fontsize=7.8)
        if col == 0:
            ax.legend(loc="upper left", fontsize=6.2, handletextpad=0.2, borderaxespad=0.2, **box)
    save(fig, "fig_motivation")
    return dict(seed=seed, median_gain=med, post=traj["post"][1], pre=traj["pre"][1],
                pre_mass=pre_mass.tolist(), post_mass=post_mass.tolist(), share=share.tolist(),
                mean_post=float(st["post_none_off"].mean()), mean_pre=float(st["pre_none_off"].mean()),
                gains=gain.round(3).to_dict())


# ============================================================================================ 2 confirmation heatmap
HYP = [("H1", r"$\mathrm{PM}_5{-}\mathrm{GF}_5$"), ("H2", r"$\mathrm{PM}_5{-}\mathrm{F}_5$"),
       ("H3", r"$D$, footprint"), ("H4", r"$\mathrm{PRE}_1{-}\mathrm{POST}_1$"),
       ("H5", r"$\mathrm{PM}_1{-}\mathrm{GF}_1$"), ("H6", r"$\mathrm{PM}_1{-}\mathrm{SC}_1$"),
       ("H7", r"$\mathrm{PM}_1{-}\mathrm{SC}_5$"), ("H8", r"$\mathrm{PM}_5{-}\mathrm{SC}_5$"),
       ("H9", r"$\mathrm{GF}_5{-}\mathrm{GO}_5$")]
EXPECT = {"H1": "+", "H2": "+", "H3": "+", "H4": "+", "H5": "+", "H6": "+", "H7": "+", "H8": "-", "H9": r"$\le 0$"}


def diverging_cmap():
    return LinearSegmentedColormap.from_list("bwr_viz", ["#e34948", "#f0efec", BLUE])


def fig_confirm():
    t = pd.read_csv(T10 / "analysis_final_tests.csv")
    s = pd.read_csv(T10 / "analysis_final_summary.csv")
    t = t[(t.block == "s20-29") & (t.metric == "mean_acc")]
    M = np.full((len(CONFIGS), len(HYP)), np.nan)
    P = np.ones_like(M)
    H = np.ones_like(M)
    for i, c in enumerate(CONFIGS):
        for j, (h, _) in enumerate(HYP):
            r = t[(t.hyp == h) & (t.dataset == c)]
            check(len(r) == 1, f"confirm: one test row for {h} {c}")
            r = r.iloc[0]
            M[i, j] = r.mean_diff_raw
            P[i, j] = r.p_exact if not math.isnan(r.p_exact) else 1.0
            H[i, j] = r.p_holm if not math.isnan(r.p_holm) else 1.0
    vmax = 0.15
    fig, ax = plt.subplots(figsize=(0.885 * TW, 0.58 * TW))
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap=diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            sv = f"{v:+.3f}"
            sv = "0" if abs(v) < 5e-4 else (sv[0] + sv[1:].lstrip("0"))
            sup = r"^{\dagger}" if H[i, j] < 0.05 else (r"^{*}" if P[i, j] < 0.05 else "")
            txt = rf"${sv}{sup}$"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                    color="white" if abs(v) > 0.10 else INK)
    ax.set_xticks(range(len(HYP)), [f"{h}\n({EXPECT[h]})" for h, lab in HYP], fontsize=6.8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(CONFIGS)), [NAME[c] for c in CONFIGS], fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y in (5.5, 7.5):
        ax.axhline(y, color="white", lw=2.0)
    # bottom row: counts better/worse (Holm) and replication flag
    ss = s[(s.block == "s20-29") & (s.metric == "mean_acc")].set_index("hyp")
    for j, (h, _) in enumerate(HYP):
        r = ss.loc[h]
        flag = r"\checkmark" if int(r.replicates) == 1 else r"$\times$"
        ax.text(j, len(CONFIGS) - 0.3, f"{int(r.better_unadj)}/{int(r.worse_unadj)}\n"
                f"({int(r.better_holm)}/{int(r.worse_holm)})\n{flag}", ha="center", va="top", fontsize=6.4,
                color=INK, linespacing=1.15)
    ax.text(-0.62, len(CONFIGS) - 0.3, "better/worse\n(Holm)\nrule met", ha="right", va="top", fontsize=6.4,
            color=INK2, linespacing=1.15)
    ax.set_ylim(len(CONFIGS) + 1.75, -0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(r"mean difference $a-b$, mean client ACC", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.3, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig_confirm")
    return {h: (int(ss.loc[h].better_unadj), int(ss.loc[h].worse_unadj), int(ss.loc[h].better_holm),
                int(ss.loc[h].worse_holm), int(ss.loc[h].replicates)) for h, _ in HYP}


# ============================================================================================ 3 gate interaction
GATES = [("EA-footprint", "footprint gate", BLUE, "o", True), ("EA-mass", "mass gate", BLUE, "s", True),
         ("EA-fpmask", "footprint mask", ORANGE, "o", False), ("EA-massmask", "mass mask", ORANGE, "s", False)]


GATE_COUNTS: dict = {}


def fig_gates():
    ea = pd.read_csv(T10 / "analysis_final_ea.csv")
    fig = plt.figure(figsize=(TW, 0.55 * TW))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], wspace=0.30, hspace=0.62, left=0.225, right=0.975,
                          top=0.89, bottom=0.115)
    ax = fig.add_subplot(gs[:, 0])
    sub = ea[(ea.block == "s20-29") & (ea.metric == "mean_acc")]
    off = {g: o for g, o in zip([g[0] for g in GATES], (-0.27, -0.09, 0.09, 0.27))}
    for g, lab, col, mk, filled in GATES:
        ys, xs, lo, hi, sig = [], [], [], [], []
        for i, c in enumerate(CONFIGS):
            r = sub[(sub.hyp == g) & (sub.dataset == c)]
            check(len(r) == 1, f"gates: one row {g} {c}")
            r = r.iloc[0]
            ys.append(i + off[g]); xs.append(r.mean_diff_raw); lo.append(r.ci95_lo); hi.append(r.ci95_hi)
            sig.append(r.p_holm < 0.05)
        xs, lo, hi = np.array(xs), np.array(lo), np.array(hi)
        ax.hlines(ys, lo, hi, color=col, lw=0.7, alpha=0.85, zorder=2)
        for y, x, sgn in zip(ys, xs, sig):
            ax.scatter([x], [y], s=16, marker=mk, zorder=3, linewidths=0.8,
                       facecolor=(col if sgn else "white"), edgecolor=col)
        ax.scatter([], [], s=16, marker=mk, facecolor=col, edgecolor=col, label=lab)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(CONFIGS)), [NAME[c] for c in CONFIGS], fontsize=6.8)
    ax.set_ylim(len(CONFIGS) - 0.5, -0.5)
    ax.set_xlabel(r"interaction $D$ in mean client ACC, seeds 20--29")
    recessive_grid(ax, "x")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.4, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0.2)
    ax.text(-0.42, 1.10, "(a)", transform=ax.transAxes, fontsize=8, va="bottom")
    # (b) Holm counts of significantly positive (and negative) interactions per gate, metric and seed block
    blocks = ["s00-09", "s10-19", "s20-29"]
    for row, (metric, mlab) in enumerate((("mean_acc", "Mean client ACC"), ("fcm_obj", "FCM objective"))):
        ax = fig.add_subplot(gs[row, 1])
        for gi, (g, lab, col, mk, filled) in enumerate(GATES):
            for bi, b in enumerate(blocks):
                r = ea[(ea.block == b) & (ea.metric == metric) & (ea.hyp == g)]
                n_pos = int(((r.p_holm < 0.05) & (r.mean_diff_oriented > 0)).sum())
                n_neg = int(((r.p_holm < 0.05) & (r.mean_diff_oriented < 0)).sum())
                GATE_COUNTS[(b, metric, g)] = (n_pos, n_neg)
                x = gi + (bi - 1) * 0.26
                ax.bar(x, n_pos, width=0.23, color=col, alpha=[0.45, 0.7, 1.0][bi], linewidth=0)
                if n_neg:
                    ax.bar(x, -n_neg, width=0.23, color=MUTED, linewidth=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(range(len(GATES)), ["footprint\ngate", "mass\ngate", "footprint\nmask", "mass\nmask"],
                      fontsize=6.4, linespacing=1.0)
        ax.set_ylim(-2, 14.5)
        ax.set_yticks([0, 7, 14])
        ax.set_ylabel(r"configurations (of 14)", fontsize=6.8)
        recessive_grid(ax, "y")
        ax.text(0.0, 1.03, rf"({'bc'[row]}) {mlab}", transform=ax.transAxes, fontsize=7.0, va="bottom")
        if row == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=INK2, alpha=a, label=l) for a, l in
                               zip((0.45, 0.7, 1.0), (r"seeds 0--9", r"10--19", r"20--29"))],
                      loc="upper right", fontsize=6.0, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    save(fig, "fig_gates")
    # cross-check against the frozen report (E-A section of analysis_final.txt: "positive x/negative y | hx/hy")
    import re
    txt = (T10 / "analysis_final.txt").read_text()
    for (b, metric, g), (n_pos, n_neg) in GATE_COUNTS.items():
        m_ = re.search(rf"^{b} {g}\s+{metric}\s+tested\s+\d+: positive \d+/negative \d+ \| (\d+)/(\d+)", txt, re.M)
        check(m_ is not None and (int(m_.group(1)), int(m_.group(2))) == (n_pos, n_neg),
              f"gates: Holm counts {b} {metric} {g} = {n_pos}/{n_neg} match analysis_final.txt")
    return GATE_COUNTS


# ============================================================================================ 4 descent
RULES = [("pre_none_off_L1", r"ungated PRE, $L=1$", BLUE, "-"), ("pre_mass_off_L1", r"mass-gated PRE, $L=1$", INK2, "--"),
         ("pre_none_off_L5", r"ungated PRE, $L=5$", ORANGE, ":")]
SHORT = {"pre_none_off_L1": r"ungated, $L{=}1$", "pre_mass_off_L1": r"gated, $L{=}1$", "pre_none_off_L5": r"ungated, $L{=}5$"}


def fig_descent():
    d = pd.read_csv(T10 / "theory" / "E_I_descent.csv")
    fig = plt.figure(figsize=(TW, 0.44 * TW))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    lin = 1e-7
    for rule, lab, col, ls in RULES:
        rr = d[(d.rule == rule) & (d.dataset == "cluster_skew_overlap")]
        for si, (_, r) in enumerate(rr.iterrows()):
            J = np.array([float(v) for v in r.J_traj.split(";")])
            rel = np.diff(J) / J[:-1]
            ax.plot(np.arange(1, len(J)), rel, color=col, ls=ls, lw=0.8, alpha=0.9,
                    label=lab if si == 0 else None)
    ax.set_yscale("symlog", linthresh=lin, linscale=0.6)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel(r"round $t$")
    ax.set_ylabel(r"$(f_\beta(V_{t})-f_\beta(V_{t-1}))/f_\beta(V_{t-1})$")
    ax.set_yticks([-1e-1, -1e-3, -1e-5, 0, 1e-5, 1e-3])
    recessive_grid(ax, "y")
    ax.set_title(r"(a) Cluster skew (overlap), seeds 0--4", loc="left", fontsize=7.6)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
               fontsize=6.8, handlelength=2.4)
    ax = fig.add_subplot(gs[0, 1])
    for k_, (rule, lab, col, ls) in enumerate(RULES):
        rr = d[d.rule == rule]
        v = rr.max_rel_increase.values.astype(float)
        n_inc = int((rr.n_rounds_increase_gt1e12 > 0).sum())
        floor = 1e-13
        vv = np.where(v > 1e-12, v, floor)
        jit = (np.arange(len(vv)) % 7 - 3) * 0.045
        ax.scatter(vv, k_ + jit, s=7, color=col, marker=["o", "s", "^"][k_], linewidths=0, alpha=0.85)
        ax.text(3e-1, k_, rf"{n_inc}/{len(rr)}", va="center", ha="left", fontsize=6.8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(5e-14, 1.0)
    ax.axvline(1e-12, color=MUTED, lw=0.6)
    ax.text(1.3e-12, 2.42, r"$10^{-12}$", fontsize=6.2, color=INK2)
    ax.set_yticks(range(3), [SHORT[r] for r, *_ in RULES], fontsize=6.6)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel(r"largest relative increase of $f_\beta$ in a run")
    recessive_grid(ax, "x")
    ax.set_title(r"(b) 14 configurations $\times$ 5 seeds", loc="left", fontsize=7.6)
    save(fig, "fig_descent")
    return {rule: int((d[d.rule == rule].n_rounds_increase_gt1e12 > 0).sum()) for rule, *_ in RULES}


# ============================================================================================ 5 trade-off
def fig_tradeoff():
    """pre_mass_off against pre_none_off (both L = 1) on the fresh seeds, per configuration: (a) relative change of OBJ,
    averaged over the seeds (the stored absolute differences are checked against the test table); (b) change of mean
    client ACC with its 95% t-interval. Paired dot plots in the configuration order of Fig. 6."""
    t = pd.read_csv(HERE / "_results" / "results_stage_tests.csv")
    mg = t[(t.hyp == "MG-L1") & (t.block == "s20-29")]
    acc = mg[mg.metric == "mean_acc"].set_index("dataset")
    obj = mg[mg.metric == "fcm_obj"].set_index("dataset")
    f = pd.read_csv(T10 / "FRESH_cells16_L1_s20-29.csv", float_precision="round_trip")
    f = f[f.cell.isin(["pre_mass_off", "pre_none_off"])]
    po = f.pivot_table(index=["dataset", "seed"], columns="cell", values="fcm_obj")
    pa = f.pivot_table(index=["dataset", "seed"], columns="cell", values="mean_acc")
    rel = ((po["pre_mass_off"] - po["pre_none_off"]) / po["pre_none_off"]).groupby(level=0).mean()
    absd = (po["pre_mass_off"] - po["pre_none_off"]).groupby(level=0).mean()
    dacc = (pa["pre_mass_off"] - pa["pre_none_off"]).groupby(level=0).mean()
    for c in CONFIGS:
        check(abs(absd[c] - obj.loc[c].mean_diff_raw) < 1e-12, f"tradeoff: absolute OBJ difference {c} matches the test table")
        check(abs(dacc[c] - acc.loc[c].mean_diff_raw) < 1e-12, f"tradeoff: ACC difference {c} matches the test table")
        check(rel[c] > 0, f"tradeoff: relative OBJ change positive on {c} (log axis)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TW, 0.50 * TW), sharey=True,
                                   gridspec_kw=dict(wspace=0.10, left=0.225, right=0.975, top=0.83, bottom=0.14))
    pts = {}
    for i, c in enumerate(CONFIGS):
        x, so = float(rel[c]), bool(obj.loc[c].p_holm < 0.05)
        y, sa = float(acc.loc[c].mean_diff_raw), bool(acc.loc[c].p_holm < 0.05)
        pts[c] = (x, y, so, sa)
        ax1.scatter([x], [i], s=18, marker="o", linewidths=0.9, edgecolor=BLUE, facecolor=BLUE if so else "white",
                    zorder=3)
        ax2.hlines(i, acc.loc[c].ci95_lo, acc.loc[c].ci95_hi, color=BLUE, lw=0.7, alpha=0.85, zorder=2)
        ax2.scatter([y], [i], s=18, marker="o", linewidths=0.9, edgecolor=BLUE, facecolor=BLUE if sa else "white",
                    zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlim(1e-4, 0.5)
    ax1.set_xticks([1e-4, 1e-3, 1e-2, 1e-1], [r"$0.01\%$", r"$0.1\%$", r"$1\%$", r"$10\%$"])
    ax1.minorticks_off()
    ax1.set_yticks(range(len(CONFIGS)), [NAME[c] for c in CONFIGS], fontsize=6.8)
    ax1.set_ylim(len(CONFIGS) - 0.5, -0.5)
    ax1.set_xlabel(r"relative change of OBJ (log scale)")
    ax1.set_title(r"(a) FCM objective", loc="left", fontsize=7.6)
    ax2.axvline(0, color=INK2, lw=0.6)
    ax2.set_xlabel(r"change of mean client ACC")
    ax2.set_title(r"(b) Mean client ACC", loc="left", fontsize=7.6)
    ax2.tick_params(axis="y", length=0)
    for ax in (ax1, ax2):
        recessive_grid(ax, "x")
    fig.legend(handles=[plt.scatter([], [], s=18, facecolor=BLUE, edgecolor=BLUE, label=r"Holm-significant change"),
                        plt.scatter([], [], s=18, facecolor="white", edgecolor=BLUE, label=r"not significant after Holm")],
               loc="lower center", bbox_to_anchor=(0.595, 0.93), ncol=2, fontsize=6.6, handletextpad=0.2)
    save(fig, "fig_tradeoff")
    i_abs = int(np.argmax([absd[c] for c in CONFIGS]))
    print("tradeoff: largest absolute OBJ rise", CONFIGS[i_abs], round(float(absd[CONFIGS[i_abs]]), 3),
          "| relative range", round(float(rel.min()), 4), round(float(rel.max()), 4))
    return {NAME[c]: (round(v[0], 4), round(v[1], 3), v[2], v[3]) for c, v in pts.items()}


# ============================================================================================ 6 decision regret
ACTS = [("F-FCM (post_none_off, L=5)", r"F-FCM, $L=5$"), ("GF-PFedFCM (L=5)", r"GF-PFedFCM, $L=5$"),
        ("ungated PRE (L=1)", r"ungated PRE, $L=1$"), ("pre_mass_off (L=1)", r"\texttt{pre\_mass\_off}, $L=1$"),
        ("pre_mass_off (L=5)", r"\texttt{pre\_mass\_off}, $L=5$"), ("SC-FFCM (L=5)", r"SC-FFCM, $L=5$"),
        ("SC-FFCM (L=1, R=50)", r"SC-FFCM, $L=1$")]
CRIT = [("mean_acc", "mean\nACC"), ("worst_acc", "worst\nACC"), ("gacc", "gACC"),
        ("fcm_obj", "OBJ\n(rel.)"), ("upload_numbers", "uploads\n(rel.)")]


def fig_regret():
    """Mean and maximum regret of the seven actions (seeds 10-29) in the post hoc analysis with a common stopping rule
    and calibrated server steps (variant C of experiments/scripts/T11_posthoc.py). Its variant 'T10' is checked against
    the frozen pre-specified analysis first."""
    allv = pd.read_csv(T10.parent / "t11_posthoc" / "decision_all.csv")
    frozen = pd.read_csv(T10 / "analysis_final_decision.csv")
    t10v = allv[(allv.variant == "T10") & (allv.block == "s10-29")]
    for _, r in frozen.iterrows():
        q = t10v[(t10v.criterion == r.criterion) & (t10v.action == r.action)].iloc[0]
        check(abs(q.mean_regret - r.mean_regret) < 1e-12 and abs(q.max_regret - r.max_regret) < 1e-12,
              f"regret: T11 variant T10 reproduces the frozen analysis ({r.criterion}, {r.action})")
    dec = allv[(allv.variant == "C") & (allv.block == "s10-29")]
    seq = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(TW, 0.48 * TW), sharey=True)
    out = {}
    for ax, (col, rec, title) in zip(axes, [("mean_regret", "rec_mean_regret", "(a) Mean regret (Bayes, uniform prior)"),
                                            ("max_regret", "rec_max_regret", "(b) Maximum regret (minimax)")]):
        M = np.zeros((len(ACTS), len(CRIT)))
        R = np.zeros_like(M, dtype=bool)
        for j, (cr, _) in enumerate(CRIT):
            for i, (a, _) in enumerate(ACTS):
                r = dec[(dec.criterion == cr) & (dec.action == a)]
                check(len(r) == 1, f"regret row {cr} {a}")
                M[i, j] = float(r.iloc[0][col]); R[i, j] = int(r.iloc[0][rec]) == 1
        C = M / M.max(axis=0, keepdims=True)
        ax.imshow(C, cmap=seq, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                num = f"{M[i, j]:.4f}" if M[i, j] < 1 else f"{M[i, j]:.3f}"
                ax.text(j, i, (rf"\textbf{{{num}}}" if R[i, j] else num), ha="center", va="center",
                        fontsize=5.9, color="white" if C[i, j] > 0.55 else INK)
                if R[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec=INK, lw=1.1))
        ax.set_xticks(range(len(CRIT)), [l for _, l in CRIT], fontsize=6.4)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=7.6)
        out[col] = {CRIT[j][0]: ACTS[int(np.argmin(M[:, j]))][0] for j in range(len(CRIT))}
    axes[0].set_yticks(range(len(ACTS)), [l for _, l in ACTS], fontsize=6.6)
    save(fig, "fig_regret")
    return out


FIGS = {"degeneracy": fig_degeneracy, "motivation": fig_motivation, "confirm": fig_confirm, "gates": fig_gates, "descent": fig_descent,
        "tradeoff": fig_tradeoff, "regret": fig_regret}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    info = {}
    for w in which:
        info[w] = FIGS[w]()
    for c in CHECKS:
        print(c)
    for k, v in info.items():
        print(f"[{k}]", v)
