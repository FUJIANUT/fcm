"""R2b analysis (EXTENDED step grid for SC-FFCM; otherwise identical to T9_L1_scffcm_analysis.py). (R2) a FAIR SC-FFCM, re-calibrated per (config, L) at m* on a non-evaluation seed with a
label-free criterion, versus pre_mass_off at the SAME L on the SAME seeds. (R1) L=1 on the held-out seeds 10-19 and
GF-PFedFCM at L=1, so that the mass-point effect is separated from the effect of L (same-L comparisons)."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
RES = Path(__file__).resolve().parents[1] / "results"; T3 = RES / "t3_design_space"
DS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
      "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
AT2 = set(DS[:8])
A = collections.defaultdict(dict)   # key: (tag, dataset, name) -> {seed: acc}
def load_t3(f, tag, keep=None):
    for r in csv.DictReader(open(T3 / f)):
        if keep is None or r["dataset"] in keep: A[(tag, r["dataset"], r["cell"])][int(r["seed"])] = float(r["mean_acc"])
load_t3("per_seed_metrics_main.csv", "L5", AT2); load_t3("per_seed_metrics_big.csv", "L5", AT2)
load_t3("per_seed_metrics__mstar.csv", "L5", set(DS) - AT2); load_t3("per_seed_metrics__confirm.csv", "L5")
load_t3("per_seed_metrics__L1all.csv", "L1"); load_t3("per_seed_metrics__T9_L1_gf.csv", "L1"); load_t3("per_seed_metrics__T9_L1_confirm.csv", "L1")
for r in csv.DictReader(open(RES / "t9b_scffcm_fair.csv")):
    if r["mean_acc"] != "nan": A[(f"L{r['L']}", r["dataset"], "scffcm_fair")][int(r["seed"])] = float(r["mean_acc"])
for r in csv.DictReader(open(RES / "t6_baselines" / "per_seed_metrics__mstar.csv")):
    if r["method"] == "scffcm": A[("L5", r["dataset"], "scffcm_old")][int(r["seed"])] = float(r["mean_acc"])
def cmp(tagA, a, tagB, b, seeds, title):
    w = l = 0; det = []
    for ds in DS:
        X, Y = A[(tagA, ds, a)], A[(tagB, ds, b)]; ks = [k for k in seeds if k in X and k in Y]
        if len(ks) < 5: det.append(f"{ds[:10]}:--"); continue
        x = [X[k] for k in ks]; y = [Y[k] for k in ks]; d = st.mean(x) - st.mean(y)
        try: p = None if all(abs(i-j) < 1e-12 for i, j in zip(x, y)) else wilcoxon(x, y).pvalue
        except Exception: p = None
        s = p is not None and p < 0.05; w += s and d > 0; l += s and d < 0; det.append(f"{ds[:10]}:{d:+.3f}{'*' if s else ''}")
    print(f"--- {title}: better {w}, worse {l}\n     " + "  ".join(det))
S0, S1, SA = range(10), range(10, 20), range(20)
print("R2. FAIR SC-FFCM (per-config, per-L label-free calibration on a non-evaluation seed)\n")
cmp("L5", "scffcm_fair", "L5", "scffcm_old", S0, "recalibrated SC-FFCM (L=5) vs the previous SC-FFCM (L=5), seeds 0-9")
for L in ("L1", "L5"):
    cmp(L, "pre_mass_off", L, "scffcm_fair", S0, f"pre_mass_off vs fair SC-FFCM, both {L}, seeds 0-9 (exploratory)")
    cmp(L, "pre_mass_off", L, "scffcm_fair", S1, f"pre_mass_off vs fair SC-FFCM, both {L}, seeds 10-19 (held-out)")
    cmp(L, "pre_none_off", L, "scffcm_fair", S1, f"pre_none_off vs fair SC-FFCM, both {L}, seeds 10-19 (held-out)")
cmp("L1", "pre_mass_off", "L50", "scffcm_fair", S0, "pre_mass_off (L=1) vs fair SC-FFCM at its default L=50, seeds 0-9 (12 non-MNIST configs)")
cmp("L1", "scffcm_fair", "L5", "scffcm_fair", S1, "fair SC-FFCM at L=1 vs L=5, seeds 10-19")
print("\nR1. SAME-L comparisons and the held-out L=1 block\n")
cmp("L1", "pre_mass_off", "L1", "post_footprint_on", S0, "pre_mass_off vs GF-PFedFCM, both L=1, seeds 0-9")
cmp("L1", "pre_mass_off", "L1", "post_footprint_on", S1, "pre_mass_off vs GF-PFedFCM, both L=1, seeds 10-19 (held-out)")
cmp("L1", "pre_none_off", "L1", "post_none_off", S1, "PRE vs POST mass (no gate), both L=1, seeds 10-19 (held-out)")
cmp("L1", "pre_mass_off", "L5", "pre_mass_off", S1, "pre_mass_off at L=1 vs L=5, seeds 10-19 (held-out)")
cmp("L1", "post_footprint_on", "L5", "post_footprint_on", S1, "GF-PFedFCM at L=1 vs L=5, seeds 10-19 (held-out)")
cmp("L1", "post_footprint_on", "L1", "post_footprint_off", S1, "personalization at L=1 (GF vs gate-only), seeds 10-19")
print("\nEACH METHOD AT ITS OWN BEST LOCAL-STEP COUNT (extended-grid SC-FFCM)")
cmp("L1", "pre_mass_off", "L5", "scffcm_fair", S1, "pre_mass_off (L=1) vs fair SC-FFCM (L=5), seeds 10-19 (held-out)")
cmp("L1", "pre_mass_off", "L5", "scffcm_fair", S0, "pre_mass_off (L=1) vs fair SC-FFCM (L=5), seeds 0-9")
cmp("L1", "pre_none_off", "L5", "scffcm_fair", S1, "pre_none_off (L=1, lossless rule) vs fair SC-FFCM (L=5), seeds 10-19 (held-out)")
