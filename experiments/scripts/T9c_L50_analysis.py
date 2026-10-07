"""R2c analysis: SC-FFCM at its source-code default L=50, re-calibrated on the EXTENDED grid (12 non-MNIST configs),
versus pre_mass_off at L=1 and versus SC-FFCM at L=5 (extended grid). Paired Wilcoxon per configuration."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
RES = Path(__file__).resolve().parents[1] / "results"; T3 = RES / "t3_design_space"
A = collections.defaultdict(dict)
for f in ("per_seed_metrics__L1all.csv", "per_seed_metrics__T9_L1_confirm.csv"):
    for r in csv.DictReader(open(T3 / f)): A[("L1", r["dataset"], r["cell"])][int(r["seed"])] = float(r["mean_acc"])
for f, L in (("t9b_scffcm_fair.csv", None), ("t9c_scffcm_fair.csv", None)):
    for r in csv.DictReader(open(RES / f)):
        if r["mean_acc"] != "nan": A[(f"L{r['L']}", r["dataset"], "scffcm")][int(r["seed"])] = float(r["mean_acc"])
DS = [d for d in ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
      "wine","satimage","pendigits","digits_pca16","digits_pca32","letter"]]
def cmp(ta, a, tb, b, seeds, title):
    w = l = 0; det = []
    for ds in DS:
        X, Y = A[(ta, ds, a)], A[(tb, ds, b)]; ks = [k for k in seeds if k in X and k in Y]
        if len(ks) < 5: det.append(f"{ds[:10]}:--"); continue
        x = [X[k] for k in ks]; y = [Y[k] for k in ks]; d = st.mean(x) - st.mean(y)
        try: p = None if all(abs(i-j) < 1e-10 for i, j in zip(x, y)) else wilcoxon(x, y).pvalue
        except Exception: p = None
        s = p is not None and p < 0.05; w += s and d > 0; l += s and d < 0; det.append(f"{ds[:10]}:{d:+.3f}{'*' if s else ''}")
    print(f"--- {title}: better {w}, worse {l}  (12 non-MNIST configurations)\n     " + "  ".join(det))
S0, S1 = range(10), range(10, 20)
print("R2c. SC-FFCM AT ITS SOURCE-CODE DEFAULT L=50, EXTENDED STEP GRID\n")
cmp("L1", "pre_mass_off", "L50", "scffcm", S1, "pre_mass_off (L=1) vs SC-FFCM (L=50), seeds 10-19 (held-out)")
cmp("L1", "pre_mass_off", "L50", "scffcm", S0, "pre_mass_off (L=1) vs SC-FFCM (L=50), seeds 0-9")
cmp("L1", "pre_none_off", "L50", "scffcm", S1, "pre_none_off (L=1, lossless) vs SC-FFCM (L=50), seeds 10-19 (held-out)")
cmp("L50", "scffcm", "L5", "scffcm", S1, "SC-FFCM L=50 vs SC-FFCM L=5, seeds 10-19 (held-out)")
cal = list(csv.DictReader(open(RES / "t9c_scffcm_calibration.csv")))
edge = sum(1 for r in cal if float(r["eta_l"]) == 4.0 or float(r["eta_g"]) == 8.0)
print(f"\ncalibration at L=50: chosen setting on the grid's upper edge for {edge}/{len(cal)}; max relative objective gap "
      f"{max(float(r['rel_obj_gap']) for r in cal):.2e}")
