"""PRE mass at L=1 (the lossless Corcuera Barcena setting: weight and estimate both from the broadcast
membership) versus SC-FFCM, centralized FCM and the L=5 default, all 14 configs at m*, seeds 0-9 (paired:
same partition and init seeds as the T6 baselines)."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
R = Path(__file__).resolve().parents[1] / "results"
acc = collections.defaultdict(dict); wst = collections.defaultdict(dict)
for r in csv.DictReader(open(R / "t3_design_space" / "per_seed_metrics__L1all.csv")):
    acc[(r["dataset"], "L1_" + r["cell"])][int(r["seed"])] = float(r["mean_acc"]); wst[(r["dataset"], "L1_" + r["cell"])][int(r["seed"])] = float(r["worst_acc"])
for r in csv.DictReader(open(R / "t6_baselines" / "per_seed_metrics__mstar.csv")):
    acc[(r["dataset"], r["method"])][int(r["seed"])] = float(r["mean_acc"]); wst[(r["dataset"], r["method"])][int(r["seed"])] = float(r["worst_acc"])
DS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
      "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
M = lambda ds, c, d=acc: st.mean(d[(ds, c)].values())
def pw(ds, a, b):
    ks = sorted(set(acc[(ds,a)]) & set(acc[(ds,b)])); x=[acc[(ds,a)][k] for k in ks]; y=[acc[(ds,b)][k] for k in ks]
    if all(abs(i-j) < 1e-12 for i, j in zip(x, y)): return None
    try: return wilcoxon(x, y).pvalue
    except Exception: return None
fp = lambda p: '  --  ' if p is None else f"{p:.4f}"
cols = ["L1_pre_none_off", "L1_pre_mass_off", "L1_post_none_off", "pre_footprint_off", "scffcm", "stallmann_avg2", "centralized_fcm"]
print("PRE mass at L=1 vs SC-FFCM and centralized FCM (mean / worst client ACC, seeds 0-9, m*)\n")
print(f"   {'dataset':<22}" + "".join(f"{c.replace('L1_','L1:')[:15]:>16}" for c in cols))
for ds in DS: print(f"   {ds:<22}" + "".join(f"{M(ds,c):>9.3f}/{M(ds,c,wst):.3f}" for c in cols))
for a, b in [("L1_pre_none_off", "scffcm"), ("L1_pre_mass_off", "scffcm"), ("L1_pre_none_off", "centralized_fcm"),
             ("L1_pre_none_off", "L1_post_none_off"), ("L1_pre_mass_off", "pre_footprint_off")]:
    w = l = 0; det = []
    for ds in DS:
        d = M(ds, a) - M(ds, b); p = pw(ds, a, b); s = p is not None and p < 0.05
        w += s and d > 0; l += s and d < 0; det.append(f"{ds}:{d:+.3f}{'*' if s else ''}")
    print(f"\n--- {a} vs {b}: significantly better {w}/14, worse {l}/14")
    print("     " + "  ".join(det))
