"""Round-budget sweep: does multi-step local training (L=5) beat L=1 when communication rounds are scarce?
R in {5,10,20,50}; R=50 comes from _L1all (L=1) and the canonical L=5 data (T3 main at m*=2, _mstar rescued)."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
R_ = Path(__file__).resolve().parents[1] / "results" / "t3_design_space"
SUB = ["cluster_skew_overlap","dirichlet_0.1","overlap_noise","wine","satimage","pendigits","digits_pca16","digits_pca32","mnist784_pca32 (20c)"]
AT2 = {"cluster_skew_overlap","dirichlet_0.1","overlap_noise","wine","satimage"}
acc = collections.defaultdict(dict)
def load(f, R, L, keep):
    p = R_ / f
    if not p.exists(): print("MISSING", f); return
    for r in csv.DictReader(open(p)):
        if r["dataset"] in keep: acc[(R, L, r["dataset"], r["cell"])][int(r["seed"])] = float(r["mean_acc"])
for R in (5, 10, 20):
    for L in (1, 5): load(f"per_seed_metrics__R{R}_L{L}.csv", R, L, set(SUB))
load("per_seed_metrics__L1all.csv", 50, 1, set(SUB))
load("per_seed_metrics_main.csv", 50, 5, AT2)
load("per_seed_metrics__mstar.csv", 50, 5, set(SUB) - AT2)
M = lambda R, L, ds, c: st.mean(acc[(R, L, ds, c)].values()) if acc[(R, L, ds, c)] else float("nan")
def pw(R, ds, a, La, b, Lb):
    A, B = acc[(R, La, ds, a)], acc[(R, Lb, ds, b)]; ks = sorted(set(A) & set(B))
    x = [A[k] for k in ks]; y = [B[k] for k in ks]
    if len(ks) < 5 or all(abs(i-j) < 1e-12 for i, j in zip(x, y)): return None
    try: return wilcoxon(x, y).pvalue
    except Exception: return None
print("ROUND-BUDGET SWEEP (m*, 10 seeds): mean client ACC\n")
for c in ["pre_mass_off", "pre_none_off", "post_none_off", "post_footprint_on"]:
    print(f"--- cell {c}")
    print(f"   {'dataset':<22}" + "".join(f"{'R='+str(R)+' L=1':>12}{'L=5':>7}" for R in (5, 10, 20, 50)))
    for ds in SUB: print(f"   {ds:<22}" + "".join(f"{M(R,1,ds,c):>12.3f}{M(R,5,ds,c):>7.3f}" for R in (5, 10, 20, 50)))
    print()
print("--- KEY TEST: pre_mass_off at L=1 vs pre_mass_off at L=5, per round budget (paired Wilcoxon)")
for R in (5, 10, 20):
    w = l = 0; det = []
    for ds in SUB:
        d = M(R, 1, ds, "pre_mass_off") - M(R, 5, ds, "pre_mass_off"); p = pw(R, ds, "pre_mass_off", 1, "pre_mass_off", 5)
        s = p is not None and p < 0.05; w += s and d > 0; l += s and d < 0; det.append(f"{ds}:{d:+.3f}{'*' if s else ''}")
    print(f"   R={R:<3} L=1 better {w}/9, worse {l}/9   " + "  ".join(det))
print("\n--- L=1 pre_mass_off vs L=5 published GF (post_footprint_on), per round budget")
for R in (5, 10, 20):
    w = l = 0
    for ds in SUB:
        d = M(R, 1, ds, "pre_mass_off") - M(R, 5, ds, "post_footprint_on"); p = pw(R, ds, "pre_mass_off", 1, "post_footprint_on", 5)
        s = p is not None and p < 0.05; w += s and d > 0; l += s and d < 0
    print(f"   R={R:<3} L=1 pre_mass_off better {w}/9, worse {l}/9")
