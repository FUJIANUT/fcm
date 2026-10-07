"""E5 analysis: dependence of the PRE/POST axis and of the substitution interaction on the number of local
steps L in {1,2,5,10}. L=5 comes from the canonical data (T3 main at m*=2, T3 _mstar for rescued)."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "t3_design_space"
SUB = ["cluster_skew_overlap","dirichlet_0.1","overlap_noise","wine","satimage","pendigits","digits_pca16","digits_pca32"]
AT2 = {"cluster_skew_overlap","dirichlet_0.1","overlap_noise","wine","satimage"}
acc = collections.defaultdict(dict)
def load(f, L, keep):
    for r in csv.DictReader(open(R / f)):
        if r["dataset"] in keep: acc[(L, r["dataset"], r["cell"])][int(r["seed"])] = float(r["mean_acc"])
load("per_seed_metrics_main.csv", 5, AT2); load("per_seed_metrics__mstar.csv", 5, set(SUB) - AT2)
for L in (1, 2, 10): load(f"per_seed_metrics__L{L}.csv", L, set(SUB))
M = lambda L, ds, c: st.mean(acc[(L, ds, c)].values())
def did(L, ds):
    ks = sorted(acc[(L, ds, "pre_none_off")])
    d = [(acc[(L,ds,"pre_none_off")][k]-acc[(L,ds,"post_none_off")][k])-(acc[(L,ds,"pre_footprint_off")][k]-acc[(L,ds,"post_footprint_off")][k]) for k in ks]
    try: p = wilcoxon(d).pvalue
    except Exception: p = None
    return st.mean(d), p
print("E5. L-SWEEP (8 configs at m*, 10 seeds)\n")
print("(a) PRE - POST without gate   [mean client ACC difference]")
print(f"   {'dataset':<22}" + "".join(f"{'L='+str(L):>10}" for L in (1,2,5,10)))
for ds in SUB: print(f"   {ds:<22}" + "".join(f"{M(L,ds,'pre_none_off')-M(L,ds,'post_none_off'):>+10.3f}" for L in (1,2,5,10)))
print("\n(b) absolute mean client ACC of the two ungated cells")
print(f"   {'dataset':<22}" + "".join(f"{'POST L='+str(L):>12}" for L in (1,2,5,10)) + "".join(f"{'PRE L='+str(L):>11}" for L in (1,2,5,10)))
for ds in SUB: print(f"   {ds:<22}" + "".join(f"{M(L,ds,'post_none_off'):>12.3f}" for L in (1,2,5,10)) + "".join(f"{M(L,ds,'pre_none_off'):>11.3f}" for L in (1,2,5,10)))
print("\n(c) substitution interaction (PRE-POST|no gate) - (PRE-POST|footprint), per-seed Wilcoxon")
print(f"   {'dataset':<22}" + "".join(f"{'L='+str(L):>16}" for L in (1,2,5,10)))
for ds in SUB:
    row = f"   {ds:<22}"
    for L in (1,2,5,10):
        d, p = did(L, ds); row += f"{d:>+8.3f} ({'--' if p is None else f'{p:.3f}'}{'*' if p is not None and p<0.05 and d>0 else ' '})"
    print(row)
cnt = {L: sum(1 for ds in SUB if (lambda t: t[1] is not None and t[1] < 0.05 and t[0] > 0)(did(L, ds))) for L in (1,2,5,10)}
print("\n   significant positive interaction per L:", ", ".join(f"L={L}: {cnt[L]}/8" for L in (1,2,5,10)))
