"""Re-aggregate E_H.csv -> E_H_config.csv means, recompute Spearman (own rank code), and E-I table from E_I_descent.csv."""
import csv, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parents[1]
eh = list(csv.DictReader(open(D/"E_H.csv"))); cf = list(csv.DictReader(open(D/"E_H_config.csv"))); ei = list(csv.DictReader(open(D/"E_I_descent.csv")))
cols = ["tilt_cv_median","disp_median","spread_median","rel_median","cv_unclipped_median","cv_clipped_median","share_q_unclipped_mean","share_r_unclipped_mean","pairs_unclipped_frac_mean"]
mx = 0; seeds=set()
for r in cf:
    rr = [x for x in eh if x["trajectory"]==r["trajectory"] and x["dataset"]==r["dataset"]]
    seeds |= {int(x["seed"]) for x in rr}
    for c in cols:
        v=[float(x[c]) for x in rr if np.isfinite(float(x[c]))]; mx=max(mx,abs(np.mean(v)-float(r[c])))
print("E_H_config mean re-aggregation max |diff|:", mx, "rows", len(cf), "seeds in E_H.csv", sorted(seeds), "n E_H rows", len(eh))
def rank(a):
    a=np.asarray(a); o=a.argsort(); r=np.empty(len(a)); r[o]=np.arange(len(a)); 
    for v in np.unique(a): r[a==v]=r[a==v].mean()
    return r
for traj in ["post_footprint_off_L5","pre_footprint_off_L5","pre_footprint_off_L1"]:
    rows=[r for r in cf if r["trajectory"]==traj]; g=np.array([float(r["acc_gap_fp_vs_mass"]) for r in rows])
    out=[]
    for c in ["tilt_cv_median","rel_median","cv_unclipped_median","cv_clipped_median","share_r_unclipped_mean"]:
        x=np.array([float(r[c]) for r in rows]); out.append(f"{c}={np.corrcoef(rank(x),rank(g))[0,1]:+.3f}")
    print(traj, " ".join(out))
for rule in ["pre_none_off_L1","pre_mass_off_L1","pre_none_off_L5"]:
    rr=[r for r in ei if r["rule"]==rule]
    Js=[np.array([float(v) for v in r["J_traj"].split(";")]) for r in rr]
    rel=[np.diff(J)/J[:-1] for J in Js]; ab=[np.diff(J) for J in Js]
    print(rule, len(rr), "seeds", sorted({int(r["seed"]) for r in rr}), f"maxrel {max(x.max() for x in rel):.3e} maxabs {max(x.max() for x in ab):.3e}",
          "n>0", sum(int((x>0).sum()) for x in ab), "n>1e-12", sum(int((x>1e-12).sum()) for x in rel), "runs", sum(int((x>1e-12).any()) for x in rel),
          "max runcell diff", max(float(r["runcell_maxabs_diff"]) for r in rr))
