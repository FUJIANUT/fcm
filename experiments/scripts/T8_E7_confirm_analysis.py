"""E7: confirmatory replication on FRESH seeds 10-19 (never seen when choosing the default), all 14 configs at m*.
Cells fixed in advance of this run: post_none_off (F-FCM), post_footprint_on (GF-PFedFCM), post_footprint_off,
pre_none_off, pre_mass_off, pre_footprint_off."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "t3_design_space"
acc = collections.defaultdict(dict)
for r in csv.DictReader(open(R / "per_seed_metrics__confirm.csv")):
    acc[(r["dataset"], r["cell"])][int(r["seed"])] = float(r["mean_acc"])
DS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
      "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
M = lambda ds, c: st.mean(acc[(ds, c)].values())
def pw(ds, a, b):
    ks = sorted(set(acc[(ds,a)]) & set(acc[(ds,b)])); x=[acc[(ds,a)][k] for k in ks]; y=[acc[(ds,b)][k] for k in ks]
    if all(abs(i-j) < 1e-12 for i, j in zip(x, y)): return None
    try: return wilcoxon(x, y).pvalue
    except Exception: return None
def tally(a, b):
    w = l = 0; rows = []
    for ds in DS:
        d = M(ds, a) - M(ds, b); p = pw(ds, a, b)
        s = p is not None and p < 0.05
        w += s and d > 0; l += s and d < 0
        rows.append((ds, d, p, s))
    return w, l, rows
fp = lambda p: '  --  ' if p is None else f"{p:.4f}"
print("E7. CONFIRMATORY REPLICATION on fresh seeds 10-19 (paired Wilcoxon per config)\n")
for a, b, title in [("pre_mass_off", "post_footprint_on", "fixed default pre_mass_off vs published GF-PFedFCM"),
                    ("pre_footprint_off", "post_footprint_on", "fixed default pre_footprint_off vs published GF-PFedFCM"),
                    ("pre_mass_off", "post_none_off", "pre_mass_off vs F-FCM (post_none_off)"),
                    ("pre_mass_off", "pre_footprint_off", "mass gate vs footprint gate at PRE"),
                    ("post_footprint_on", "post_footprint_off", "personalization effect (GF minus gate-only)")]:
    w, l, rows = tally(a, b)
    print(f"--- {title}: significantly better {w}/14, worse {l}/14")
    for ds, d, p, s in rows: print(f"     {ds:<23} {d:+.3f}  p={fp(p)} {'*' if s else ''}")
    print()
print("--- SUBSTITUTION interaction on fresh seeds (per-seed diff-in-diff)")
pos = 0; shr = []
for ds in DS:
    ks = sorted(acc[(ds, "pre_none_off")])
    d = [(acc[(ds,"pre_none_off")][k]-acc[(ds,"post_none_off")][k])-(acc[(ds,"pre_footprint_off")][k]-acc[(ds,"post_footprint_off")][k]) for k in ks]
    try: p = wilcoxon(d).pvalue
    except Exception: p = None
    s = p is not None and p < 0.05 and st.mean(d) > 0; pos += s; shr.append(st.mean(d))
    print(f"     {ds:<23} interaction {st.mean(d):+.3f}  p={fp(p)} {'*' if s else ''}")
print(f"   => significant positive on {pos}/14; pooled over configs: mean {st.mean(shr):+.3f}, Wilcoxon p={wilcoxon(shr).pvalue:.4f}")
