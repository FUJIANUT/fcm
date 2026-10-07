"""R5 analysis: do the main contrasts hold across the fuzzifier m in {1.3,1.5,2.0}? 8 configurations that are
non-degenerate over the whole range; seeds 0-9; no early stopping for personalized cells. Two criteria: mean client
ACC (higher better) and the FCM objective of the evaluation prototypes, a label-free fuzzy criterion (lower better)."""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "t3_design_space"
CF = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme","wine","satimage"]
d = collections.defaultdict(dict)
for mm in ("13", "15", "20"):
    for r in csv.DictReader(open(R / f"per_seed_metrics__T9_m{mm}.csv")):
        for key in ("mean_acc", "fcm_obj"): d[(mm, r["dataset"], r["cell"], key)][int(r["seed"])] = float(r[key])
M = lambda mm, ds, c, key: st.mean(d[(mm, ds, c, key)].values())
def test(mm, a, b, key, higher_better=True):
    w = l = 0; det = []
    for ds in CF:
        A, B = d[(mm, ds, a, key)], d[(mm, ds, b, key)]; ks = sorted(set(A) & set(B))
        x = [A[k] for k in ks]; y = [B[k] for k in ks]; diff = st.mean(x) - st.mean(y)
        if all(abs(i - j) < 1e-12 for i, j in zip(x, y)): p = None
        else:
            try: p = wilcoxon(x, y).pvalue
            except Exception: p = None
        good = diff > 0 if higher_better else diff < 0
        s = p is not None and p < 0.05; w += s and good; l += s and not good
        det.append(f"{ds[:12]}:{diff:+.3f}{'*' if s else ''}")
    return w, l, det
def interaction(mm):
    pos = 0; det = []
    for ds in CF:
        ks = sorted(d[(mm, ds, "pre_none_off", "mean_acc")])
        g = lambda c: d[(mm, ds, c, "mean_acc")]
        diff = [(g("pre_none_off")[k]-g("post_none_off")[k])-(g("pre_footprint_off")[k]-g("post_footprint_off")[k]) for k in ks]
        try: p = wilcoxon(diff).pvalue
        except Exception: p = None
        s = p is not None and p < 0.05 and st.mean(diff) > 0; pos += s; det.append(f"{ds[:12]}:{st.mean(diff):+.3f}{'*' if s else ''}")
    return pos, det
print("FUZZINESS SENSITIVITY (8 configs non-degenerate for m in {1.3,1.5,2.0}; seeds 0-9; paired Wilcoxon)\n")
for mm, lab in (("13", "m=1.3"), ("15", "m=1.5"), ("20", "m=2.0")):
    print(f"==== {lab}")
    pos, det = interaction(mm); print(f"  substitution interaction: significant positive {pos}/8   " + "  ".join(det))
    for a, b, name in (("pre_mass_off", "post_footprint_on", "pre_mass_off vs GF"), ("pre_mass_off", "post_none_off", "pre_mass_off vs F-FCM"),
                       ("post_footprint_on", "post_footprint_off", "personalization (GF vs gate-only)")):
        w, l, det = test(mm, a, b, "mean_acc"); print(f"  ACC  {name:<36} better {w}/8 worse {l}/8   " + "  ".join(det))
        w, l, det = test(mm, a, b, "fcm_obj", higher_better=False); print(f"  OBJ  {name:<36} better {w}/8 worse {l}/8   " + "  ".join(det))
    print()
