"""R3: the personalization effect WITHOUT the stopping confound. Personalized cells rerun with FFCM_NO_EARLYSTOP=1 so they
use the full 50 rounds like the non-personalized cells. Effect = cell_on(no early stop) - cell_off (fixed 50 rounds).
Also: how much did early stopping change the personalized cells themselves (noES vs the original early-stopped run)?"""
import csv, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
T3 = Path(__file__).resolve().parents[1] / "results" / "t3_design_space"
DS = ["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme",
      "wine","satimage","pendigits","digits_pca16","digits_pca32","letter","mnist784_pca32 (20c)","mnist784_pca32 (50c)"]
AT2 = set(DS[:8])
A = collections.defaultdict(dict); O = collections.defaultdict(dict)
def load(f, store, keep=None, key="mean_acc"):
    for r in csv.DictReader(open(T3 / f)):
        if keep is None or r["dataset"] in keep: store[(r["dataset"], r["cell"])][int(r["seed"])] = float(r[key])
for s in (A, ):
    load("per_seed_metrics_main.csv", s, AT2); load("per_seed_metrics_big.csv", s, AT2)
    load("per_seed_metrics__mstar.csv", s, set(DS) - AT2); load("per_seed_metrics__confirm.csv", s)
E = collections.defaultdict(dict)            # early-stopped originals, for the "did stopping matter" check
for k, v in A.items(): E[k] = dict(v)
N = collections.defaultdict(dict)            # no-early-stop reruns
load("per_seed_metrics__T9_noES.csv", N); load("per_seed_metrics__T9_noES_confirm.csv", N)
def cmp(X, a, Y, b, seeds, title):
    w = l = 0; det = []
    for ds in DS:
        x0, y0 = X[(ds, a)], Y[(ds, b)]; ks = [k for k in seeds if k in x0 and k in y0]
        if len(ks) < 5: continue
        x = [x0[k] for k in ks]; y = [y0[k] for k in ks]; d = st.mean(x) - st.mean(y)
        try: p = None if all(abs(i-j) < 1e-12 for i, j in zip(x, y)) else wilcoxon(x, y).pvalue
        except Exception: p = None
        s = p is not None and p < 0.05; w += s and d > 0; l += s and d < 0; det.append(f"{ds[:10]}:{d:+.3f}{'*' if s else ''}")
    print(f"--- {title}: better {w}, worse {l}\n     " + "  ".join(det))
S0, S1 = range(10), range(10, 20)
print("R3. PERSONALIZATION without the early-stopping confound (L=5, m*)\n")
print("(a) did early stopping change the personalized cells? (no-early-stop rerun minus original early-stopped run)")
for c in ("post_footprint_on", "pre_footprint_on", "pre_mass_on"):
    cmp(N, c, E, c, S0, f"{c}: noES - original, seeds 0-9")
print("\n(b) personalization effect with EQUAL round budgets: cell_on (no early stop) - cell_off")
for on, off in (("post_footprint_on", "post_footprint_off"), ("pre_footprint_on", "pre_footprint_off"),
                ("pre_mass_on", "pre_mass_off"), ("post_mass_on", "post_mass_off")):
    cmp(N, on, A, off, S0, f"{on} vs {off}, seeds 0-9")
    if on != "post_mass_on": cmp(N, on, A, off, S1, f"{on} vs {off}, seeds 10-19 (held-out)")
