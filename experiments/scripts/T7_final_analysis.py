"""T7: final analysis of the aggregation-weight design space + external baselines.

Every dataset is evaluated at its label-free non-degenerate fuzzifier m* (the largest m in
{2.0,1.5,1.3,1.2,1.1} at which centralized FCM keeps all c prototypes distinct on every seed):
  m*=2.0 : 6 synthetic scenarios, wine, satimage     -> T3 main/big runs (m=2)
  m*<2.0 : pendigits 1.5, digits_pca16 1.3, letter 1.3,
           digits_pca32 1.1, mnist784_pca32 (20c/50c) 1.1  -> T3 `_mstar` run
Baselines (stallmann_avg2, fednova, scffcm, centralized_fcm) come from T6 `_mstar` for all 14.
"""
import csv, collections, statistics as st, sys
from pathlib import Path
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parents[1] / "results"
AT2 = ['cluster_skew_hard','cluster_skew_overlap','dirichlet_0.03','dirichlet_0.1','overlap_noise',
       'quantity_skew_extreme','wine','satimage']
RESCUED = ['pendigits','digits_pca16','digits_pca32','letter','mnist784_pca32 (20c)','mnist784_pca32 (50c)']
MSTAR = {d: 2.0 for d in AT2} | {'pendigits':1.5,'digits_pca16':1.3,'letter':1.3,'digits_pca32':1.1,
                                  'mnist784_pca32 (20c)':1.1,'mnist784_pca32 (50c)':1.1}
DS = AT2 + RESCUED
CELLS = [f"{p}_{g}_{e}" for p in ['post','pre'] for g in ['none','footprint','mass'] for e in ['off','on']]
GF = 'post_footprint_on'

acc = collections.defaultdict(dict); worst = collections.defaultdict(dict); md = collections.defaultdict(dict)
def load(path, key, keep):
    if not path.exists(): print(f"MISSING {path}", file=sys.stderr); return
    for r in csv.DictReader(open(path)):
        if r['dataset'] not in keep: continue
        k = (r['dataset'], r[key]); s = int(r['seed'])
        acc[k][s] = float(r['mean_acc']); worst[k][s] = float(r['worst_acc']); md[k][s] = float(r['min_dist'])
for f in ['per_seed_metrics_main.csv','per_seed_metrics_big.csv']:
    load(R/'t3_design_space'/f, 'cell', set(AT2))
load(R/'t3_design_space'/'per_seed_metrics__mstar.csv', 'cell', set(RESCUED))
load(R/'t6_baselines'/'per_seed_metrics__mstar.csv', 'method', set(DS))

M = lambda ds, c, d=acc: st.mean(d[(ds,c)].values()) if d[(ds,c)] else float('nan')
def pw(ds, a, b, d=acc):
    ks = sorted(set(d[(ds,a)]) & set(d[(ds,b)])); x=[d[(ds,a)][k] for k in ks]; y=[d[(ds,b)][k] for k in ks]
    if len(ks) < 5 or all(abs(i-j) < 1e-12 for i,j in zip(x,y)): return None
    try: return wilcoxon(x, y).pvalue
    except Exception: return None
fp = lambda p: '  --  ' if p is None else f"{p:.4f}"

print("="*100); print("0. DEGENERACY AUDIT at m* (global prototypes, min-dist < 1e-3 = collapsed)"); print("="*100)
for ds in DS:
    col = sum(1 for c in CELLS if c.endswith('_off') for v in md[(ds,c)].values() if v < 1e-3)
    tot = sum(len(md[(ds,c)]) for c in CELLS if c.endswith('_off'))
    print(f"  {ds:<23} m*={MSTAR[ds]:<4} collapsed shared-model runs: {col}/{tot}")

print("\n"+"="*100); print("1. BEST DESIGN-SPACE CELL vs published GF-PFedFCM (paired Wilcoxon, 10 seeds)"); print("="*100)
sig = 0
for ds in DS:
    b = max(CELLS, key=lambda c: M(ds,c)); p = pw(ds, b, GF); d = M(ds,b) - M(ds,GF)
    s = p is not None and p < 0.05 and d > 0; sig += s
    print(f"  {ds:<23}{b:<21}{M(ds,b):.3f}/{M(ds,b,worst):.3f}  GF {M(ds,GF):.3f}/{M(ds,GF,worst):.3f}  d={d:+.3f} p={fp(p)} {'*' if s else ''}")
print(f"  => best cell significantly beats published GF on {sig}/{len(DS)}")

print("\n"+"="*100); print("2. SUBSTITUTION: PRE-POST benefit without vs with the footprint gate"); print("="*100)
sh = []
for ds in DS:
    ks = sorted(acc[(ds,'post_none_off')])
    a = st.mean(acc[(ds,'pre_none_off')][k]-acc[(ds,'post_none_off')][k] for k in ks)
    b = st.mean(acc[(ds,'pre_footprint_off')][k]-acc[(ds,'post_footprint_off')][k] for k in ks)
    sh.append(a-b); print(f"  {ds:<23} no-gate {a:+.3f}   gated {b:+.3f}   shrink {a-b:+.3f}")
print(f"  => mean shrink {st.mean(sh):+.3f}, Wilcoxon p={wilcoxon(sh).pvalue:.4f}, n={len(sh)}")

print("\n"+"="*100); print("3. FOOTPRINT (60-pt PoJG) vs plain normalized-mass gate"); print("="*100)
wins = []
for ds in DS:
    row = f"  {ds:<23}"
    for suf in ['off','on']:
        d = M(ds,f'post_footprint_{suf}') - M(ds,f'post_mass_{suf}'); p = pw(ds,f'post_footprint_{suf}',f'post_mass_{suf}')
        row += f"  {suf}: {d:+.3f} (p={fp(p)})"
        if p is not None and p < 0.05: wins.append((ds, suf, 'footprint' if d > 0 else 'mass'))
    print(row)
print(f"  => significant differences: {wins}")

print("\n"+"="*100); print("4. EXTERNAL BASELINES vs design space (mean / worst client ACC)"); print("="*100)
B = ['post_none_off', GF, 'stallmann_avg2', 'fednova', 'scffcm', 'centralized_fcm']
print(f"  {'dataset':<23}{'best cell':>14}" + "".join(f"{b[:13]:>15}" for b in B))
for ds in DS:
    bc = max(CELLS, key=lambda c: M(ds,c))
    print(f"  {ds:<23}{M(ds,bc):>8.3f}/{M(ds,bc,worst):.2f}" + "".join(f"{M(ds,b):>9.3f}/{M(ds,b,worst):.2f}" for b in B))
print("\n  SC-FFCM vs best design-space cell (paired Wilcoxon):")
for ds in DS:
    bc = max(CELLS, key=lambda c: M(ds,c)); p = pw(ds,'scffcm',bc); d = M(ds,'scffcm') - M(ds,bc)
    print(f"    {ds:<23} scffcm-best {d:+.3f}  p={fp(p)}")
