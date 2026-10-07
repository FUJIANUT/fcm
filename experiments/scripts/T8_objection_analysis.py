"""T8: analyses answering the outline verifier's objections (existing T3/T6 data, canonical selection)."""
import sys, statistics as st
from pathlib import Path
src = (Path(__file__).resolve().parent / "T7_final_analysis.py").read_text().split('print("="*100); print("0.')[0]
ns = {"__file__": str(Path(__file__).resolve().parent / "T7_final_analysis.py")}; exec(src, ns)
acc, DS, M, pw = ns['acc'], ns['DS'], ns['M'], ns['pw']
from scipy.stats import wilcoxon
fp = lambda p: '  --  ' if p is None else f"{p:.4f}"
def within(ds, a1, b1, a2, b2):
    ks = sorted(set(acc[(ds,a1)]) & set(acc[(ds,b1)]) & set(acc[(ds,a2)]) & set(acc[(ds,b2)]))
    d = [(acc[(ds,a1)][k]-acc[(ds,b1)][k]) - (acc[(ds,a2)][k]-acc[(ds,b2)][k]) for k in ks]
    if all(abs(x) < 1e-12 for x in d): return st.mean(d), None
    try: return st.mean(d), wilcoxon(d).pvalue
    except Exception: return st.mean(d), None

print("E1. SUBSTITUTION as a WITHIN-CONFIG interaction (per-seed diff-in-diff, Wilcoxon over 10 seeds)")
print("    interaction = (PRE-POST | no gate) - (PRE-POST | footprint gate); >0: gate absorbs PRE's benefit")
print("    ceiling gap = C-FCM mean ACC minus gated POST cell (post_footprint_off)\n")
pos = pos_far = far = 0
for ds in DS:
    d, p = within(ds, 'pre_none_off', 'post_none_off', 'pre_footprint_off', 'post_footprint_off')
    gap = M(ds, 'centralized_fcm') - M(ds, 'post_footprint_off')
    s = p is not None and p < 0.05 and d > 0; pos += s
    isfar = gap > 0.03; far += isfar; pos_far += (s and isfar)
    print(f"   {ds:<23} interaction {d:+.3f} p={fp(p)} {'*' if s else ' '}   ceiling gap {gap:+.3f} {'FAR' if isfar else 'near'}")
print(f"\n   => significant positive interaction on {pos}/14; on configs FAR from ceiling (gap>0.03): {pos_far}/{far}")

print("\n\nE2. FOOTPRINT vs MASS gate at BOTH mass points (per-config Wilcoxon)")
for mp in ['post', 'pre']:
    for suf in ['off', 'on']:
        sig = []
        for ds in DS:
            d = M(ds, f'{mp}_footprint_{suf}') - M(ds, f'{mp}_mass_{suf}'); p = pw(ds, f'{mp}_footprint_{suf}', f'{mp}_mass_{suf}')
            if p is not None and p < 0.05: sig.append(f"{ds}({d:+.3f})")
        print(f"   {mp.upper():<4} pers {suf:<3}: significant {len(sig)}/14  {sig}")

print("\n\nE3. PERSONALIZATION effect (on minus off), per-config Wilcoxon")
for mp in ['post', 'pre']:
    for g in ['none', 'footprint', 'mass']:
        harm, hlp = [], []
        for ds in DS:
            d = M(ds, f'{mp}_{g}_on') - M(ds, f'{mp}_{g}_off'); p = pw(ds, f'{mp}_{g}_on', f'{mp}_{g}_off')
            if p is not None and p < 0.05: (hlp if d > 0 else harm).append(f"{ds}({d:+.3f})")
        print(f"   {mp}_{g:<9}: helps {len(hlp)}/14 {hlp}")
        print(f"   {'':<14} harms {len(harm)}/14 {harm}")
