import numpy as np, sys
sys.path.insert(0,'.')
exec(open('/private/tmp/claude-501/-Users-chunmaojiang-Library-CloudStorage-Dropbox------------z----------------2026-ffcm-fuzzysetandsystem/42a2808e-3dd8-40af-b680-6f9ac2559a98/scratchpad/VERIFY4.py').read().split('print("cluster_skew')[0])
SC=["cluster_skew_hard","cluster_skew_overlap","dirichlet_0.03","dirichlet_0.1","overlap_noise","quantity_skew_extreme"]
print(f"{'scenario':<24}{'F-FCM (post)':>20}{'ALIGNED (pre)':>20}{'GF (post+gate)':>20}")
print(f"{'':<24}{'mean / worst':>20}{'mean / worst':>20}{'mean / worst':>20}")
for sc in SC:
    out={}
    for name,wa,g in [("post","post",False),("pre","pre",False),("gate","post",True)]:
        am,wm=[],[]
        for seed in range(10):
            cls=make_synthetic_clients(sc,12,220,random_state=seed)
            r=np.random.default_rng(seed); pooled=np.vstack([c.x for c in cls])
            V0=pooled[r.choice(len(pooled),4,replace=False)].copy()
            V=run(cls,V0,5,wa,gate=g); a,w=acc(cls,V); am.append(a); wm.append(w)
        out[name]=(np.mean(am),np.mean(wm))
    f=lambda t: f"{t[0]:.3f} / {t[1]:.3f}"
    print(f"{sc:<24}{f(out['post']):>20}{f(out['pre']):>20}{f(out['gate']):>20}")
