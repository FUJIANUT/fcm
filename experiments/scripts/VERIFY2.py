"""Go/No-Go: does the collapse boundary move with the number of local steps L?"""
import numpy as np, sys
sys.path.insert(0,'.')
from fedfcmsim.synthetic import make_synthetic_clients, BASE_CENTERS
from fedfcmsim.fcm import run_local_fcm_steps, predict_membership
from fedfcmsim.federated import personalized_federated_fcm
M=2.0
def mindist(C):
    D=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1)); np.fill_diagonal(D,np.inf); return D.min()

# --- A. phantom mass measured from the IDEAL global model (not a collapsed one)
print("[A] Starting from the TRUE centers V* (the model the federation should converge to).")
print("    L : phi_L = mean mass on UNSUPPORTED prototypes / mean mass on SUPPORTED   (12 clients x 10 seeds)")
for L in [1,2,5,10,20]:
    phis, amps = [], []
    for seed in range(10):
        cls = make_synthetic_clients("cluster_skew_hard",12,220,random_state=seed)
        for c in cls:
            sup=np.array([j in set(np.unique(c.y).tolist()) for j in range(4)])
            ubar=predict_membership(c.x, BASE_CENTERS, m=M); Mbar=(ubar**M).sum(0)
            loc=run_local_fcm_steps(c.x, BASE_CENTERS, steps=L, m=M); ML=(loc.membership**M).sum(0)
            if sup.any() and (~sup).any():
                phis.append(ML[~sup].mean()/ML[sup].mean())
                amps.extend((ML[~sup]/np.maximum(Mbar[~sup],1e-300)).tolist())
    phi=np.mean(phis)
    print(f"    L={L:<3d} phi_L={phi:.4f}   pi*=phi/(1+phi)={phi/(1+phi):.3f}   median amplification vs L=0: {np.median(amps):.2f}x")

# --- B. does the ACTUAL collapse depend on L?  (delta_lr=0, so no personalization at all)
print("\n[B] Actual collapse, personalization OFF (delta_lr=0), 10 seeds, min global center distance (true=7.500):")
print(f"    {'L':<5}{'ungated':>18}{'server-gated':>18}")
for L in [1,2,5,10]:
    u,g=[],[]
    for seed in range(10):
        cls=make_synthetic_clients("cluster_skew_hard",12,220,random_state=seed)
        r=np.random.default_rng(seed); pooled=np.vstack([c.x for c in cls])
        V=pooled[r.choice(len(pooled),4,replace=False)].copy()
        a=personalized_federated_fcm(cls,V,rounds=45,local_steps=L,m=M,delta_lr=0.0,
                                     use_footprint=False,fixed_relevance=0.5,track_objective=False)
        b=personalized_federated_fcm(cls,V,rounds=45,local_steps=L,m=M,delta_lr=0.0,
                                     use_footprint=True,track_objective=False)
        u.append(mindist(a.centers)); g.append(mindist(b.centers))
    print(f"    {L:<5}{np.mean(u):>10.3f} +/-{np.std(u):<5.2f}{np.mean(g):>10.3f} +/-{np.std(g):<5.2f}")
