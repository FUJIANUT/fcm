"""Is the L=1 collapse an artifact of beta_p double-weighting, or does the STRICT
lossless (Barcena-style) aggregation also collapse under cluster-support skew?"""
import numpy as np, sys
sys.path.insert(0,'.')
from fedfcmsim.synthetic import make_synthetic_clients, BASE_CENTERS
from fedfcmsim.fcm import predict_membership, run_local_fcm_steps, fcm
from fedfcmsim.federated import granular_footprint, _footprint_relevance
M=2.0
def mindist(C):
    D=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1)); np.fill_diagonal(D,np.inf); return D.min()

def run(clients, V0, rounds=60, mode="strict", gate=False):
    """mode=strict : v_j = sum_p sum_k u^m x / sum_p sum_k u^m, memberships at the BROADCAST
       prototypes, full participation, server_lr=1.  This IS centralized FCM (lossless).
       mode=beta   : the repo's rule, weight beta_p = N_p/sum N_q multiplied on top."""
    V=V0.copy(); N=np.array([len(c.x) for c in clients],float); beta=N/N.sum()
    for _ in range(rounds):
        num=np.zeros_like(V); den=np.zeros(V.shape[0])
        for p,c in enumerate(clients):
            u=predict_membership(c.x,V,m=M); w=u**M
            mass=w.sum(0); ctr=(w.T@c.x)/np.maximum(mass,1e-300)[:,None]
            a=np.ones(V.shape[0])
            if gate:
                fp,_=granular_footprint(c.x,V,u,m=M,grid_size=60)
                a=_footprint_relevance(fp,0.05)
            s = beta[p] if mode=="beta" else 1.0
            num+= (s*mass*a)[:,None]*ctr; den+= s*mass*a
        Vn=V.copy(); ok=den>1e-12; Vn[ok]=num[ok]/den[ok,None]
        if np.linalg.norm(Vn-V)<1e-6: V=Vn; break
        V=Vn
    return V

print("cluster_skew_hard, 12 clients x 220 samples, L=1 (one membership update per round).")
print("True inter-prototype min distance = 7.500\n")
rows={"strict lossless (no beta)":[], "repo rule (x beta_p)":[], "strict + footprint gate":[], "centralized FCM":[]}
for seed in range(10):
    cls=make_synthetic_clients("cluster_skew_hard",12,220,random_state=seed)
    r=np.random.default_rng(seed); pooled=np.vstack([c.x for c in cls])
    V0=pooled[r.choice(len(pooled),4,replace=False)].copy()
    rows["strict lossless (no beta)"].append(mindist(run(cls,V0,mode="strict")))
    rows["repo rule (x beta_p)"].append(mindist(run(cls,V0,mode="beta")))
    rows["strict + footprint gate"].append(mindist(run(cls,V0,mode="strict",gate=True)))
    rows["centralized FCM"].append(mindist(fcm(pooled,4,m=M,init_centers=V0,max_iter=200).centers))
for k,v in rows.items():
    print(f"  {k:<28} min-dist = {np.mean(v):6.3f} +/- {np.std(v):5.3f}   (per-seed: {' '.join(f'{x:.2f}' for x in v)})")
