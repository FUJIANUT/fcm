"""Isolate the WEIGHT MISALIGNMENT: same local centers, mass measured pre- vs post-adaptation."""
import numpy as np, sys
sys.path.insert(0,'.')
from fedfcmsim.synthetic import make_synthetic_clients
from fedfcmsim.fcm import predict_membership, run_local_fcm_steps
from fedfcmsim.federated import granular_footprint, _footprint_relevance
from fedfcmsim.metrics import clustering_accuracy
M=2.0
def mindist(C):
    D=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1)); np.fill_diagonal(D,np.inf); return D.min()

def run(clients, V0, L, weight_at, gate=False, rounds=45, slr=0.75):
    V=V0.copy(); N=np.array([len(c.x) for c in clients],float); beta=N/N.sum()
    for _ in range(rounds):
        num=np.zeros_like(V); den=np.zeros(V.shape[0])
        for p,c in enumerate(clients):
            ubar=predict_membership(c.x,V,m=M)                 # at BROADCAST prototypes
            loc=run_local_fcm_steps(c.x,V,steps=L,m=M)          # L local Picard steps
            mass = (ubar**M).sum(0) if weight_at=="pre" else (loc.membership**M).sum(0)
            a=np.ones(V.shape[0])
            if gate:
                fp,_=granular_footprint(c.x,V,ubar,m=M,grid_size=60)
                a=_footprint_relevance(fp,0.05)
            w=beta[p]*mass*a
            num+=w[:,None]*loc.centers; den+=w
        T=V.copy(); ok=den>1e-12; T[ok]=num[ok]/den[ok,None]
        V=(1-slr)*V+slr*T
    return V

def acc(clients,V):
    out=[]
    for c in clients:
        lab=np.argmin(((c.x[:,None,:]-V[None,:,:])**2).sum(-1),axis=1)
        out.append(clustering_accuracy(c.y,lab))
    return float(np.mean(out)), float(np.min(out))

print("cluster_skew_hard, 12 clients, L=5 local steps, 10 seeds.  True min-dist = 7.500\n")
print(f"  {'aggregation weight':<42}{'min-dist':>12}{'mean ACC':>11}{'worst ACC':>11}")
cfg=[("mass measured POST-adaptation  (repo / F-FCM)","post",False),
     ("mass measured PRE-adaptation   (aligned)","pre",False),
     ("POST-adaptation + footprint gate  (GF)","post",True),
     ("PRE-adaptation + footprint gate","pre",True)]
for name,wa,g in cfg:
    md,am,wm=[],[],[]
    for seed in range(10):
        cls=make_synthetic_clients("cluster_skew_hard",12,220,random_state=seed)
        r=np.random.default_rng(seed); pooled=np.vstack([c.x for c in cls])
        V0=pooled[r.choice(len(pooled),4,replace=False)].copy()
        V=run(cls,V0,5,wa,gate=g)
        a,w=acc(cls,V); md.append(mindist(V)); am.append(a); wm.append(w)
    print(f"  {name:<42}{np.mean(md):>9.3f}   {np.mean(am):>8.3f}   {np.mean(wm):>8.3f}")
