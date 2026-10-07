"""Independent verification of the workflow's key empirical claims."""
import numpy as np, sys
sys.path.insert(0, '.')
from fedfcmsim.synthetic import make_synthetic_clients, BASE_CENTERS
from fedfcmsim.fcm import run_local_fcm_steps, predict_membership
from fedfcmsim.federated import federated_fcm, personalized_federated_fcm, granular_footprint

M = 2.0
rng = np.random.default_rng(0)

# --- 0. geometry
d = np.sqrt(((BASE_CENTERS[:,None,:]-BASE_CENTERS[None,:,:])**2).sum(-1))
mind = d[d>0].min()
print(f"[0] BASE_CENTERS min pairwise dist = {mind:.3f}, std=0.75 -> Delta/sigma = {mind/0.75:.2f}")

def init_centers(clients, seed):
    r = np.random.default_rng(seed)
    pooled = np.vstack([c.x for c in clients])
    idx = r.choice(len(pooled), size=4, replace=False)
    return pooled[idx].copy()

amps, phi0s, phiLs = [], [], []
fact_err = 0.0
for seed in range(10):
    clients = make_synthetic_clients("cluster_skew_hard", n_clients=12,
                                     samples_per_client=220, random_state=seed)
    V = init_centers(clients, seed)
    # run ungated federation a few rounds so V is a realistic broadcast model
    res = federated_fcm(clients, V, rounds=8, local_steps=5, m=M, track_objective=False)
    Vb = res.centers
    for c in clients:
        supported = set(np.unique(c.y).tolist())
        ubar = predict_membership(c.x, Vb, m=M)          # pre-adaptation, at broadcast V
        Mbar = (ubar**M).sum(axis=0)
        loc = run_local_fcm_steps(c.x, Vb, steps=5, m=M)  # L=5 local Picard steps
        ML = (loc.membership**M).sum(axis=0)
        sup = np.array([j in supported for j in range(4)])
        # phantom amplification on unsupported prototypes
        amps.extend((ML[~sup]/np.maximum(Mbar[~sup],1e-300)).tolist())
        if sup.any() and (~sup).any():
            phi0s.append(Mbar[~sup].mean()/Mbar[sup].mean())
            phiLs.append(ML[~sup].mean()/ML[sup].mean())
        # factorization check: T == (Mbar/N) * Phi
        T, rho = granular_footprint(c.x, Vb, ubar, m=M, grid_size=60)
        dist = np.sqrt(((c.x[:,None,:]-Vb[None,:,:])**2).sum(-1))
        for j in range(4):
            Rq = max(float(np.quantile(dist[:,j],0.95)),1e-12)
            radii = np.linspace(0.0,Rq,60); w=(ubar[:,j]**M)
            best=-1.0
            for r_ in radii:
                F = w[dist[:,j]<=r_].sum()/max(Mbar[j],1e-300)
                best=max(best,F*(1.0-r_/Rq))
            fact_err=max(fact_err, abs(T[j]-(Mbar[j]/len(c.x))*best))

amps=np.array(amps); print(f"[1] phantom amplification M^(5)/Mbar on UNSUPPORTED pairs, 10 seeds, n={len(amps)}:")
print(f"    median={np.median(amps):.1f}x  mean={amps.mean():.1f}x  p90={np.percentile(amps,90):.1f}x  max={amps.max():.1f}x")
print(f"[2] phi_0 (pre-adapt unsup/sup mass ratio) = {np.mean(phi0s):.5f}")
print(f"    phi_5 (post-adapt, L=5)               = {np.mean(phiLs):.4f}   -> pi* = phi/(1+phi) = {np.mean(phiLs)/(1+np.mean(phiLs)):.3f}")
print(f"[3] max |T - (Mbar/N)*Phi| over all (seed,client,j) = {fact_err:.3e}   (exact factorization)")

# --- 4. collapse with personalization OFF (delta_lr=0): server gate alone
def mindist(C):
    D=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1)); np.fill_diagonal(D,np.inf); return D.min()
ung, gat = [], []
for seed in range(10):
    clients = make_synthetic_clients("cluster_skew_hard", 12, 220, random_state=seed)
    V = init_centers(clients, seed)
    a = personalized_federated_fcm(clients, V, rounds=45, local_steps=5, m=M, delta_lr=0.0,
                                   use_footprint=False, fixed_relevance=0.5, track_objective=False)
    b = personalized_federated_fcm(clients, V, rounds=45, local_steps=5, m=M, delta_lr=0.0,
                                   use_footprint=True, server_gate=True, client_gate=True,
                                   track_objective=False)
    ung.append(mindist(a.centers)); gat.append(mindist(b.centers))
print(f"[4] delta_lr=0 (NO personalization), min global center distance over 10 seeds:")
print(f"    ungated (naive mass-weighted) = {np.mean(ung):.3f} +/- {np.std(ung):.3f}")
print(f"    server gate on               = {np.mean(gat):.3f} +/- {np.std(gat):.3f}")
print(f"    true inter-center min dist   = {mind:.3f}")
