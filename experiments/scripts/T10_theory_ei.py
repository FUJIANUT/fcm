"""E-I helpers.
(1) descent_job: replicate the loop of T3.run_cell(mp, gate, 'off', local_steps=L) (personalization off => fixed 50 rounds,
    no early stop) and log J_beta(V_t) = sum_p beta_p J_p(U_t, V_t), with U_t = the optimal FCM memberships at V_t
    (predict_membership, i.e. the reduced objective min_U J), for t = 0..50. The final V is checked against an
    independent call of T3.run_cell with the same arguments.
(2) counterexample(): two Gaussian blobs, one per client; pooled FCM fixed point; one damped PRE round with L steps."""
import numpy as np
from T10_theory_common import setup


def jbeta(clients, beta, V, m):
    from fedfcmsim.fcm import predict_membership, squared_distances
    tot = 0.0
    for b, c in zip(beta, clients):
        u = predict_membership(c.x, V, m=m)
        tot += b * float(((u ** m) * squared_distances(c.x, V)).sum())
    return tot


def descent_job(arg):
    label, seed, mp, gate, L = arg
    T, clients, k, V0, beta, m = setup(label, seed)
    from fedfcmsim.fcm import predict_membership, run_local_fcm_steps
    V = V0.copy(); gate_fn = T.GATE_FNS[gate]
    J = [jbeta(clients, beta, V, m)]
    for _ in range(T.ROUNDS):
        num = np.zeros_like(V); den = np.zeros(k)
        for p, c in enumerate(clients):
            ub = predict_membership(c.x, V, m=m)
            loc = run_local_fcm_steps(c.x, V, steps=L, m=m)
            mass = (ub ** m).sum(0) if mp == "pre" else (loc.membership ** m).sum(0)
            w = beta[p] * mass * gate_fn(c.x, V, ub)
            num += w[:, None] * loc.centers; den += w
        Tt = V.copy(); ok = den > T.EPS; Tt[ok] = num[ok] / den[ok, None]
        V = (1.0 - T.SERVER_LR) * V + T.SERVER_LR * Tt
        J.append(jbeta(clients, beta, V, m))
    Vrc, _, _ = T.run_cell(clients, V0, mp, gate, "off", rounds=T.ROUNDS, local_steps=L, slr=T.SERVER_LR)
    J = np.array(J); dJ = np.diff(J); rel = dJ / np.abs(J[:-1])
    inc = np.where(rel > 1e-12)[0]
    acc, _ = T.client_accs(clients, [V] * len(clients))
    return dict(rule=f"{mp}_{gate}_off_L{L}", dataset=label, seed=seed, m=m, L=L,
                J0=float(J[0]), J50=float(J[-1]), max_rel_increase=float(rel.max()), max_abs_increase=float(dJ.max()),
                argmax_round=int(np.argmax(rel)) + 1, n_rounds_increase_gt0=int((dJ > 0).sum()),
                n_rounds_increase_gt1e12=int(len(inc)), first_increase_round=int(inc[0]) + 1 if len(inc) else -1,
                increase_rounds=";".join(str(i + 1) for i in inc),
                runcell_maxabs_diff=float(np.abs(V - Vrc).max()), final_mean_acc=acc,
                J_traj=";".join(f"{x:.17g}" for x in J))


# ---------------------------------------------------------------------------------------------------------------
def draw_blobs(variant):
    """Two blobs of 40 points each, N(0, 0.3^2 I) then N((4,4), 0.3^2 I), one generator default_rng(1)."""
    rng = np.random.default_rng(1); mu2 = np.array([4.0, 4.0])
    if variant == "normal(loc,scale,size)":
        a = rng.normal(0.0, 0.3, size=(40, 2)); b = rng.normal(mu2, 0.3, size=(40, 2))
    elif variant == "0.3*standard_normal+mu":
        a = 0.3 * rng.standard_normal((40, 2)); b = mu2 + 0.3 * rng.standard_normal((40, 2))
    elif variant == "multivariate_normal":
        a = rng.multivariate_normal([0.0, 0.0], 0.09 * np.eye(2), size=40); b = rng.multivariate_normal(mu2, 0.09 * np.eye(2), size=40)
    elif variant == "normal(size=(2,40)).T (column-major draw)":
        a = rng.normal(0.0, 0.3, size=(2, 40)).T; b = (mu2[:, None] + rng.normal(0.0, 0.3, size=(2, 40))).T
    else:
        raise ValueError(variant)
    return a, b


def pooled_J(X, V, m, U=None):
    from fedfcmsim.fcm import predict_membership, squared_distances
    U = predict_membership(X, V, m=m) if U is None else U
    return float(((U ** m) * squared_distances(X, V)).sum())


def pre_round(blocks, V, m, L, alpha=0.75, beta=None):
    from fedfcmsim.fcm import predict_membership, run_local_fcm_steps
    beta = np.full(len(blocks), 1.0 / len(blocks)) if beta is None else beta
    num = np.zeros_like(V); den = np.zeros(V.shape[0])
    for b, x in zip(beta, blocks):
        ub = predict_membership(x, V, m=m); loc = run_local_fcm_steps(x, V, steps=L, m=m)
        w = b * (ub ** m).sum(0); num += w[:, None] * loc.centers; den += w
    return (1 - alpha) * V + alpha * num / den[:, None]


def counterexample(variant="normal(loc,scale,size)", m=2.0, Ls=(1, 2, 5)):
    from fedfcmsim.fcm import fcm, predict_membership
    a, b = draw_blobs(variant); X = np.vstack([a, b]); init = np.vstack([a.mean(0), b.mean(0)])
    out = dict(variant=variant, means=init.tolist())
    for fp_name, kw in (("30 iterations", dict(max_iter=30, tol=-1.0)), ("converged tol 1e-12", dict(max_iter=100000, tol=1e-12))):
        res = fcm(X, 2, m=m, init_centers=init, **kw)
        Vs = res.centers; Us = predict_membership(X, Vs, m=m); J0 = pooled_J(X, Vs, m)
        rows = []
        for L in Ls:
            Vp = pre_round([a, b], Vs, m, L)
            rows.append(dict(L=L, dV_fro=float(np.linalg.norm(Vp - Vs)), dT_fro=float(np.linalg.norm(pre_round([a, b], Vs, m, L, alpha=1.0) - Vs)), dV_maxrow=float(np.linalg.norm(Vp - Vs, axis=1).max()),
                             dJ_reduced=pooled_J(X, Vp, m) - J0, dJ_fixedU=pooled_J(X, Vp, m, U=Us) - J0,
                             dJ_beta_reduced=0.5 * (pooled_J(X, Vp, m) - J0), dJ_mean_reduced=(pooled_J(X, Vp, m) - J0) / len(X),
                             V_plus=Vp.tolist()))
        out[fp_name] = dict(n_iter=res.n_iter, V_star=Vs.tolist(), J_star=J0, J_star_beta=0.5 * J0, J_star_mean=J0 / len(X),
                            fixed_point_residual=float(np.linalg.norm(pre_round([a, b], Vs, m, 1, alpha=1.0) - Vs)), rows=rows)
    return out
