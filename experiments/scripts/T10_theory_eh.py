"""E-H: one-round tilt diagnostics of T9_theory_link_b along a footprint-gated, non-personalized trajectory
(mass point `mp` in {post, pre}, L local steps), extended by a split into unclipped / clipped (client, prototype) pairs.

Per round t and prototype j (all quantities evaluated at the broadcast V_t, exactly as in T9_theory_link_b):
  W_pj   = beta_p * Mass_pj   (Mass = post-adaptation fuzzy mass for mp=post, pre-adaptation mass Mbar for mp=pre)
  pq     = W q / sum (mass-gated weights = the base measure),  pr = W r / sum (footprint-gated = server update)
  g      = r / q (tilt);   cv = sqrt(Var_pq g) / E_pq g
  disp   = ||sum pr c - sum pq c||,  spread = sqrt(E_pq ||c - tq||^2),  rel = disp / spread
Split: U = {r > r_min and q > r_min} (unclipped), C = complement (either at the floor).
  cv_U / cv_C : the same weighted CV with pq restricted to the subset and renormalized (NaN if < 2 clients)
  shq_U       : share of mass-gated weight pq on U (shq_C = 1 - shq_U)
  shr_U       : share of footprint-gated server weight pr on U
The trajectory update is the footprint-gated server step of run_cell(mp, 'footprint', 'off', local_steps=L).
Per (config, seed): medians over (round, prototype) [nanmedian for subset quantities] + mean shares; and the final
mean client ACC of the trajectory (for cross-checking against the stored per-seed CSVs)."""
import numpy as np
from T10_theory_common import setup


def eh_job(arg):
    label, seed, mp, L = arg
    T, clients, k, V, beta, m = setup(label, seed)
    from fedfcmsim.fcm import predict_membership, run_local_fcm_steps
    P = len(clients); rmin = T.MIN_RELEVANCE
    cvs, disps, spreads, rels = [], [], [], []
    cvU, cvC, shqU, shrU, fracU = [], [], [], [], []
    for t in range(T.ROUNDS):
        MW = np.zeros((P, k)); C = np.zeros((P, k, V.shape[1])); R = np.zeros((P, k)); Q = np.zeros((P, k))
        for p, c in enumerate(clients):
            ub = predict_membership(c.x, V, m=m); loc = run_local_fcm_steps(c.x, V, steps=L, m=m)
            MW[p] = (ub ** m).sum(0) if mp == "pre" else (loc.membership ** m).sum(0)
            C[p] = loc.centers
            R[p] = T.gate_footprint(c.x, V, ub); Q[p] = T.gate_mass(c.x, V, ub)
        U = (R > rmin + 1e-12) & (Q > rmin + 1e-12)
        for j in range(k):
            wq = beta * MW[:, j] * Q[:, j]; wr = beta * MW[:, j] * R[:, j]
            if wq.sum() <= 0 or wr.sum() <= 0:
                continue
            pq = wq / wq.sum(); pr = wr / wr.sum()
            tq = (pq[:, None] * C[:, j]).sum(0); tr = (pr[:, None] * C[:, j]).sum(0)
            disp = float(np.linalg.norm(tr - tq)); spread = float(np.sqrt((pq * ((C[:, j] - tq) ** 2).sum(1)).sum()))
            g = R[:, j] / Q[:, j]; gm = (pq * g).sum()
            cv = float(np.sqrt((pq * (g - gm) ** 2).sum()) / gm) if gm > 0 else np.nan
            disps.append(disp); spreads.append(spread); cvs.append(cv)
            if spread > 1e-12:
                rels.append(disp / spread)
            u = U[:, j]
            for mask, store in ((u, cvU), (~u, cvC)):
                s = pq[mask].sum()
                if mask.sum() >= 2 and s > 0:
                    w = pq[mask] / s; gg = g[mask]; mu = (w * gg).sum()
                    store.append(float(np.sqrt((w * (gg - mu) ** 2).sum()) / mu))
                else:
                    store.append(np.nan)
            shqU.append(float(pq[u].sum())); shrU.append(float(pr[u].sum())); fracU.append(float(u.mean()))
        w = beta[:, None] * MW * R; num = (w[:, :, None] * C).sum(0); den = w.sum(0)
        Tt = V.copy(); ok = den > T.EPS; Tt[ok] = num[ok] / den[ok, None]; V = (1 - T.SERVER_LR) * V + T.SERVER_LR * Tt
    acc, worst = T.client_accs(clients, [V] * P)
    nm = lambda a: float(np.nanmedian(a)) if np.any(np.isfinite(a)) else float("nan")
    return dict(trajectory=f"{mp}_footprint_off_L{L}", dataset=label, seed=seed, m=m, L=L,
                tilt_cv_median=float(np.nanmedian(cvs)), disp_median=float(np.median(disps)),
                spread_median=float(np.median(spreads)), rel_median=float(np.median(rels)),
                cv_unclipped_median=nm(cvU), cv_clipped_median=nm(cvC),
                share_q_unclipped_median=float(np.median(shqU)), share_q_unclipped_mean=float(np.mean(shqU)),
                share_r_unclipped_median=float(np.median(shrU)), share_r_unclipped_mean=float(np.mean(shrU)),
                pairs_unclipped_frac_mean=float(np.mean(fracU)),
                n_cv_clipped_defined=int(np.sum(np.isfinite(cvC))), n_round_proto=len(cvs),
                final_mean_acc=acc, final_worst_acc=worst)
