"""Independent re-computation of selected numbers in results/t10/theory/ (E_H.csv, E_I_descent.csv, counter-example).
Does NOT import T10_theory*. Uses T3 only for data/clients/init/gates/local steps (the harness), with FFCM_MSTAR=1,
T.M = m* per dataset, local_steps explicit. Own code for: memberships in J (log-domain), all E-H statistics, the
unclipped split, J_beta, and the full counter-example (pure numpy FCM, no fedfcmsim). Seeds 0-4 only; <= 6 workers.
  python3 verify_t10_theory.py  -> prints and writes verify_out.json next to this file."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ["FFCM_MSTAR"] = "1"
import sys, json, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[3]
THEORY = HERE.parent


def harness(label, seed):
    assert seed in range(5)
    os.environ["FFCM_MSTAR"] = "1"
    for k in ("FFCM_M", "FFCM_L", "FFCM_ROUNDS", "FFCM_SEED_START", "FFCM_NO_EARLYSTOP"):
        os.environ.pop(k, None)
    for p in (str(EXP / "scripts"), str(EXP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import T3_design_space as T
    from fedfcmsim.fcm import initialize_centers_from_data
    T.M = m = T.fuzzifier_for(label)
    specs = [("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] + [("real",) + tuple(d) for d in T.REAL_DATASETS]
    spec = [s for s in specs if s[1] == label][0]
    clients, k = T.build_clients(spec, seed)
    V0 = initialize_centers_from_data(np.vstack([c.x for c in clients]), k, random_state=T.SEED_OFFSET + seed + 77)
    n = np.array([len(c.x) for c in clients], float)
    return T, clients, k, V0, n / n.sum(), m


# ---- own FCM primitives (log-domain memberships) -------------------------------------------------------------------
def d2(X, V):
    return np.maximum((X * X).sum(1)[:, None] - 2 * X @ V.T + (V * V).sum(1)[None, :], 0.0)


def d2_exact(X, V):
    return ((X[:, None, :] - V[None, :, :]) ** 2).sum(2)


def memb(D, m):
    D = np.maximum(D, 1e-300)
    e = -np.log(D) / (m - 1.0)
    e -= e.max(1, keepdims=True)
    u = np.exp(e)
    return u / u.sum(1, keepdims=True)


def Jreduced(Xs, beta, V, m):
    tot = 0.0
    for b, X in zip(beta, Xs):
        D = d2_exact(X, V); tot += b * float(((memb(D, m) ** m) * D).sum())
    return tot


# ---- E-H ------------------------------------------------------------------------------------------------------------
def eh(arg):
    label, seed, mp, L = arg
    T, clients, k, V, beta, m = harness(label, seed)
    from fedfcmsim.fcm import predict_membership, run_local_fcm_steps
    rmin = T.MIN_RELEVANCE; P = len(clients); V0 = V.copy()
    cv_all, disp_l, spr_l, rel_l, cvU, cvC, shq, shr, frac = [], [], [], [], [], [], [], [], []
    for t in range(T.ROUNDS):
        mass = np.empty((P, k)); C = np.empty((P, k, V.shape[1])); r = np.empty((P, k)); q = np.empty((P, k))
        for p, c in enumerate(clients):
            ub = predict_membership(c.x, V, m=m)
            loc = run_local_fcm_steps(c.x, V, steps=L, m=m)
            mass[p] = (ub ** m).sum(0) if mp == "pre" else (loc.membership ** m).sum(0)
            C[p] = loc.centers; r[p] = T.gate_footprint(c.x, V, ub); q[p] = T.gate_mass(c.x, V, ub)
        unc = (r > rmin) & (q > rmin)          # strictly above the floor (no tolerance)
        frac.append(unc.mean())
        for j in range(k):
            W = beta * mass[:, j]
            a = W * q[:, j]; bb = W * r[:, j]
            if a.sum() <= 0 or bb.sum() <= 0:
                continue
            a = a / a.sum(); bb = bb / bb.sum(); g = r[:, j] / q[:, j]
            tq = a @ C[:, j]; tr = bb @ C[:, j]
            Eg = a @ g; Eg2 = a @ (g * g)
            cv_all.append(np.sqrt(max(Eg2 - Eg * Eg, 0.0)) / Eg)
            dsp = np.linalg.norm(tr - tq); spr = np.sqrt(a @ ((C[:, j] - tq) ** 2).sum(1))
            disp_l.append(dsp); spr_l.append(spr)
            if spr > 1e-12:
                rel_l.append(dsp / spr)
            for msk, st in ((unc[:, j], cvU), (~unc[:, j], cvC)):
                s = a[msk].sum()
                if msk.sum() >= 2 and s > 0:
                    w = a[msk] / s; e1 = w @ g[msk]; e2 = w @ (g[msk] ** 2)
                    st.append(np.sqrt(max(e2 - e1 * e1, 0.0)) / e1)
                else:
                    st.append(np.nan)
            shq.append(a[unc[:, j]].sum()); shr.append(bb[unc[:, j]].sum())
        w = beta[:, None] * mass * r; den = w.sum(0); Tt = V.copy(); ok = den > T.EPS
        Tt[ok] = np.einsum("pj,pjd->jd", w, C)[ok] / den[ok, None]
        V = (1 - T.SERVER_LR) * V + T.SERVER_LR * Tt
    Vrc, _, _ = T.run_cell(clients, V0, mp, "footprint", "off", rounds=T.ROUNDS, local_steps=L, slr=T.SERVER_LR)
    return dict(kind="eh", trajectory=f"{mp}_footprint_off_L{L}", dataset=label, seed=seed,
                tilt_cv_median=float(np.nanmedian(cv_all)), disp_median=float(np.median(disp_l)),
                spread_median=float(np.median(spr_l)), rel_median=float(np.median(rel_l)),
                cv_unclipped_median=float(np.nanmedian(cvU)), cv_clipped_median=float(np.nanmedian(cvC)),
                share_q_unclipped_mean=float(np.mean(shq)), share_r_unclipped_mean=float(np.mean(shr)),
                pairs_unclipped_frac_mean=float(np.mean(frac)), runcell_maxdiff=float(np.abs(V - Vrc).max()))


# ---- E-I descent ----------------------------------------------------------------------------------------------------
def ei(arg):
    label, seed, gate, L = arg
    T, clients, k, V, beta, m = harness(label, seed)
    from fedfcmsim.fcm import predict_membership, run_local_fcm_steps
    Xs = [c.x for c in clients]; V0 = V.copy(); alpha = T.SERVER_LR
    J = [Jreduced(Xs, beta, V, m)]; suff_viol = 0.0
    for t in range(T.ROUNDS):
        num = np.zeros_like(V); den = np.zeros(k)
        for p, c in enumerate(clients):
            ub = predict_membership(c.x, V, m=m)
            loc = run_local_fcm_steps(c.x, V, steps=L, m=m)
            w = beta[p] * (ub ** m).sum(0) * T.GATE_FNS[gate](c.x, V, ub)
            num += w[:, None] * loc.centers; den += w
        Tt = V.copy(); ok = den > T.EPS; Tt[ok] = num[ok] / den[ok, None]
        Vn = (1 - alpha) * V + alpha * Tt
        J.append(Jreduced(Xs, beta, Vn, m))
        if gate == "none" and L == 1:   # sufficient decrease: J(V)-J(V+) >= alpha(2-alpha) sum_j a_j ||T_j - V_j||^2
            bound = alpha * (2 - alpha) * float((den * ((Tt - V) ** 2).sum(1)).sum())
            suff_viol = max(suff_viol, (bound - (J[-2] - J[-1])) / max(J[-2], 1e-300))
        V = Vn
    Vrc, _, _ = T.run_cell(clients, V0, "pre", gate, "off", rounds=T.ROUNDS, local_steps=L, slr=alpha)
    J = np.array(J); dJ = np.diff(J); rel = dJ / J[:-1]
    return dict(kind="ei", rule=f"pre_{gate}_off_L{L}", dataset=label, seed=seed, J0=float(J[0]), J50=float(J[-1]),
                max_rel_increase=float(rel.max()), max_abs_increase=float(dJ.max()),
                n_rounds_increase_gt1e12=int((rel > 1e-12).sum()), runcell_maxdiff=float(np.abs(V - Vrc).max()),
                suff_decrease_max_rel_violation=suff_viol)


# ---- counter-example from scratch (pure numpy) -----------------------------------------------------------------------
def counterexample():
    rng = np.random.default_rng(1)
    A = rng.normal(0.0, 0.3, size=(40, 2)); B = rng.normal(np.array([4.0, 4.0]), 0.3, size=(40, 2))
    X = np.vstack([A, B]); m = 2.0; alpha = 0.75

    def U(X, V): return memb(d2_exact(X, V), m)
    def cstep(X, u): w = u ** m; return (w.T @ X) / w.sum(0)[:, None]
    def J(X, V, u=None):
        D = d2_exact(X, V); u = memb(D, m) if u is None else u; return float(((u ** m) * D).sum())

    V = np.vstack([A.mean(0), B.mean(0)])
    for it in range(100000):
        Vn = cstep(X, U(X, V))
        if np.linalg.norm(Vn - V) < 1e-13:
            V = Vn; break
        V = Vn
    Vs = V; Us = U(X, Vs); J0 = J(X, Vs)
    out = dict(n_iter=it + 1, V_star=Vs.tolist(), J_star=J0, rows=[])
    for L in (1, 2, 5):
        num = np.zeros_like(Vs); den = np.zeros(2)
        for x in (A, B):
            w = 0.5 * (U(x, Vs) ** m).sum(0)            # beta_p * Mbar_pj at V*
            c = Vs.copy()
            for _ in range(L):
                c = cstep(x, U(x, c))
            num += w[:, None] * c; den += w
        Tt = num / den[:, None]; Vp = (1 - alpha) * Vs + alpha * Tt
        out["rows"].append(dict(L=L, dT=float(np.linalg.norm(Tt - Vs)), dV=float(np.linalg.norm(Vp - Vs)),
                                dJ_reduced=J(X, Vp) - J0, dJ_fixedU=J(X, Vp, Us) - J0))
    return out


if __name__ == "__main__":
    EH = [(l, s, "pre", L) for l in ("wine", "pendigits", "digits_pca32") for s in (0, 3) for L in (5, 1)]
    EI = [(l, 2, g, L) for l in ("cluster_skew_overlap", "pendigits", "digits_pca16") for g, L in (("none", 1), ("mass", 1), ("none", 5))]
    jobs = [("eh", a) for a in EH] + [("ei", a) for a in EI]
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(eh if kd == "eh" else ei, a) for kd, a in jobs]
        cex = counterexample()
        res = [f.result() for f in futs]
    stored_eh = {(r["trajectory"], r["dataset"], int(r["seed"])): r for r in csv.DictReader(open(THEORY / "E_H.csv"))}
    stored_ei = {(r["rule"], r["dataset"], int(r["seed"])): r for r in csv.DictReader(open(THEORY / "E_I_descent.csv"))}
    for r in res:
        if r["kind"] == "eh":
            s = stored_eh[(r["trajectory"], r["dataset"], r["seed"])]
            r["maxdiff_vs_stored"] = max(abs(r[c] - float(s[c])) for c in r if c not in ("kind", "trajectory", "dataset", "seed", "runcell_maxdiff") and np.isfinite(r[c]))
        else:
            s = stored_ei[(r["rule"], r["dataset"], r["seed"])]
            r["stored_max_rel_increase"] = float(s["max_rel_increase"]); r["stored_n_gt1e12"] = int(s["n_rounds_increase_gt1e12"])
            r["J_traj_maxreldiff_vs_stored"] = None
            sj = np.array([float(x) for x in s["J_traj"].split(";")]) if "J_traj" in s else None
            r["J0_diff"] = abs(r["J0"] - float(s["J0"])); r["J50_diff"] = abs(r["J50"] - float(s["J50"]))
    out = dict(results=res, counterexample=cex)
    (HERE / "verify_out.json").write_text(json.dumps(out, indent=1, default=float))
    for r in res:
        print(json.dumps(r, default=float))
    print(json.dumps(cex, default=float))
