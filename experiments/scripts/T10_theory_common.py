"""Shared setup for the T10 theory diagnostics (E-H, E-I). Imports T3 with FFCM_MSTAR=1 and the protocol defaults
(50 rounds, seeds 0-4); every caller passes local_steps explicitly and sets T.M per dataset."""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "results" / "t10" / "theory"
CONFIGS = ["cluster_skew_hard", "cluster_skew_overlap", "dirichlet_0.03", "dirichlet_0.1", "overlap_noise",
           "quantity_skew_extreme", "wine", "satimage", "pendigits", "digits_pca16", "digits_pca32", "letter",
           "mnist784_pca32 (20c)", "mnist784_pca32 (50c)"]
SEEDS = [0, 1, 2, 3, 4]


def import_T3():
    """Import T3 with m* on and every other override cleared (protocol defaults)."""
    os.environ["FFCM_MSTAR"] = "1"
    for k in ("FFCM_M", "FFCM_L", "FFCM_ROUNDS", "FFCM_SEED_START", "FFCM_NO_EARLYSTOP"):
        os.environ.pop(k, None)
    for p in (str(HERE), str(EXP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import T3_design_space as T
    assert T.USE_MSTAR and T.ROUNDS == 50 and T.SERVER_LR == 0.75 and T.FORCE_M is None
    return T


def setup(label, seed):
    """(T, clients, k, init V, beta, m) for one configuration and seed; sets T.M = m*."""
    assert seed in SEEDS, "T10 theory diagnostics use seeds 0-4 only"
    T = import_T3()
    from fedfcmsim.fcm import initialize_centers_from_data
    import numpy as np
    T.M = m = T.fuzzifier_for(label)
    spec = next(s for s in ([("synthetic", x) for x in T.SYNTHETIC_SCENARIOS] +
                            [("real",) + tuple(d) for d in T.REAL_DATASETS]) if s[1] == label)
    clients, k = T.build_clients(spec, seed)
    X = np.vstack([c.x for c in clients])
    V = initialize_centers_from_data(X, k, random_state=T.SEED_OFFSET + seed + 77)
    N = np.array([len(c.x) for c in clients], float)
    return T, clients, k, V, N / N.sum(), m
