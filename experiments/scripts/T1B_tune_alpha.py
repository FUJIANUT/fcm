"""Scratch: tune alpha for pedrycz_gradient_fcm on the iid scenario (sanity check A)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedfcmsim.fcm import fcm, predict_membership
from fedfcmsim.federated import federated_fcm, pedrycz_gradient_fcm
from fedfcmsim.metrics import clustering_accuracy
from fedfcmsim.synthetic import make_synthetic_clients

M = 2.0


def min_dist(centers):
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def client_accs(clients, centers):
    out = []
    for c in clients:
        lab = np.argmax(predict_membership(c.x, centers, m=M), axis=1)
        out.append(clustering_accuracy(c.y, lab))
    return float(np.mean(out)), float(np.min(out))


for alpha in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    am, wm, md, nr = [], [], [], []
    for seed in range(10):
        cls = make_synthetic_clients("iid", 12, 220, random_state=seed)
        r = np.random.default_rng(seed)
        pooled = np.vstack([c.x for c in cls])
        V0 = pooled[r.choice(len(pooled), 4, replace=False)].copy()
        res = pedrycz_gradient_fcm(cls, V0, rounds=50, local_steps=10, m=M, alpha=alpha)
        a, w = client_accs(cls, res.centers)
        am.append(a); wm.append(w); md.append(min_dist(res.centers)); nr.append(res.n_rounds)
    print(f"alpha={alpha:<6} min-dist={np.mean(md):.3f}  meanACC={np.mean(am):.4f}  "
          f"worstACC={np.mean(wm):.4f}  rounds={np.mean(nr):.1f}")

# centralized reference
am, wm, md = [], [], []
for seed in range(10):
    cls = make_synthetic_clients("iid", 12, 220, random_state=seed)
    r = np.random.default_rng(seed)
    pooled = np.vstack([c.x for c in cls])
    V0 = pooled[r.choice(len(pooled), 4, replace=False)].copy()
    res = fcm(pooled, 4, init_centers=V0, m=M)
    a, w = client_accs(cls, res.centers)
    am.append(a); wm.append(w); md.append(min_dist(res.centers))
print(f"centralized    min-dist={np.mean(md):.3f}  meanACC={np.mean(am):.4f}  worstACC={np.mean(wm):.4f}")
