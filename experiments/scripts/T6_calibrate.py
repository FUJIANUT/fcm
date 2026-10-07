"""T6 calibration: pick step sizes for scffcm (eta_l, eta_g) and fednova_fcm
(server_lr) on the `iid` synthetic scenario ONLY -- never on support-skew
scenarios. 2 seeds, protocol identical to T3/T6 (SEED_OFFSET=1201, partition
rs=offset+seed, init rs=offset+seed+77, rounds=50, local_steps=5, m=2).

Selection rule: among non-diverging settings (finite centers AND final total
objective <= 5x the centralized-FCM objective), choose the one whose mean
client ACC is closest to centralized FCM; ties broken by |objective gap|.

Usage:
  python3 scripts/T6_calibrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedfcmsim.fcm import EPS, fcm, initialize_centers_from_data, predict_membership
from fedfcmsim.federated import fednova_fcm, scffcm
from fedfcmsim.metrics import clustering_accuracy
from fedfcmsim.synthetic import BASE_CENTERS, make_synthetic_clients

M = 2.0
ROUNDS = 50
LOCAL_STEPS = 5
SEED_OFFSET = 1201
SEEDS = [0, 1]

SCFFCM_GRID = [
    (eta_l, eta_g)
    for eta_l in (0.02, 0.05, 0.1, 0.2)
    for eta_g in (0.25, 0.5, 1.0, 2.0)
]
FEDNOVA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]


def mean_client_acc(clients, centers) -> float:
    accs = []
    for c in clients:
        labels = np.argmax(predict_membership(c.x, centers, m=M), axis=1)
        accs.append(clustering_accuracy(c.y, labels))
    return float(np.mean(accs))


def main() -> None:
    runs = []
    for seed in SEEDS:
        clients = make_synthetic_clients("iid", n_clients=12, samples_per_client=220,
                                       random_state=SEED_OFFSET + seed)
        pooled = np.vstack([c.x for c in clients])
        init = initialize_centers_from_data(pooled, BASE_CENTERS.shape[0],
                                            random_state=SEED_OFFSET + seed + 77)
        cen = fcm(pooled, BASE_CENTERS.shape[0], init_centers=init, m=M)
        ref_acc = mean_client_acc(clients, cen.centers)
        ref_obj = float(cen.objective_history[-1])
        print(f"seed={seed} centralized acc={ref_acc:.4f} obj={ref_obj:.4f}")

        for eta_l, eta_g in SCFFCM_GRID:
            res = scffcm(clients, init, rounds=ROUNDS, local_steps=LOCAL_STEPS, m=M,
                         eta_l=eta_l, eta_g=eta_g, seed=SEED_OFFSET + seed)
            obj = res.objective_history[-1] if res.objective_history else np.inf
            diverged = (not np.isfinite(res.centers).all()) or obj > 5.0 * ref_obj
            acc = mean_client_acc(clients, res.centers) if not diverged else float("nan")
            runs.append(("scffcm", f"eta_l={eta_l},eta_g={eta_g}", seed, acc, obj, diverged))

        for slr in FEDNOVA_GRID:
            res = fednova_fcm(clients, init, rounds=ROUNDS, local_steps=LOCAL_STEPS, m=M,
                              server_lr=slr)
            obj = res.objective_history[-1] if res.objective_history else np.inf
            diverged = (not np.isfinite(res.centers).all()) or obj > 5.0 * ref_obj
            acc = mean_client_acc(clients, res.centers) if not diverged else float("nan")
            runs.append(("fednova", f"server_lr={slr}", seed, acc, obj, diverged))

    print("\nmethod      setting                    seed  acc      obj        diverged")
    for method, setting, seed, acc, obj, div in runs:
        print(f"{method:11s} {setting:24s} {seed}     "
              f"{acc if np.isfinite(acc) else float('nan'):.4f}   {obj:10.4f}  {div}")

    print("\n--- selection (mean over seeds; among non-diverging on BOTH seeds) ---")
    refs = {}
    for seed in SEEDS:
        clients = make_synthetic_clients("iid", n_clients=12, samples_per_client=220,
                                       random_state=SEED_OFFSET + seed)
        pooled = np.vstack([c.x for c in clients])
        init = initialize_centers_from_data(pooled, BASE_CENTERS.shape[0],
                                            random_state=SEED_OFFSET + seed + 77)
        cen = fcm(pooled, BASE_CENTERS.shape[0], init_centers=init, m=M)
        refs[seed] = (mean_client_acc(clients, cen.centers), float(cen.objective_history[-1]))

    for method in ("scffcm", "fednova"):
        best = None
        settings = sorted({s for m_, s, _, _, _, _ in runs if m_ == method})
        for setting in settings:
            sub = [r for r in runs if r[0] == method and r[1] == setting]
            if any(r[5] for r in sub):
                continue
            accs = np.array([r[3] for r in sub])
            objs = np.array([r[4] for r in sub])
            acc_gap = float(np.mean([abs(a - refs[s][0]) for r, s, a in
                                     zip(sub, [r[2] for r in sub], accs)]))
            obj_gap = float(np.mean([abs(o - refs[s][1]) for s, o in
                                     zip([r[2] for r in sub], objs)]))
            key = (acc_gap, obj_gap)
            print(f"  {method} {setting:24s} mean_acc={accs.mean():.4f} "
                  f"acc_gap={acc_gap:.4f} obj_gap={obj_gap:.4f}")
            if best is None or key < best[0]:
                best = (key, setting)
        print(f"  => {method} frozen: {best[1] if best else 'ALL DIVERGED'}")


if __name__ == "__main__":
    main()
