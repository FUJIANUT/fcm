"""T1b rerun at STABLE alpha=0.01.

Identical to scripts/T1B_pedrycz_gradient.py except ALPHA = 0.01 instead of
the diverging 0.1. Overwrites results/t1b_pedrycz_gradient/t1b_table.md and
per_seed_metrics.csv -- the previous versions were run at alpha=0.1 where the
Pedrycz arm numerically diverges (min-dist 1e14-1e37 next to ACC 0.92), so
those numbers were self-contradictory and must not stand.
alpha_sensitivity.md is untouched.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedfcmsim.fcm import fcm, predict_membership
from fedfcmsim.federated import federated_fcm, pedrycz_gradient_fcm
from fedfcmsim.metrics import clustering_accuracy
from fedfcmsim.synthetic import make_synthetic_clients

M = 2.0
N_CLIENTS = 12
SAMPLES_PER_CLIENT = 220
N_SEEDS = 10
N_CLUSTERS = 4
ALPHA = 0.01  # stable alpha; the original run used 0.1 which diverges

SCENARIOS = [
    "iid",
    "cluster_skew_hard",
    "cluster_skew_overlap",
    "dirichlet_0.03",
    "dirichlet_0.1",
    "overlap_noise",
    "quantity_skew_extreme",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "t1b_pedrycz_gradient"


def min_dist(centers: np.ndarray) -> float:
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def client_accs(clients, centers) -> tuple[float, float]:
    accs = []
    for c in clients:
        labels = np.argmax(predict_membership(c.x, centers, m=M), axis=1)
        accs.append(clustering_accuracy(c.y, labels))
    return float(np.mean(accs)), float(np.min(accs))


def evaluate(clients, centers) -> dict[str, float]:
    mean_acc, worst_acc = client_accs(clients, centers)
    return {"min_dist": min_dist(centers), "mean_acc": mean_acc, "worst_acc": worst_acc}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            clients = make_synthetic_clients(
                scenario, N_CLIENTS, SAMPLES_PER_CLIENT, random_state=seed
            )
            rng = np.random.default_rng(seed)
            pooled = np.vstack([c.x for c in clients])
            init = pooled[rng.choice(len(pooled), N_CLUSTERS, replace=False)].copy()

            arms = {
                "pedrycz_gradient": pedrycz_gradient_fcm(
                    clients, init, rounds=50, local_steps=10, m=M, alpha=ALPHA
                ).centers,
                "federated_fcm_L5": federated_fcm(
                    clients, init, rounds=50, local_steps=5, m=M
                ).centers,
                "federated_fcm_L10": federated_fcm(
                    clients, init, rounds=50, local_steps=10, m=M
                ).centers,
                "centralized_fcm": fcm(pooled, N_CLUSTERS, init_centers=init, m=M).centers,
            }
            for method, centers in arms.items():
                metrics = evaluate(clients, centers)
                rows.append({"scenario": scenario, "seed": seed, "method": method, **metrics})
        print(f"done {scenario}", flush=True)

    csv_path = OUT_DIR / "per_seed_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["scenario", "seed", "method", "min_dist", "mean_acc", "worst_acc"]
        )
        writer.writeheader()
        writer.writerows(rows)

    methods = ["pedrycz_gradient", "federated_fcm_L5", "federated_fcm_L10", "centralized_fcm"]
    lines = [
        f"alpha={ALPHA} (stable; the previous version of this table used alpha=0.1, at which "
        f"pedrycz_gradient diverges -- min-dist 1e14-1e37 alongside ACC 0.92, self-contradictory); "
        f"L=10 local steps for pedrycz; 12 clients x 220 samples; {N_SEEDS} seeds; m=2.0; "
        f"true BASE_CENTERS min-dist = 7.5",
        "",
        "| scenario | method | min-dist | mean ACC | worst ACC |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for method in methods:
            sub = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            md = np.array([r["min_dist"] for r in sub])
            ma = np.array([r["mean_acc"] for r in sub])
            wa = np.array([r["worst_acc"] for r in sub])
            lines.append(
                f"| {scenario} | {method} | {md.mean():.3f} +/- {md.std():.3f} "
                f"| {ma.mean():.4f} | {wa.mean():.4f} |"
            )
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "t1b_table.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"wrote {csv_path} and {OUT_DIR / 't1b_table.md'}")


if __name__ == "__main__":
    main()
