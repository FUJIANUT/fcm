"""Label-free degeneracy data for the m* rule: CENTRALIZED FCM on the pooled support-skew data of each
real configuration, for m in {2.0,1.5,1.3,1.2,1.1} and 5 seeds (partition 1201+s, init 1201+s+77).
Writes figures/degeneracy_data.csv and a sentinel figures/degeneracy_data.DONE."""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "experiments"))
CFG = [("wine","wine",10,2,5),("satimage","satimage",10,3,5),("pendigits","pendigits",10,3,5),
       ("digits_pca16","digits_pca16",10,2,5),("letter","letter",10,6,50),("digits_pca32","digits_pca32",10,2,5),
       ("mnist784_pca32 (20c)","mnist784_pca32",20,3,50),("mnist784_pca32 (50c)","mnist784_pca32",50,3,20)]
GRID = [2.0, 1.5, 1.3, 1.2, 1.1]; SEEDS = range(5)

def distinct(V, tol=1e-3):
    k = len(V); lab = list(range(k))
    def f(i):
        while lab[i] != i: i = lab[i]
        return i
    for i in range(k):
        for j in range(i + 1, k):
            if np.linalg.norm(V[i] - V[j]) < tol: lab[f(i)] = f(j)
    return len({f(i) for i in range(k)})

def task(args):
    label, ds, nc, lp, ms, m, s = args
    sys.path.insert(0, str(ROOT.parent / "experiments"))
    from fedfcmsim.datasets import make_feature_clients
    from fedfcmsim.fcm import fcm, initialize_centers_from_data, predict_membership
    cls, k = make_feature_clients(dataset=ds, n_clients=nc, partition="support_skew",
                                  labels_per_client=lp, min_size=ms, random_state=1201 + s)
    X = np.vstack([c.x for c in cls])
    V0 = initialize_centers_from_data(X, k, random_state=1201 + s + 77)
    V = fcm(X, k, m=m, init_centers=V0, max_iter=300).centers
    u = predict_membership(X, V, m=m)
    H = float(-(u * np.log(np.maximum(u, 1e-300))).sum(1).mean() / np.log(k))
    return dict(dataset=label, m=m, seed=s, n=X.shape[0], d=X.shape[1], c=k,
                distinct=distinct(V), entropy_norm=round(H, 6))

if __name__ == "__main__":
    out = ROOT / "figures" / "degeneracy_data.csv"; done = ROOT / "figures" / "degeneracy_data.DONE"
    done.unlink(missing_ok=True)
    jobs = [(l, d, nc, lp, ms, m, s) for (l, d, nc, lp, ms) in CFG for m in GRID for s in SEEDS]
    rows = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, fu in enumerate(as_completed([ex.submit(task, j) for j in jobs]), 1):
            rows.append(fu.result())
            if i % 20 == 0: print(f"[{i}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: ([c[0] for c in CFG].index(r["dataset"]), -r["m"], r["seed"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    done.write_text(f"{len(rows)} rows, {time.time()-t0:.0f}s\n")
    print("wrote", out, len(rows), "rows")
