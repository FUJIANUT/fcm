"""E4: is the m=2 degeneracy caused by PCA whitening? Centralized FCM on the pooled support-skew data with the
repository's preprocessing (StandardScaler -> PCA whiten=True) versus the SAME pipeline without whitening,
and versus standardized raw pixels. Same partitions/inits as the paper (partition 1201+s, init 1201+s+77)."""
import sys, csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
def distinct(V, tol=1e-3):
    k=len(V); lab=list(range(k))
    def f(i):
        while lab[i]!=i: i=lab[i]
        return i
    for i in range(k):
        for j in range(i+1,k):
            if np.linalg.norm(V[i]-V[j])<tol: lab[f(i)]=f(j)
    return len({f(i) for i in range(k)})
def job(a):
    base, nc, lp, ms, variant, m, s = a
    sys.path.insert(0, str(ROOT))
    from fedfcmsim.datasets import make_feature_clients, load_openml_feature
    from fedfcmsim.fcm import fcm, initialize_centers_from_data, predict_membership
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    # the paper's partition (defines WHICH samples are pooled) — use the whitened loader only for indices
    cls, k = make_feature_clients(dataset=base, n_clients=nc, partition="support_skew",
                                  labels_per_client=lp, min_size=ms, random_state=1201+s)
    Xw = np.vstack([c.x for c in cls])
    if variant == "whitened (paper)":
        X = Xw
    else:
        if base.startswith("digits"):
            raw = StandardScaler().fit_transform(load_digits().data.astype(float))
        else:
            raw, _, _ = load_openml_feature("mnist_784")
        ncomp = int(base.split("pca")[1]) if "pca" in base else None
        if variant == "PCA, no whitening":
            full = PCA(n_components=ncomp, whiten=False, random_state=0).fit_transform(raw)
        else:
            full = raw
        # map pooled whitened rows back to dataset rows via the whitened full matrix
        if base.startswith("digits"):
            W = PCA(n_components=ncomp, whiten=True, random_state=0).fit_transform(raw)
        else:
            W = PCA(n_components=ncomp, whiten=True, random_state=0).fit_transform(raw)
        idx = {tuple(np.round(r, 8)): i for i, r in enumerate(W)}
        rows = [idx.get(tuple(np.round(r, 8))) for r in Xw]
        if any(r is None for r in rows): return dict(dataset=base, variant=variant, m=m, seed=s, c=k, d=-1, distinct=-1, entropy_norm=-1)
        X = full[rows]
    V0 = initialize_centers_from_data(X, k, random_state=1201+s+77)
    V = fcm(X, k, m=m, init_centers=V0, max_iter=300).centers
    u = predict_membership(X, V, m=m)
    H = float(-(u*np.log(np.maximum(u,1e-300))).sum(1).mean()/np.log(k))
    return dict(dataset=base, variant=variant, m=m, seed=s, c=k, d=X.shape[1], distinct=distinct(V), entropy_norm=round(H,4))
if __name__ == "__main__":
    CFG=[("digits_pca16",10,2,5),("digits_pca32",10,2,5),("mnist784_pca32",20,3,50)]
    V=["whitened (paper)","PCA, no whitening","standardized raw (no PCA)"]
    jobs=[(b,nc,lp,ms,v,m,s) for (b,nc,lp,ms) in CFG for v in V for m in (2.0,1.5,1.3,1.1) for s in range(3)
          if not (b.startswith("mnist") and v=="standardized raw (no PCA)" and m!=2.0)]
    with ProcessPoolExecutor(max_workers=3) as ex: rows=list(ex.map(job, jobs))
    out=ROOT/"results"/"t8_e4_unwhitened.csv"
    with open(out,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import collections
    g=collections.defaultdict(list)
    for r in rows: g[(r['dataset'],r['variant'],r['m'])].append(r)
    print(f"{'dataset':<16}{'variant':<28}{'d':>5}" + "".join(f"{'m='+str(m):>14}" for m in (2.0,1.5,1.3,1.1)))
    for (b,_,_,_) in CFG:
        for v in V:
            line=f"{b:<16}{v:<28}"; d=None
            for m in (2.0,1.5,1.3,1.1):
                rr=g.get((b,v,m))
                if not rr: line+=f"{'--':>14}"; continue
                d=rr[0]['d']; line+=f"{min(r['distinct'] for r in rr):>6}/{rr[0]['c']:<2} H{sum(r['entropy_norm'] for r in rr)/len(rr):.2f}"
            print(line.replace(f"{v:<28}",f"{v:<28}{d if d else '':>5}",1))
    print("\n(min distinct prototypes over 3 seeds / c ; H = mean normalized membership entropy, 1.00 = uniform)")
