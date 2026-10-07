# T4A — Diagnosis of the `digits_pca32` personalization anomaly

Script: `experiments/scripts/T4A_digits32_diagnosis.py` (5 seeds, T3 protocol:
SEED_OFFSET=1201, support_skew 10 clients × 2 labels, m=2.0, server_lr=0.75,
local_steps=5, 50 rounds). Raw data: `per_seed_metrics.csv` (720 rows),
`geometry.csv`.

## The anomaly

Personalization gain (`*_on` − `*_off`, mean ACC over 5 seeds), all six
mass×gate pairs behave identically within each setting:

| n_comp | whiten | post_none | post_footprint | post_mass | pre_none | pre_footprint | pre_mass |
|---|---|---|---|---|---|---|---|
| 8  | T | −0.243 | −0.019 | −0.033 | −0.326 | −0.029 | −0.069 |
| 8  | F | −0.209 | −0.028 | −0.038 | −0.361 | −0.024 | −0.035 |
| 16 | T | −0.117 | −0.054 | −0.099 | −0.194 | −0.001 | +0.060 |
| 16 | F | −0.243 | −0.077 | −0.064 | −0.312 | −0.021 | −0.095 |
| 24 | T | −0.095 | −0.082 | −0.121 | −0.152 | −0.040 | −0.132 |
| 24 | F | −0.257 | −0.070 | −0.162 | −0.338 | −0.029 | −0.125 |
| **32** | **T** | **+0.167** | **+0.158** | **+0.149** | **+0.175** | **+0.154** | **+0.145** |
| 32 | F | −0.242 | −0.033 | −0.140 | −0.284 | +0.018 | −0.057 |
| 48 | T | −0.193 | −0.082 | −0.148 | −0.170 | +0.002 | −0.067 |
| 48 | F | −0.268 | −0.089 | −0.190 | −0.256 | +0.028 | −0.143 |
| 64 | T | −0.265 | −0.201 | −0.249 | −0.254 | −0.123 | −0.206 |
| 64 | F | −0.263 | −0.129 | −0.224 | −0.288 | −0.019 | −0.173 |

The positive gain is a **knife-edge spike**: it exists only at
(n=32, whiten=True), reproducible across seeds (post_none_on 0.66 ± 0.04 vs
post_none_off 0.49 ± 0.01), and vanishes in BOTH directions — n=24 and
n=48 whitened both show the usual penalty.

## Verdicts on the hypotheses

### H1 (PCA whitening) — CONFIRMED as the necessary condition

- At n=32 the anomaly appears **only with whitening** (`+0.15` vs `−0.24`
  unwhitened). Whitening is necessary.
- Whitening inflates trailing near-noise components: the digits eigenvalue
  spectrum has only 17 components with λ>1 (λ: 7.3 … 0.42 by PC32, →0 by
  PC64). Whitened n=32 rescales ~15 sub-unit-variance components to unit
  variance; n=48/64 inflate ~31/~47 of them.
- Geometric fingerprints (`geometry.csv`): whitening doubles the pooled
  entropy effective rank at every n (e.g. 22.6→32.0 at n=32, 31.1→61.0 at
  n=64) and drives distance concentration up (pairwise mean/std 3.23 at
  n=64-whitened vs 2.77 unwhitened).
- **Consequence, the key finding**: for whitened n≥24 the mass-weighted
  federated run collapses to a *fully degenerate* global model — all 10
  prototypes coincide (min center distance ~1e-17, all 10 duplicated),
  because near-uniform memberships at V (entropy ≈ ln 10 = 2.30, measured
  at every whitened setting) make every local center ≈ the client mean.
  The "non-personalized baseline" of ~0.49 is the majority-class floor of
  a collapsed solution, not a real clustering. At n=32 unwhitened the same
  cells reach 0.71.

So the "+0.15 gain" is measured **against a broken baseline**. The anomaly
is not that personalization is strong here — it is that the baseline is
collapsed.

### H2 (distance concentration / intrinsic dimension) — CONFIRMED as mechanism

The whitened settings show exactly the predicted signatures (effective-rank
inflation, concentration, saturated memberships) and they coincide with the
prototype collapse above. The collapse is total for n≥24 whitened.

### H3 (partition interaction) — RULED OUT

The support-skew partition is the same construction used for all 14
datasets: client p gets labels {p, p+1} mod 10; every label is held by
exactly 2 clients (~89–92 samples each); per-client label histogram is a
perfect circulant band. Nothing dataset-specific, nothing unidentifiable.

### H4 (evaluation, not training) — PARTIALLY confirmed, but not the main effect

Scoring the *same* personalized run on the global V vs on V+δ_p
(`mean_acc_global` column): global-scored 0.54–0.63, personal-scored
0.57–0.71. Both beat the 0.49 floor, so personalization genuinely improves
the learned prototypes (per-client δ de-symmetrizes the collapsed fixed
point — coincident V plus distinct δ_p yield distinct local starts, and the
aggregate inherits the diversity), and roughly +0.05–0.10 of the gain is an
eval-side effect on top of that.

## Residual question — why exactly n=32?

Partially explained, not fully:

- Pure local FCM on each client's 2 classes is easy at all whitened n≥24
  (mean acc 0.82/0.92/0.90/0.86 at n=24/32/48/64), and the personalized
  prototypes DO land near the true class means at every n (nearest-proto
  distance 1.31/1.72/1.82/1.85). So the deltas find the right centers even
  at n=48/64.
- Yet personalized ACC collapses at n=48/64 (0.30/0.23). With ~31–47
  whitened noise dims, eval-time argmin over the 10 prototypes is hijacked
  by the 8 unsupported/drifted prototypes — the good prototype is present
  but loses the nearest-neighbor contest in the noise directions.
- At n≤24 whitened the non-personalized cells are only partially
  degenerate (n=16: 0–3/10 duplicated prototypes; n=24: collapse but
  0.59 acc via Hungarian luck), so there is less headroom and the usual
  personalization penalty dominates.

In other words, n=32-whitened sits at a boundary: collapse is total (maximal
headroom) while the noise-dim count (~15 inflated components) is still low
enough for the correct personalized prototype to win the argmin. The width
of that window is partly luck.

## Bottom line for the paper

`digits_pca32` should be reported as a **pathological cell**, not as
evidence that personalization helps: whitening at 32 components destroys
the federated global solution (coincident prototypes, uniform memberships),
and the measured "gain" is the recovery of easy 2-class local structure
against a collapsed ~0.49 baseline. Recommended handling: keep the dataset,
report the collapse (min-dist ≈ 0) alongside ACC, and note that the same
rescue disappears one PCA notch away in either direction.
