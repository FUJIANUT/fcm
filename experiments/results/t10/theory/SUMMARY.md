# T10 theory checks (addendum §3: E-H, E-I) — m*, seeds 0-4, 50 rounds
Generated 2026-09-24 10:09 by `scripts/T10_theory.py` (helpers `T10_theory_common.py`, `T10_theory_eh.py`, `T10_theory_ei.py`). Raw: `E_H.csv` (per seed), `E_H_config.csv` (config means), `E_I_descent.csv`, `E_I_counterexample.txt`.

## E-H — one-round tilt diagnostics along three footprint-gated trajectories
Definitions as in T9_theory_link_b: weights W_pj = beta_p * mass_pj (post-adaptation mass for POST, pre-adaptation mass for PRE); base pi^q = W q (mass gate), server weights pi^r = W r (footprint gate); tilt g = r/q; CV = weighted (pi^q) coefficient of variation of g over clients; disp = ||target(footprint) - target(mass)||; spread = pi^q-weighted RMS distance of the local centres to target(mass); rel = disp/spread. Per seed: median over (round, prototype); table = mean over seeds 0-4. Unclipped pairs: r > r_min and q > r_min; clipped: either at the floor r_min = 0.05. CV_U / CV_C = the same weighted CV within each subset (pi^q renormalized; undefined when < 2 clients in the subset, then skipped); share_q(U) / share_r(U) = mean share of the mass-gated / footprint-gated (= server-update) weight carried by unclipped pairs.

**Reproduction of T9 (post_footprint_off, L=5)**: 70/70 (config, seed) rows; max |diff| vs `results/t9_theory_link_b.csv`: tilt_cv_median 0.0e+00, disp_median 0.0e+00, spread_median 0.0e+00, rel_median 0.0e+00.
**Trajectory check**: max |final mean ACC of the logged trajectory - stored per-seed ACC| = post_footprint_off_L5: 0.0e+00, pre_footprint_off_L5: 0.0e+00, pre_footprint_off_L1: 0.0e+00 (stored: T7 selection for L=5; `__L1all.csv` for L=1 — so `__L1all` is confirmed to be at m*).

### post_footprint_off_L5

| config | m* | CV (all) | disp | spread | disp/spread | CV unclipped | CV clipped | share_q(U) | share_r(U) | frac pairs U | |ACC fp - ACC mass| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster_skew_hard | 2.0 | 0.025 | 0.0011 | 2.576 | 0.000 | 0.021 | 0.000 | 0.927 | 0.927 | 0.499 | 0.0000 |
| cluster_skew_overlap | 2.0 | 0.052 | 0.0287 | 1.449 | 0.020 | 0.050 | 0.045 | 0.958 | 0.959 | 0.678 | 0.0005 |
| dirichlet_0.03 | 2.0 | 0.025 | 0.0179 | 3.015 | 0.009 | 0.021 | 0.000 | 0.905 | 0.903 | 0.358 | 0.0009 |
| dirichlet_0.1 | 2.0 | 0.035 | 0.0176 | 1.737 | 0.010 | 0.027 | 0.013 | 0.953 | 0.953 | 0.505 | 0.0006 |
| overlap_noise | 2.0 | 0.056 | 0.0218 | 0.853 | 0.024 | 0.051 | 0.076 | 0.985 | 0.986 | 0.774 | 0.0015 |
| quantity_skew_extreme | 2.0 | 0.053 | 0.0081 | 0.669 | 0.012 | 0.043 | 0.148 | 0.913 | 0.928 | 0.645 | 0.0026 |
| wine | 2.0 | 0.153 | 0.0966 | 1.135 | 0.081 | 0.142 | 0.149 | 0.983 | 0.993 | 0.849 | 0.0044 |
| satimage | 2.0 | 0.062 | 0.0316 | 1.173 | 0.025 | 0.061 | 0.004 | 0.981 | 0.981 | 0.678 | 0.0008 |
| pendigits | 1.5 | 0.131 | 0.0935 | 1.180 | 0.079 | 0.123 | 0.126 | 0.956 | 0.961 | 0.594 | 0.0097 |
| digits_pca16 | 1.3 | 0.146 | 0.1152 | 1.524 | 0.074 | 0.126 | 0.188 | 0.937 | 0.942 | 0.522 | 0.0304 |
| digits_pca32 | 1.1 | 0.131 | 0.1291 | 2.171 | 0.062 | 0.121 | 0.145 | 0.935 | 0.936 | 0.533 | 0.0406 |
| letter | 1.3 | 0.124 | 0.0806 | 1.389 | 0.057 | 0.123 | 0.070 | 0.969 | 0.971 | 0.766 | 0.0006 |
| mnist784_pca32 (20c) | 1.1 | 0.163 | 0.1529 | 2.022 | 0.084 | 0.152 | 0.153 | 0.931 | 0.920 | 0.724 | 0.0101 |
| mnist784_pca32 (50c) | 1.1 | 0.156 | 0.1374 | 2.036 | 0.070 | 0.147 | 0.168 | 0.943 | 0.940 | 0.765 | 0.0228 |

Ranges over the 14 configurations: CV 0.025-0.163; disp/spread 0.000-0.084; CV unclipped 0.021-0.152; CV clipped 0.000-0.188; share_q(U) 0.905-0.985; share_r(U) 0.903-0.993. Gap source: T7 (main/big m=2, _mstar rescued), seeds 0-9.

Spearman across the 14 configurations vs |ACC(footprint) - ACC(mass)| of the matching cells (scipy, n = 14):
- weighted tilt CV (all pairs): rho = +0.780, p = 0.0010 (n = 14)
- disp/spread: rho = +0.705, p = 0.0048 (n = 14)
- weighted tilt CV (unclipped pairs): rho = +0.657, p = 0.0107 (n = 14)
- weighted tilt CV (clipped pairs): rho = +0.829, p = 0.0003 (n = 14)
- server-weight share of unclipped pairs: rho = -0.138, p = 0.6369 (n = 14)

### pre_footprint_off_L5

| config | m* | CV (all) | disp | spread | disp/spread | CV unclipped | CV clipped | share_q(U) | share_r(U) | frac pairs U | |ACC fp - ACC mass| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster_skew_hard | 2.0 | 0.050 | 0.0035 | 0.463 | 0.007 | 0.042 | 0.009 | 0.999 | 0.999 | 0.518 | 0.0191 |
| cluster_skew_overlap | 2.0 | 0.027 | 0.0084 | 0.641 | 0.012 | 0.027 | 0.043 | 0.998 | 0.998 | 0.681 | 0.0138 |
| dirichlet_0.03 | 2.0 | 0.012 | 0.0027 | 0.407 | 0.003 | 0.010 | 0.009 | 0.999 | 1.000 | 0.375 | 0.0001 |
| dirichlet_0.1 | 2.0 | 0.028 | 0.0101 | 0.501 | 0.011 | 0.024 | 0.005 | 0.998 | 0.998 | 0.486 | 0.0010 |
| overlap_noise | 2.0 | 0.045 | 0.0077 | 0.514 | 0.014 | 0.044 | 0.113 | 0.997 | 0.998 | 0.793 | 0.0006 |
| quantity_skew_extreme | 2.0 | 0.026 | 0.0038 | 0.449 | 0.011 | 0.023 | 0.071 | 0.989 | 0.989 | 0.629 | 0.0161 |
| wine | 2.0 | 0.119 | 0.0475 | 0.973 | 0.047 | 0.113 | 0.156 | 0.995 | 0.998 | 0.838 | 0.0006 |
| satimage | 2.0 | 0.054 | 0.0101 | 0.539 | 0.017 | 0.054 | 0.007 | 0.998 | 0.998 | 0.686 | 0.0048 |
| pendigits | 1.5 | 0.091 | 0.0411 | 0.666 | 0.054 | 0.082 | 0.135 | 0.989 | 0.992 | 0.566 | 0.0150 |
| digits_pca16 | 1.3 | 0.099 | 0.0508 | 0.948 | 0.052 | 0.094 | 0.165 | 0.984 | 0.987 | 0.515 | 0.0017 |
| digits_pca32 | 1.1 | 0.105 | 0.0895 | 1.558 | 0.054 | 0.100 | 0.127 | 0.949 | 0.950 | 0.514 | 0.0053 |
| letter | 1.3 | 0.103 | 0.0493 | 1.043 | 0.049 | 0.103 | 0.076 | 0.992 | 0.993 | 0.706 | 0.0014 |
| mnist784_pca32 (20c) | 1.1 | 0.122 | 0.0896 | 1.602 | 0.056 | 0.107 | 0.123 | 0.950 | 0.943 | 0.632 | 0.0186 |
| mnist784_pca32 (50c) | 1.1 | 0.132 | 0.0754 | 1.681 | 0.051 | 0.118 | 0.154 | 0.949 | 0.946 | 0.685 | 0.0102 |

Ranges over the 14 configurations: CV 0.012-0.132; disp/spread 0.003-0.056; CV unclipped 0.010-0.118; CV clipped 0.005-0.165; share_q(U) 0.949-0.999; share_r(U) 0.943-1.000. Gap source: T7 (main/big m=2, _mstar rescued), seeds 0-9.

Spearman across the 14 configurations vs |ACC(footprint) - ACC(mass)| of the matching cells (scipy, n = 14):
- weighted tilt CV (all pairs): rho = +0.134, p = 0.6477 (n = 14)
- disp/spread: rho = +0.248, p = 0.3919 (n = 14)
- weighted tilt CV (unclipped pairs): rho = +0.051, p = 0.8637 (n = 14)
- weighted tilt CV (clipped pairs): rho = +0.029, p = 0.9228 (n = 14)
- server-weight share of unclipped pairs: rho = -0.459, p = 0.0985 (n = 14)

### pre_footprint_off_L1

| config | m* | CV (all) | disp | spread | disp/spread | CV unclipped | CV clipped | share_q(U) | share_r(U) | frac pairs U | |ACC fp - ACC mass| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster_skew_hard | 2.0 | 0.010 | 0.0003 | 0.150 | 0.001 | 0.009 | 0.000 | 0.999 | 0.999 | 0.496 | 0.0000 |
| cluster_skew_overlap | 2.0 | 0.026 | 0.0029 | 0.328 | 0.008 | 0.025 | 0.029 | 0.998 | 0.998 | 0.693 | 0.0001 |
| dirichlet_0.03 | 2.0 | 0.005 | 0.0000 | 0.082 | 0.000 | 0.003 | 0.000 | 0.999 | 1.000 | 0.347 | 0.0104 |
| dirichlet_0.1 | 2.0 | 0.025 | 0.0082 | 0.412 | 0.010 | 0.016 | 0.013 | 0.998 | 0.999 | 0.490 | 0.0190 |
| overlap_noise | 2.0 | 0.041 | 0.0025 | 0.209 | 0.012 | 0.040 | 0.076 | 0.997 | 0.998 | 0.772 | 0.0186 |
| quantity_skew_extreme | 2.0 | 0.015 | 0.0012 | 0.133 | 0.007 | 0.015 | 0.078 | 0.991 | 0.991 | 0.637 | 0.0007 |
| wine | 2.0 | 0.116 | 0.0347 | 0.785 | 0.042 | 0.108 | 0.105 | 0.994 | 0.998 | 0.840 | 0.0005 |
| satimage | 2.0 | 0.056 | 0.0047 | 0.347 | 0.016 | 0.055 | 0.030 | 0.993 | 0.994 | 0.642 | 0.0032 |
| pendigits | 1.5 | 0.072 | 0.0150 | 0.382 | 0.039 | 0.064 | 0.169 | 0.989 | 0.991 | 0.519 | 0.0090 |
| digits_pca16 | 1.3 | 0.091 | 0.0213 | 0.545 | 0.040 | 0.082 | 0.180 | 0.984 | 0.985 | 0.391 | 0.0078 |
| digits_pca32 | 1.1 | 0.086 | 0.0301 | 0.653 | 0.046 | 0.081 | 0.072 | 0.984 | 0.984 | 0.387 | 0.0072 |
| letter | 1.3 | 0.089 | 0.0241 | 0.626 | 0.040 | 0.090 | 0.080 | 0.991 | 0.992 | 0.704 | 0.0019 |
| mnist784_pca32 (20c) | 1.1 | 0.115 | 0.0339 | 0.720 | 0.048 | 0.107 | 0.148 | 0.968 | 0.963 | 0.651 | 0.0035 |
| mnist784_pca32 (50c) | 1.1 | 0.124 | 0.0371 | 0.768 | 0.045 | 0.122 | 0.133 | 0.980 | 0.979 | 0.689 | 0.0034 |

Ranges over the 14 configurations: CV 0.005-0.124; disp/spread 0.000-0.048; CV unclipped 0.003-0.122; CV clipped 0.000-0.180; share_q(U) 0.968-0.999; share_r(U) 0.963-1.000. Gap source: __L1all, seeds 0-9.

Spearman across the 14 configurations vs |ACC(footprint) - ACC(mass)| of the matching cells (scipy, n = 14):
- weighted tilt CV (all pairs): rho = -0.059, p = 0.8403 (n = 14)
- disp/spread: rho = +0.046, p = 0.8755 (n = 14)
- weighted tilt CV (unclipped pairs): rho = -0.081, p = 0.7823 (n = 14)
- weighted tilt CV (clipped pairs): rho = +0.112, p = 0.7028 (n = 14)
- server-weight share of unclipped pairs: rho = -0.077, p = 0.7938 (n = 14)

## E-I (1) — descent of J_beta along the damped PRE rules
J_beta(V_t) = sum_p beta_p J_p(U_t, V_t), U_t = the FCM-optimal memberships at V_t (reduced objective), t = 0..50; relative increase = (J_{t+1} - J_t)/J_t. The logged loop is checked against an independent T3.run_cell(mp, gate, 'off', local_steps=L) call (final V).

| rule | runs | max rel. increase | max abs. increase | rounds with increase > 0 | rounds with rel. increase > 1e-12 | runs with any increase > 1e-12 | max |V - V_run_cell| |
|---|---|---|---|---|---|---|---|
| pre_none_off_L1 | 70 | 4.398e-16 | 2.274e-13 | 94/3500 (2.7%) | 0/3500 (0.0%) | 0/70 | 0.0e+00 |
| pre_mass_off_L1 | 70 | 2.654e-02 | 2.227e+01 | 1292/3500 (36.9%) | 1158/3500 (33.1%) | 57/70 | 0.0e+00 |
| pre_none_off_L5 | 70 | 4.358e-02 | 3.681e+01 | 1463/3500 (41.8%) | 1411/3500 (40.3%) | 60/70 | 0.0e+00 |

Where pre_none_off_L1 increases (rel > 1e-12), per configuration: runs with an increase / rounds with an increase / max over rounds of the relative change (J_{t+1}-J_t)/J_t, negative = J decreased in every round [first increase round range]:
- cluster_skew_hard: 0/5 runs, 0/250 rounds, max 2.53e-16
- cluster_skew_overlap: 0/5 runs, 0/250 rounds, max -9.60e-12
- dirichlet_0.03: 0/5 runs, 0/250 rounds, max 2.44e-16
- dirichlet_0.1: 0/5 runs, 0/250 rounds, max 2.56e-16
- overlap_noise: 0/5 runs, 0/250 rounds, max 4.40e-16
- quantity_skew_extreme: 0/5 runs, 0/250 rounds, max 1.88e-16
- wine: 0/5 runs, 0/250 rounds, max 3.90e-16
- satimage: 0/5 runs, 0/250 rounds, max -1.34e-10
- pendigits: 0/5 runs, 0/250 rounds, max -8.80e-11
- digits_pca16: 0/5 runs, 0/250 rounds, max -1.55e-06
- digits_pca32: 0/5 runs, 0/250 rounds, max -1.81e-05
- letter: 0/5 runs, 0/250 rounds, max -5.14e-05
- mnist784_pca32 (20c): 0/5 runs, 0/250 rounds, max -4.76e-06
- mnist784_pca32 (50c): 0/5 runs, 0/250 rounds, max -2.47e-06

Where pre_mass_off_L1 increases (rel > 1e-12), per configuration: runs with an increase / rounds with an increase / max over rounds of the relative change (J_{t+1}-J_t)/J_t, negative = J decreased in every round [first increase round range]:
- cluster_skew_hard: 3/5 runs, 41/250 rounds, max 8.94e-06 [first increase at round 6-10]
- cluster_skew_overlap: 4/5 runs, 91/250 rounds, max 2.64e-03 [first increase at round 5-22]
- dirichlet_0.03: 5/5 runs, 70/250 rounds, max 1.71e-03 [first increase at round 7-12]
- dirichlet_0.1: 3/5 runs, 37/250 rounds, max 1.56e-03 [first increase at round 6-8]
- overlap_noise: 2/5 runs, 43/250 rounds, max 1.77e-04 [first increase at round 9-20]
- quantity_skew_extreme: 4/5 runs, 86/250 rounds, max 2.65e-02 [first increase at round 6-10]
- wine: 5/5 runs, 177/250 rounds, max 1.92e-03 [first increase at round 5-13]
- satimage: 5/5 runs, 149/250 rounds, max 2.27e-04 [first increase at round 9-35]
- pendigits: 5/5 runs, 146/250 rounds, max 1.43e-03 [first increase at round 8-26]
- digits_pca16: 4/5 runs, 12/250 rounds, max 1.86e-03 [first increase at round 5-12]
- digits_pca32: 4/5 runs, 59/250 rounds, max 1.59e-04 [first increase at round 10-33]
- letter: 5/5 runs, 95/250 rounds, max 8.29e-04 [first increase at round 13-39]
- mnist784_pca32 (20c): 4/5 runs, 72/250 rounds, max 1.53e-04 [first increase at round 23-43]
- mnist784_pca32 (50c): 4/5 runs, 80/250 rounds, max 1.11e-04 [first increase at round 16-32]

Where pre_none_off_L5 increases (rel > 1e-12), per configuration: runs with an increase / rounds with an increase / max over rounds of the relative change (J_{t+1}-J_t)/J_t, negative = J decreased in every round [first increase round range]:
- cluster_skew_hard: 5/5 runs, 148/250 rounds, max 2.37e-02 [first increase at round 3-8]
- cluster_skew_overlap: 5/5 runs, 150/250 rounds, max 3.15e-03 [first increase at round 8-25]
- dirichlet_0.03: 5/5 runs, 83/250 rounds, max 4.36e-02 [first increase at round 5-34]
- dirichlet_0.1: 4/5 runs, 94/250 rounds, max 3.29e-02 [first increase at round 5-9]
- overlap_noise: 4/5 runs, 107/250 rounds, max 1.43e-03 [first increase at round 4-13]
- quantity_skew_extreme: 3/5 runs, 67/250 rounds, max 2.19e-03 [first increase at round 6-20]
- wine: 3/5 runs, 66/250 rounds, max 7.81e-04 [first increase at round 5-9]
- satimage: 2/5 runs, 72/250 rounds, max 3.19e-03 [first increase at round 7-7]
- pendigits: 5/5 runs, 104/250 rounds, max 2.19e-03 [first increase at round 7-33]
- digits_pca16: 5/5 runs, 113/250 rounds, max 1.75e-03 [first increase at round 5-19]
- digits_pca32: 5/5 runs, 111/250 rounds, max 1.13e-03 [first increase at round 6-13]
- letter: 5/5 runs, 77/250 rounds, max 3.29e-04 [first increase at round 10-33]
- mnist784_pca32 (20c): 4/5 runs, 116/250 rounds, max 4.27e-04 [first increase at round 10-18]
- mnist784_pca32 (50c): 5/5 runs, 103/250 rounds, max 8.68e-04 [first increase at round 5-12]

## E-I (2) — counter-example (two blobs, one per client, m = 2)

| V* | L | ||V+ - V*|| (damped) | ||T - V*|| (undamped target) | dJ pooled (reduced) | dJ (fixed U*) |
|---|---|---|---|---|---|
| 30 iterations | 1 | 0.000000 | 0.000000 | +0.000000 | +0.000000 |
| 30 iterations | 2 | 0.103843 | 0.138457 | +0.422833 | +0.427526 |
| 30 iterations | 5 | 0.195056 | 0.260075 | +1.490405 | +1.508509 |
| converged tol 1e-12 | 1 | 0.000000 | 0.000000 | -0.000000 | +0.000000 |
| converged tol 1e-12 | 2 | 0.103843 | 0.138457 | +0.422833 | +0.427526 |
| converged tol 1e-12 | 5 | 0.195056 | 0.260075 | +1.490405 | +1.508509 |

J(V*) = 11.236762 (both fixed points; FCM from the sample means converges in 6 iterations at tol 1e-12). Drawing variants `rng.normal(loc, 0.3, size)`, `0.3*standard_normal + mu` and `multivariate_normal` give identical numbers; a column-major draw gives J(V*) = 11.137593, dJ(L=2) = +0.397952, dJ(L=5) = +1.500825 (same qualitative result).
**Verdict:** reproduced. J(V*) = 11.236762 and dJ = 0 / +0.422833 / +1.490405 for L = 1 / 2 / 5 match exactly; the referee's displacement figures 0.138457 / 0.260075 are the undamped target displacement ||T - V*||; the damped iterate moves by 0.75 times that (0.103843 / 0.195056). L = 1 leaves V* fixed (PRE at L=1 is exactly the pooled FCM step), L >= 2 increases the pooled objective: the damped PRE rule is not a descent method for L >= 2.
