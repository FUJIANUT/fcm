# T10 — pre-registration of the fresh seed block 20-29 and of the review-driven experiments

Written 2026-09-24 BEFORE any run on seeds 20-29. No result on seeds 20-29 existed when this file was written
(checked 2026-09-24 08:04 EDT: no CSV in `results/` contains a design-space seed index >= 20 other than the calibration
seed 100; the only larger values, 12401 and 12408 in `overhead_*/metrics.csv`, are raw random states of the accepted
paper's runtime-overhead runs, a different protocol). Motivation: the external review (Grok 4.7, FODM referee brief)
correctly noted that the L=1 and SC-FFCM counts were a second look at seeds 10-19, which the first held-out block had
already unblinded, and that the per-configuration counts lead the abstract without the Holm adjustment.

## 1. Fresh confirmatory block: seeds 20-29 (run once, analysed once, no re-runs after looking)

Protocol: identical to the paper (T3 at m*, 50 rounds, server step 0.75, delta_lr 0.7, delta_reg 0.05, r_min 0.05,
G = 60, partition seed 1201+s, init 1201+s+77); SC-FFCM with the step sizes ALREADY calibrated on seed 100
(`results/t9b_scffcm_calibration.csv` for L in {1,5}; `results/t9c_scffcm_calibration.csv` for L = 50) — no
re-calibration. Metric: mean client ACC (per-client Hungarian); secondary: pooled FCM objective (`fcm_obj`).
Test: two-sided paired Wilcoxon over the 10 seeds per configuration, level 0.05; per-hypothesis Holm adjustment over
the 14 configurations (12 for L = 50). Report "better/worse" counts unadjusted AND Holm-adjusted.

Primary hypotheses (direction = what the earlier blocks showed; a hypothesis "replicates" if the Holm-adjusted count in
the stated direction is >= 1 and no Holm-significant count in the opposite direction exceeds it):
- H1 (L=5) pre_mass_off vs GF-PFedFCM (post_footprint_on): pre_mass_off better.
- H2 (L=5) pre_mass_off vs F-FCM (post_none_off): pre_mass_off better.
- H3 (L=5) substitution interaction (pre_none - post_none) - (pre_footprint - post_footprint), all personalization off:
  positive.
- H4 (L=1) PRE (pre_none_off) vs POST (post_none_off): PRE better.
- H5 (L=1) pre_mass_off vs GF-PFedFCM (both L=1): pre_mass_off better.
- H6 pre_mass_off (L=1) vs SC-FFCM (L=1): pre_mass_off better.
- H7 pre_mass_off (L=1) vs SC-FFCM (L=5): pre_mass_off better.
- H8 (L=5) pre_mass_off vs SC-FFCM (L=5): SC-FFCM better.
- H9 (L=5) personalization, post_footprint_on vs post_footprint_off: no improvement in ACC (reported as counts).
Secondary (reported, not used for "replicates"): the same pairs on `fcm_obj`; pre_mass_off (L=1) vs SC-FFCM (L=50) on
the 12 non-MNIST configurations; pre_none_off (L=1) vs SC-FFCM (L=1 and L=5).

## 2. Review-driven experiments (exploratory unless stated; seeds 0-9 and 10-19 unless stated)

- E-A Substitution mechanism. Two support-MASK gates with no fuzzy-mass magnitude: `fpmask` (1 where the footprint
  relevance exceeds its floor r_min, else r_min) and `massmask` (1 where the normalized mass q exceeds r_min, else
  r_min) — i.e. the clipped/unclipped split of each existing gate. Interaction (pre_none - post_none) - (pre_G - post_G)
  for G in {footprint, mass, fpmask, massmask} at L = 5; also on seeds 20-29 (secondary).
- E-B Evaluation criteria. New metrics for every run: `gacc` (one Hungarian alignment over the pooled data, each point
  assigned by its own client's evaluation prototypes), `xb` (Xie-Beni index of the shared prototypes on the pooled
  data), `hnorm` (mean normalized membership entropy under the evaluation prototypes). Re-runs of the 12 cells at L in
  {1,5} on seeds 0-19 must reproduce the existing `mean_acc` bit for bit.
- E-C Split-sample personalization check: each client's data split 80/20 (seeded); fit on 80%, evaluate `fcm_obj`
  and ACC on the 20% for post_footprint_off/on and pre_mass_off/on (L=5).
- E-D SC-FFCM convergence and compute matching: SC-FFCM at L=1 with R=250 rounds (250 local passes, as for L=5, R=50),
  re-calibrated on seed 100 on a grid extended to eta_l <= 16, eta_g <= 32; pre_mass_off and pre_none_off at L=1,
  R=250. Seeds 10-29. Also SC-FFCM at the released default steps (eta_l 0.2, eta_g 0.5) at L in {1,5,50}.
- E-E Pedrycz's gradient federated FCM (Pedrycz 2022) as a baseline: step alpha calibrated on seed 100 with the same
  label-free criterion as SC-FFCM; L in {1,5,10}; seeds 0-19.
- E-F m = 2 audit robustness: centralized FCM at every m in the grid from (a) the paper's random data-point init,
  (b) k-means++ best of 10 restarts by FCM objective, (c) oracle class-mean init; distinct prototypes and H/ln c;
  seeds 0-4. And the audit rule executed federatedly through the lossless PRE one-step exchange (unweighted, undamped),
  checking that it returns the same m* on every configuration.
- E-G Partial participation (exploratory, seeds 0-9): half of the clients sampled per round (seeded), for post_none_off,
  pre_none_off, pre_mass_off, post_footprint_on at L in {1,5}, and SC-FFCM with cfraction 0.5 at its calibrated steps.

## 3. Addendum (2026-09-24, written after the second external review and BEFORE any run on seeds 20-29)

Checked at the time of writing: no CSV under `results/t10/` contains a seed in 20-29. Additions (they do not change §1):
- Statistics for every table (all blocks): p-values of the paired Wilcoxon signed-rank test computed EXACTLY by
  enumerating the 2^n' sign assignments of the mid-ranks (zeros dropped as in Wilcoxon's rule; n' = effective n reported);
  the smallest attainable two-sided p with n' = 10 is 1/512. Headline contrasts also get the paired mean difference with a
  95% t-interval over seeds. Sensitivity analysis for §1: Holm over ALL primary tests of the fresh block (H1-H8 x
  configurations) in addition to the per-hypothesis Holm of §1. The earlier normal-approximation p-values are replaced
  wherever they differ; any count that changes is reported.
- E-H (exploratory diagnostic): the one-round tilt diagnostics of T9 R6 (weighted CV of r/q, displacement/spread) also
  along `pre_footprint_off` (L=5 and L=1), split into unclipped and clipped (client, prototype) pairs, seeds 0-4.
- E-I (theory check): numerical reproduction of a non-descent counter-example for the damped PRE rule with L >= 2 (two
  Gaussian blobs, one per client, from a fixed point of the pooled FCM objective), and a numerical check of the descent
  property of the damped ungated PRE rule at L = 1 (J_beta non-increasing along all 14 configurations, seeds 0-4).
- Decision analysis (uses held-out blocks 10-19 and 20-29 only): actions = {F-FCM (post_none_off, L=5), GF-PFedFCM
  (post_footprint_on, L=5), ungated PRE L=1, pre_mass_off L=1, pre_mass_off L=5, SC-FFCM L=5, SC-FFCM L=1 (R=50)}; criteria
  = mean client ACC, worst-client ACC, global-alignment ACC (gacc), FCM objective, uploaded numbers per run; states = the 14
  configurations; per criterion: mean regret (uniform prior over configurations) and maximum regret against the best
  action of each configuration; the recommended action per criterion under each rule.

## 4. Addendum 2 (2026-09-24 about 10:48 EDT, before the run plan started at 10:49 and before any run on seeds 20-29)

Clarification of H5's protocol: the earlier one-step GF-PFedFCM blocks that H5 replicates (`__T9_L1_gf.csv`,
`__T9_L1_confirm.csv`) were run for the full 50 rounds (FFCM_NO_EARLYSTOP=1), which is also what §1's "50 rounds"
states. H5 therefore compares pre_mass_off (L=1) with GF-PFedFCM at L=1 WITHOUT early stopping (primary); the same
contrast with T3's default early stop is reported as the secondary H5-ES. All other hypotheses unchanged. Family-wide Holm
over all H1-H8 x configurations (K = 112) cannot reject with ten seeds (smallest exact two-sided p = 1/512 > 0.05/112);
it is reported with that note, not as evidence of absence.

## 5. Erratum (2026-09-24, after addendum 2; does not change any test or hypothesis)

Section 3 says "the earlier normal-approximation p-values are replaced". That description of the earlier analyses is
inaccurate: the installed SciPy (1.17.1, used for T7-T9) computes `scipy.stats.wilcoxon` with its default method
exactly for n <= 13 also when ties or zeros occur (sign-flip enumeration with mid-ranks; checked 2026-09-24: p = 0.00390625
for nine nonzero differences, against 0.0077 asymptotic). The T10 test differs only in fixing the zero tolerance
(|d| <= 1e-12) and the tie rounding (12 decimals) instead of exact floating-point equality. Counts that change are
reported as stated in Section 3.
