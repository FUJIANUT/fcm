# T10 implementation notes (2026-09-24, revised after the code/protocol review, 10:45 EDT)

Implements every experiment of `experiments/T10_PREREGISTRATION.md` (unchanged) and, since this revision, its
statistics. Nothing in `fedfcmsim/`, `scripts/T3_design_space.py` or any pre-existing file under `results/` was
modified. No run touched seeds 20-29 (the smoke plan's "FRESH" steps run on seed 0 only).

## Files

| file | role |
|---|---|
| `scripts/T10_review.py` | all experiments and the analysis as subcommands: `cells`, `scffcm`, `scffcm-calibrate`, `cfcm`, `pedrycz-calibrate`, `pedrycz`, `audit`, `compare`, `argcheck`, `analyze` |
| `scripts/T10_selftest.py` | tests (i)-(xiv), 23 checks; output in `results/t10/SELFTEST.txt` |
| `scripts/T10_run_all.sh` | the complete run plan (`zsh T10_run_all.sh` from any directory; re-executes itself under `caffeinate -i`); `T10_SMOKE=1` runs a smoke-sized copy |
| `results/t10/<tag>.csv` + `<tag>.meta.json` | one CSV per command (meta: argv, parsed args, FFCM_* environment, host, wall time, sha256 of the code) |
| `results/t10/selftest/`, `smoke/`, `smoke_plan/` | scratch outputs of the self-test, the timing smoke runs and the smoke run of the plan (seeds 0-4, 10-11, 100 only) |
| `results/t10/theory/` | E-H / E-I, produced by `scripts/T10_theory*.py` (a separate job that finished at 10:09; now also step 11 of the plan) |

## Environment and T3

* T3 reads `FFCM_MSTAR/L/ROUNDS/SEED_START/M/NO_EARLYSTOP` at import, and `run_cell`'s `rounds`/`local_steps`
  defaults are bound then. `T10_review.py` never imports T3 at module level. Each subcommand writes
  `FFCM_MSTAR=1, FFCM_L, FFCM_ROUNDS, FFCM_SEED_START` into `os.environ` and deletes `FFCM_M` and
  `FFCM_NO_EARLYSTOP` (protocol = m*, T3's stopping rule; `--no-earlystop` sets it) BEFORE the first import. Pool
  workers use the `spawn` context with an initializer that applies the same environment and then imports T3.
* `get_T()` asserts after import that `T.USE_MSTAR`, `T.FORCE_M is None`, `T.NO_EARLYSTOP`, `T.LOCAL_STEPS`,
  `T.ROUNDS` **and the default arguments of `T.run_cell`** equal the requested values. `T.M` is set per dataset in
  every task (`T.M = T.fuzzifier_for(label)`).
* Mask gates are registered at worker start-up: `T.GATE_FNS['fpmask'] = where(gate_footprint > r_min, 1, r_min)`,
  `T.GATE_FNS['massmask'] = where(gate_mass > r_min, 1, r_min)`, `r_min = T.MIN_RELEVANCE`, used by `T.run_cell`
  unchanged. Cell parser: `<pre|post>_<none|footprint|mass|fpmask|massmask>_<off|on>`, T3's aliases, groups
  `all12`, `all16`, `masks`.
* Clients and init are built exactly as `T3.run_one_seed` / `T9b.setup` build them (partition 1201+s, init 1201+s+77).

## Metrics and columns (every run)

T3's own `min_dist(V)`, `client_accs` (mean_acc, worst_acc) and `fcm_objective_mean` (fcm_obj), plus `gacc` (one
Hungarian alignment over the pooled data, each point assigned by its own client's evaluation prototypes), `xb`
(Xie-Beni index of the shared prototypes on the pooled data; for personalized cells V is the global model) and
`hnorm` (mean normalized membership entropy under the evaluation prototypes). A non-finite model gives NaN for every
metric.

**Uniform descriptive columns (new).** Every method row now has `n_rounds` (rounds actually executed), `participation`
(= cfraction for SC-FFCM, 1.0 otherwise), `split` (0 except E-C), `n_part`, `n_clients`, `m`, `n_clusters` (c), `dim`
(d) and, for cells and SC-FFCM, `upload_numbers`. Placeholder rows for configurations with no stable calibrated
setting (`error=no_stable_setting`) carry `m = T3.fuzzifier_for(d)`, `n_rounds = -1`, `upload_numbers = -1`,
participation and split. cfcm has no L/rounds (not applicable); it has `n_iter` and `max_iter` instead.

* **`n_rounds` of the cells (review issues 1/10).** `T.run_cell` does not report its round count, and personalized
  cells early-stop at tol 1e-5. Every full-participation cell now runs through `run_cell_counted`: T3's `run_cell`,
  unchanged and with its defaults, while `T.GATE_FNS[gate]` is temporarily wrapped by a call counter. `run_cell`
  looks the gate up once per call and calls it exactly once per client per round (record_gates=False), so
  `n_rounds = calls / n_clients` (a non-integer quotient raises an error). The wrapper returns the gate's value
  untouched. Self-test (i): bit-for-bit identical to `T.run_cell`, and `n_rounds` equals `run_cell_pp`'s count on
  20 runs; personalized L=5 runs stop after 19-50 rounds. Self-test (ii-n): every cells row of the regression runs
  has `n_rounds` (personalized minimum 17 at L=5 and 14 at L=1; non-personalized always 50). The
  partial-participation path writes `run_cell_pp`'s count.
* **`upload_numbers`** (the decision-analysis criterion "uploaded numbers per run"). The convention is fixed here,
  before the fresh block. T3 cells: each participating client uploads, per round, its c x d local prototypes and the
  c aggregation weights beta_p * mass * gate (personalized offsets stay local), giving `n_rounds * n_part * c * (d+1)`.
  SC-FFCM: each client uploads Delta_V and Delta_c (fedfcmsim.federated.scffcm), giving `n_rounds * n_part * 2 c d`.
  Downlink is not counted.

## Subcommands - design decisions

* **cells** with `--participation 1 --split 0` is the T3 call (via `run_cell_counted`, see above).
  `--participation P<1` (E-G) uses `run_cell_pp`, a line-by-line copy of `run_cell` with client sampling: one
  generator `default_rng(1201 + seed + 555)` per run; each round draws `sort(rng.choice(P, max(1,int(P*frac)),
  replace=False))`; aggregation runs over the sampled clients with the same weights; non-sampled clients keep their
  offsets; T3's early-stop rule applies. Self-test (i): identical to `T3.run_cell` bit for bit at frac=1 (40 runs).
* **split** (E-C, `--split 0.2`): one generator `default_rng(1201 + seed + 999)`; per client, `perm =
  rng.permutation(n_p)`, `n_test = clip(floor(0.2 n_p + 0.5), 1, n_p - 1)`, and the test part is the first `n_test`
  entries of perm. The init comes from the pooled TRAIN data. In-sample metrics are on the 80% (no prefix). Held-out
  metrics are on the 20%, with each client's evaluation prototypes (review issue 2):
  - **primary**: `test_fcm_obj`; `test_gacc_trmap` (a label map fitted by one Hungarian alignment on the pooled
    TRAIN predictions, applied to the pooled test points); `test_mean_acc_trmap` / `test_worst_acc_trmap` (each
    client's label map fitted on its own TRAIN part and applied to its test points; a cluster missing from the map
    counts as an error). No test-point label is used to fit any map. Self-test (xi): on the points it was fitted on,
    the map reproduces `clustering_accuracy` (300 random cases).
  - reported: `test_gacc` (one Hungarian on the pooled test part), `test_hnorm`;
  - **diagnostic only**: `test_mean_acc_refit` / `test_worst_acc_refit`, which re-fit a Hungarian map on each
    client's few test points. This is biased upward on small test sets. **A client with one test point always scores
    1.0** (`clustering_accuracy([3],[0]) = 1.0`; the earlier note "their test ACC is 0/1" was wrong), and three
    points with two classes score at least 2/3. Measured: wine has 3-4 test points on every client, and
    quantity_skew_extreme has 4 on some clients. On quantity_skew_extreme s0, the `post_footprint_off` test worst ACC
    is 0.612 with the re-fitted map and 0.200 with the train map. Each row carries `min_client_n_test` and
    `n_clients_test_le5`. The analysis flags test ACC as **unreliable on wine and quantity_skew_extreme** (any client
    with <= 5 test points), and the write-up must say so.
* **scffcm**: `fedfcmsim.federated.scffcm(..., rounds=R, local_steps=L, m=T.M, eta_l, eta_g, cfraction,
  seed=1201+seed+555, track_objective=False)`. Steps come from a calibration CSV (matched on dataset and L; a CSV
  without a `rounds` column is taken as R=50, as t9b/t9c are; a mismatch with `--rounds` is an error) or from
  `--eta-l/--eta-g`. With cfraction 1 the seed is irrelevant (t9b is reproduced bit for bit, self-test iii). E-G
  uses the t9b steps with cfraction 0.5.
* **scffcm-calibrate / pedrycz-calibrate**: T9b's rule verbatim (seed 100; a setting is stable iff finite and
  `J <= 5 Jc`; choose min `|J-Jc|/Jc`, ties to the first setting in grid order). Grids: `t9b` (reproduces
  `t9b_scffcm_calibration.csv`, self-test vii), `ed` (9 x 7), `ee` (alpha, 10 values). A `--cal-seed` in 20-29 is
  refused without `--allow-fresh` (review issues 6/19).
* **cfcm**: `fcm(pooled, k, init_centers=init, m=T.M, max_iter=150)`, which is T6's centralized_fcm (self-test iv).
* **pedrycz**: `pedrycz_gradient_fcm(..., local_steps=L, m=T.M, alpha)`; the fuzzifier is used throughout (ix).
* **audit** (E-F) now covers **all 14 configurations** (review issue 16; the 6 synthetic ones are cheap) x m in
  {2,1.5,1.3,1.2,1.1} x seeds 0-4. Inits: (a) paper, (b) k-means++ best of 10, (c) oracle class means, (d) the
  unweighted, undamped PRE one-step exchange (300 iterations), with its iteration-wise gap to centralized FCM.
  `<tag>_mstar.csv` applies the label-free rule per init and compares the result with `T3.MSTAR` (2.0 for the
  synthetic configurations). (a) reproduces degeneracy_data.csv exactly on the 8 real configurations (self-test v).
* **compare** (review issues 4/12/13) tests float equality (NaN==NaN) against every prior CSV that holds the same
  key. Rows are either **strict** or **advisory**. Advisory rows are the personalized rows of files produced WITHOUT
  early stopping (`T9_L1_gf`, `T9_L1_confirm` at L=1 and, new, `T9_noES`, `T9_noES_confirm` at L=5), compared with
  new rows that used T3's early stop. These are compared on ACC only, reported, and never fatal, because they follow
  a different stopping protocol. All other rows are strict. `compare` exits 1 on:
  - (a) any strict mismatch;
  - (b) zero strict overlap;
  - (c) any in-scope prior row that the new CSV does not cover. The scope is the prior rows for the new CSV's seeds
    and cells and for the datasets the command was ASKED for (read from its meta.json), so a dataset-label drift or a
    missing task fails;
  - (d) fewer strict overlapping runs than `--min-runs`.

  The plan passes the exact expected counts, computed from the prior files: cells L5 s0-9 2240 (main 960 + mstar
  720 + T6 560), L5 s10-19 840, L1 s0-9 700, L1 s10-19 560, GF L1 noES s0-19 280, SC-FFCM 280 / 280 / 240, cfcm 140,
  audit 200. For cfcm, T6 has seeds 0-9 only, so the reviewer's 280 cannot be reached. The reviewer's cells figures
  count (dataset, seed) pairs; ours count (source, dataset, seed, cell) runs. Self-test (xiii): label drift, zero
  overlap and a too-high `--min-runs` each exit 1.
* **argcheck** (review issue 5): `T10_run_all.sh` skips a step only if `<tag>.meta.json` records the same parsed
  arguments (ignoring `--workers` / `--force`; paths resolved); otherwise the plan stops. A change in the code hash
  is printed as a note and is not fatal.
* Outputs are written atomically (tmp + rename) at the end of a command. An existing output is overwritten only
  with `--force`, and `--outdir` must be `results/t10` or below. Any seed in 20-29 requires `--allow-fresh`.

## analyze: the pre-registered statistics (review issue 9)

`T10_review.py analyze` implements §1 and §3 and is **frozen before the fresh block**. The plan runs it on the seeds
0-19 CSVs (`analysis_s00-19`), writes the sha256 of `T10_review.py` into `ANALYSIS_FROZEN`, and refuses to run the
final analysis (`analysis_final`, blocks s00-09, s10-19, s20-29) if the file changed afterwards.

* **Test**: two-sided paired Wilcoxon signed-rank, computed exactly.
  - Differences with `|d| <= 1e-12` count as zeros and are dropped (Wilcoxon's rule); n' is reported as `n_eff`.
  - Mid-ranks of |d| are used, with |d| rounded to 12 decimals so that float noise does not break exact ties.
  - `p = P(|W+ - E| >= |w+ - E|)` over the 2^n' equally likely sign assignments. These are counted by the
    convolution recursion over the doubled ranks, which gives the same count as explicit enumeration. Self-test
    (xii-a) checks both on 200 random vectors with ties and zeros, and against scipy's exact test on 103 tie-free
    vectors.
  - Level 0.05. The direction of a "better/worse" count is the sign of the paired mean difference (as in T7-T9).
  - Each test row also carries the paired mean difference with its 95 % t-interval, n, n_nan, n' and W+.
* **Holm** is applied per hypothesis over its configuration family (14; 12 when SC-FFCM L=50 is involved), and counts
  are reported unadjusted and Holm-adjusted. `replicates` means the Holm count in the stated direction is >= 1 and
  the opposite Holm count is <= it. **H9** ("no improvement", reported as counts) has no replication rule in §1. The
  analysis reports H9 as consistent iff the Holm-adjusted count of "personalization better" is 0; this is an
  interpretation and is labelled as such.
* **Sensitivity (§3)**: Holm over all H1-H8 x configurations of the block (K = 112). **A design property found while
  implementing it: with n' = 10, the smallest attainable exact two-sided p is 1/512 = 0.00195, but the first Holm
  step at K = 112 needs p <= 0.05/112 = 0.00045. This sensitivity analysis therefore cannot reject any test, and its
  counts are 0/0 by construction.** The analysis prints this note, and the paper should state it (it is a property
  of the pre-registered design, not a result). Per-hypothesis Holm (K = 14, first step 0.00357) is attainable.
* **Diverged runs (NaN) in a paired test.** The pre-registration does not specify this; it is fixed here, before the
  fresh run. Primary: the seed is dropped from that configuration's test (n and n_nan are reported), as the T9
  analyses did. Sensitivity (`p_holm_nanloss`, reported): a diverged run counts as a loss for the diverged method,
  with the largest |d| rank. A configuration with no data, or with only zero differences, stays in the Holm family
  with p = 1.
* **Re-analysis of the earlier blocks**: every table is computed for s00-09 and s10-19 with both the exact test AND
  the earlier blocks' test (`scipy.stats.wilcoxon` default, T7-T9). `<tag>_count_changes.csv` lists every count that
  changes. Self-test (xii-b) runs the analysis on the PRIOR CSVs. The legacy column reproduces all 14 documented
  counts of T8 E7 and T9 R2b/R2c/R3 (e.g. held-out H1 8/1, H2 11/1, H3 9/14, H4 12/1, H5 4/0, H6 6/2, H7 6/2,
  H8 0/6, L=50 4/2). The exact test gives the same unadjusted mean_acc counts on 14/14; one secondary fcm_obj count
  changes (5 -> 4).
* **Secondary** (reported, not used for "replicates"): every pair on fcm_obj (lower is better), worst_acc and gacc;
  pre_mass_off (L=1) vs SC-FFCM (L=50) on the 12 non-MNIST configurations; pre_none_off (L=1) vs SC-FFCM (L=1 and
  L=5); **H5-noES** (see Deviations, item 1); the E-D pairs at R=250.
* **E-A**: the interaction (pre_none - post_none) - (pre_G - post_G) for G in {footprint, mass, fpmask, massmask} at
  L=5, per configuration and block, with the exact test and Holm per G over 14 configurations, on mean_acc and
  fcm_obj; s20-29 is secondary.
* **E-C** (exploratory, seeds 0-19, n = 20): post_footprint_on vs off and pre_mass_on vs off, on the held-out
  primary metrics above and on in-sample fcm_obj / mean_acc / gacc. `test_acc_unreliable` is flagged where a client
  has <= 5 test points.
* **Decision analysis** (§3, seeds 10-29): 7 actions x 5 criteria (mean_acc, worst_acc, gacc, fcm_obj,
  upload_numbers) x 14 configurations.
  - A (configuration, seed) enters only if every action has a row with that criterion.
  - A diverged run (or one with no stable setting) scores the worst value any action obtained on that
    (configuration, seed); these substitutions are counted.
  - State value = mean over seeds.
  - Regret = best - value for the ACC-type criteria, and (value - best) / |best| for fcm_obj and uploaded numbers,
    whose scale differs by orders of magnitude across configurations (an interpretation fixed here).
  - Reported: mean regret (uniform prior over configurations) and maximum regret per action, and the recommended
    action per criterion under each rule.
  - Self-test (xiv) runs it on T10-format CSVs: all five criteria are populated.
* Outputs: `<tag>.txt` (report), `_tests.csv`, `_summary.csv`, `_count_changes.csv`, `_ea.csv`, `_ec.csv`,
  `_decision.csv`, `_decision_values.csv`, `.meta.json`.

## Provenance of the prior CSVs (test ii)

* L=5: `per_seed_metrics__mstar.csv` (6 m* configs, seeds 0-9, 12 cells), `per_seed_metrics_main.csv` (the 8 configs
  with m* = 2, seeds 0-9), `per_seed_metrics__confirm.csv` (14 configs, seeds 10-19, 6 cells) and
  `t6_baselines/per_seed_metrics__mstar.csv` (4 T3 cells) are reproduced bit for bit with FFCM_MSTAR=1, L=5, R=50.
* **`per_seed_metrics__T9_noES.csv` (s0-9) and `__T9_noES_confirm.csv` (s10-19), L=5, were run with
  FFCM_NO_EARLYSTOP=1.** This is a new source; self-test (ii'): 42 runs, 168 values incl. min_dist/fcm_obj,
  0 mismatches with `--no-earlystop`. With T3's early stop, mean_acc was equal on all 48 checked personalized runs
  (max |d min_dist| 1.4e-5). These files add pre_footprint_on, pre_mass_on and post_mass_on to E-B's coverage as
  advisory acc-only rows.
* L=1: `per_seed_metrics__L1all.csv` (s0-9, 4 cells) is reproduced with the default environment.
  **`__T9_L1_gf.csv` and `__T9_L1_confirm.csv` were run with FFCM_NO_EARLYSTOP=1**: their personalized rows reproduce
  in every field only with it, and their non-personalized rows with the default environment. Early stop vs no early
  stop: mean_acc equal on 12/12 checked runs, max |d min_dist| 8.3e-6.
* SC-FFCM: `t9b_scffcm_fair.csv` / `t9c_scffcm_fair.csv` are reproduced bit for bit. T6 centralized FCM and
  degeneracy_data.csv are reproduced exactly.
* E-B coverage that cannot be verified, because no prior value exists: at L=5 s10-19, every cell other than the 6 of
  `confirm` and the 3 personalized noES cells (acc only); at L=1 s0-9, every cell other than L1all's 4 and
  T9_L1_gf's 2; at L=1 s10-19, every cell other than T9_L1_confirm's 5; the four mask cells everywhere. The paper's
  E-B description should list this.

## Deviations from / interpretations of the pre-registration

1. **Stopping rule of GF at L=1 (H5; review issues 3/11). The user should confirm this decision before launching.**
   §1 fixes the protocol as "identical to the paper (T3 ...)", i.e. T3's default, under which personalized cells
   early-stop. The earlier L=1 GF blocks that H5 replicates (T9_L1_gf, T9_L1_confirm) were run with
   FFCM_NO_EARLYSTOP=1. The plan now runs BOTH on seeds 20-29, with the primary declared before any fresh run:
   - **primary H5 = T3 default** (`FRESH_cells16_L1_s20-29`), as §1 specifies;
   - **secondary H5-noES = no early stop** (`FRESH_gf_L1_noES_s20-29`), like-for-like with the earlier blocks
     (~3 min).

   The seeds 0-19 arm of H5-noES (`gf_L1_noES_s00-19`) is now the plan's first step. It is compared with T9_L1_gf /
   T9_L1_confirm in ALL fields (strict, 280 runs), so a provenance problem halts the plan within minutes rather than
   hours. The acc-only comparisons across stopping rules are advisory and can no longer halt the plan. To make noES
   the primary instead, swap the `H5` / `H5-noES` arms between `H_PRIMARY` and `H_SECONDARY` BEFORE the analysis is
   frozen.
2. E-B asks that the re-runs "reproduce the existing mean_acc" of the 12 cells at L in {1,5} on seeds 0-19. Only
   existing values can be reproduced (see the coverage list above).
3. E-D's "seeds 10-29" is run as two commands (s10-19 in section 5, s20-29 in the fresh-block section).
4. E-D's default-step SC-FFCM at L=50 now includes the two MNIST configurations (review issues 7/15). No calibration
   is needed at the default steps; measured 102 s / 162 s per run for 20c / 50c on seed 0, about +8 min wall. The
   CALIBRATED L=50 runs stay on the 12 non-MNIST configurations, because t9c exists only for those (§1: "12 for
   L = 50").
5. Self-test (v) runs the audit on seeds 0-4 of wine and digits_pca16 (E-F's own seeds).
6. The split test-set size rounds half up (`floor(0.2 n + 0.5)`, with >= 1 test point and >= 1 training point).
7. **Centralized FCM on seeds 20-29 is not pre-registered** (review issue 14). It is kept only as a reference line,
   renamed `FRESH_cfcm_s20-29_descriptive`, and enters no test. The paper must label it descriptive/unregistered.
8. Interpretations fixed in code before the fresh run (all described above): NaN handling in paired tests, the H9
   verdict, relative regret for fcm_obj / uploaded numbers, the uploaded-numbers convention, the direction of a
   count (paired mean difference), and the zero/tie tolerances.

## Smoke / regression runs (seeds 0-1, 10-11; audit 0-4; calibration seed 100)

* Self-test: **23/23 pass** (`SELFTEST.txt`) in 224 s, with the machine shared with other processes.
  `selftest/SELFTEST_PASSED` holds the code hash of this pass, so the plan skips the self-test unless the code
  changes.
* Earlier: all 14 configs x 16 cells x seeds 0-1 at L=5 and L=1, with compare: 0 mismatches.
* `T10_SMOKE=1 zsh T10_run_all.sh` with the revised plan: every step, every `compare`, both analyses and the freeze
  pass on wine / seed 0 in 1 min 37 s (`smoke_plan/`). A second invocation skipped every step via argcheck.
* Timing of E-D default-step SC-FFCM, L=50, MNIST, seed 0: `smoke/timing_scffcm_default_L50_mnist_s0.csv`.
* `results/t10/` holds no top-level CSV, so no smoke output collides with a tag of the full run.

## Runtime

Measured smoke wall times (W=12, 16-core machine; per-task times are recorded in each CSV's `runtime_s`):

| smoke run (seeds 0-1, all 14 configs unless noted) | wall | slowest task |
|---|---|---|
| cells, 16 cells, L=5 | 1658 s | mnist (50c) 1650 s/seed, mnist (20c) 1145 s, letter 870 s |
| cells, 16 cells, L=1 | 685 s | mnist (50c) ~680 s |
| cells, split 0.2, 4 cells, L=5 | 480 s | |
| cells, participation 0.5, 4 cells, L=5 | 405 s | |
| scffcm L=5 (t9b steps) | 37 s | |
| scffcm L=1, R=250 | 40 s | mnist ~35 s |
| scffcm L=50 default steps, 12 non-MNIST | 114 s | |
| scffcm L=50 default steps, MNIST 20c / 50c, seed 0 | 165 s | 102 s / 162 s per run |
| pedrycz L=10 | 197 s | mnist (50c) ~275 s (single) |
| audit, one (m, seed) task | | mnist (50c) m=1.1: 994 s; letter m=1.3: 561 s |

Projection (longest-processing-time schedule of the measured per-task times on 12 workers, idle machine):

| plan section | projected wall |
|---|---|
| 0 self-test (re-run only if the code hash changed; the sentinel matches the current code) | 0-8 min |
| 0b GF L=1 no-early-stop s0-19 (+ strict compare) | 5 min |
| 1 calibrations (E-D, E-E x3) | 25 min |
| 2 cells16 L=5 s0-9, s10-19; L=1 s0-9, s10-19 (+compare) | 3 h 15 min |
| 3 E-C split s0-19 | 36 min |
| 4 SC-FFCM calibrated L=1/5/50 s0-19, cfcm s0-19 | 15 min |
| 5 E-D defaults L=1/5/50 (L=50 now incl. MNIST), R=250 SC-FFCM and cells s10-19 | 33 min |
| 6 E-E Pedrycz L=1/5/10 s0-19 | 25 min |
| 7 E-F audit (14 configs) | 1 h 45 min (upper bound) |
| 8 E-G | 21 min |
| 8b analysis s0-19 + freeze | < 1 min |
| 9 FRESH block s20-29 (incl. GF L=1 no-early-stop, descriptive cfcm) | 1 h 55 min |
| 10 final analysis; 11 theory (already complete: cache hit) | < 2 min |
| **total** | **about 9 h 15 min (allow 8.5-13 h), about 115 CPU-hours** |

Caveats (review issue 20): the projection assumes an idle machine. The M4 Max has 12 performance + 4 efficiency
cores, and W = 12 fills the P cores. The theory job has finished (10:09), but other processes were active at 10:35
(load average ~8-11, including a local LLM server and an unrelated `run_b3.py` job). Start the plan when the machine
is otherwise idle, or set `T10_WORKERS=8` (then ~12-13 h). The plan re-executes itself under `caffeinate -i` so the
Mac does not sleep, and it prints the load average at start.

## How to run

```
zsh experiments/scripts/T10_run_all.sh          # full plan; resumable (finished steps are skipped if same arguments)
T10_WORKERS=8 T10_PYTHON=python3 zsh ...        # overrides
T10_SMOKE=1 zsh experiments/scripts/T10_run_all.sh   # ~2-min smoke copy into results/t10/smoke_plan/
python3 experiments/scripts/T10_selftest.py     # self-test only
python3 experiments/scripts/T10_review.py analyze --source prior --tag X --outdir results/t10/<scratch>
                                                # the analysis on the earlier blocks' CSVs
```
