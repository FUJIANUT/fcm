#!/usr/bin/env zsh
# T10 run plan: every command needed to execute experiments/T10_PREREGISTRATION.md, in order.
#
#   zsh T10_run_all.sh            (from any directory; one command at a time, W workers each; re-executes itself under
#                                  `caffeinate -i` so the Mac does not sleep during the ~9 h run)
#
# * every step writes its own results/t10/<tag>.csv (+ <tag>.meta.json); its log goes to results/t10/logs/<tag>.log
# * a step whose CSV already exists is SKIPPED only if its meta.json records the SAME command (T10_review.py argcheck);
#   otherwise the plan stops. CSVs are written atomically at the end of a step, so an existing CSV is a complete one.
#   The plan is resumable after a failure and never re-runs a finished step -- in particular no fresh-block (seeds
#   20-29) step is ever run twice
# * the self-test is re-run whenever the code changed since it last passed (sha256 of T10_review.py, T10_selftest.py,
#   T3_design_space.py and fedfcmsim/*.py is stored in selftest/SELFTEST_PASSED)
# * the plan stops at the first failing step, including a failing 'compare' against the prior CSVs: a strict
#   mismatch, zero overlap, an uncovered in-scope prior row, or fewer overlapping runs than expected (--min-runs).
#   acc-only comparisons across stopping rules (prior no-early-stop personalized rows vs new T3 early stop) are
#   ADVISORY: reported in the log and the cmp CSV, never fatal
# * the analysis code (T10_review.py analyze) is run on seeds 0-19 and then FROZEN (sha256 in ANALYSIS_FROZEN) before the
#   fresh block; the final analysis refuses to run if T10_review.py changed after the freeze
# * calibrations first, then exploratory/regression evaluations on seeds 0-19, and the pre-registered fresh block
#   (seeds 20-29, --allow-fresh) LAST, so that a bug found anywhere stops the plan before the fresh block is touched
# * results/t10/RUN_DONE is written at the very end (it also requires the E-H/E-I theory job's SUMMARY.md)
#
# Expected runtimes (wall, W=12 on the 16-core M4 Max = 12 P + 4 E cores, IDLE machine) are projected from the smoke
# runs documented in results/t10/IMPLEMENTATION_NOTES.md (section "Runtime"); "~" values are +-50%.
# Projected total: about 9.5 h wall (allow 8-13 h), ~115 CPU-hours. Start it only when no other heavy job runs
# (or lower T10_WORKERS): the plan prints the load average at start.
#
# Smoke mode (checks the plan's mechanics in ~10 min without touching results/t10/ proper or seeds >= 1):
#   T10_SMOKE=1 zsh T10_run_all.sh   -> every step runs on wine, seed 0 only (calibrations on wine, seed 100),
#   outputs in results/t10/smoke_plan/; the self-test, the theory job and the --min-runs coverage counts are skipped.

set -e
set -o pipefail
setopt NO_NOMATCH

if [[ -z ${T10_CAFFEINATED:-} ]] && command -v caffeinate >/dev/null 2>&1; then
  export T10_CAFFEINATED=1
  exec caffeinate -i zsh "${0:A}" "$@"
fi

SCRIPT_DIR=${0:A:h}
EXP=${SCRIPT_DIR:h}
RES=$EXP/results
OUT=$RES/t10
SMOKE=${T10_SMOKE:-0}
[[ $SMOKE == 1 ]] && OUT=$RES/t10/smoke_plan
LOG=$OUT/logs
PY=${T10_PYTHON:-python3}
W=${T10_WORKERS:-12}
T10=$SCRIPT_DIR/T10_review.py
T9B=$RES/t9b_scffcm_calibration.csv     # SC-FFCM steps, L in {1,5}, calibrated on seed 100 (NOT re-calibrated)
T9C=$RES/t9c_scffcm_calibration.csv     # SC-FFCM steps, L = 50 (12 non-MNIST configs)
mkdir -p $LOG $OUT/selftest

code_hash() {  # sha256 of every file whose change must invalidate the self-test
  (cd $EXP && shasum -a 256 scripts/T10_review.py scripts/T10_selftest.py scripts/T3_design_space.py fedfcmsim/*.py)
}

step() {   # step <tag> <subcommand> [args...]
  local tag=$1; shift
  local -a extra
  extra=(--outdir $OUT)
  if [[ $SMOKE == 1 ]]; then                      # last occurrence of an option wins in argparse
    extra+=(--datasets wine)
    [[ $1 == *-calibrate ]] || extra+=(--seed-start 0 --seeds 1)
  fi
  if [[ -f $OUT/delegated/$tag ]]; then           # run on another machine (see $OUT/delegated/README); awaited before 8b
    print "[$(date '+%F %T')] deleg $tag (delegated; awaited before the analysis)"; return 0
  fi
  if [[ -f $OUT/$tag.csv ]]; then
    if ! $PY $T10 argcheck --meta $OUT/$tag.meta.json -- "$@" --tag $tag --workers $W $extra > $LOG/$tag.argcheck.log 2>&1; then
      print "[$(date '+%F %T')] FAILED $tag: $OUT/$tag.csv exists but was produced by a different command -- see $LOG/$tag.argcheck.log"
      exit 1
    fi
    print "[$(date '+%F %T')] skip  $tag (exists, same arguments)"; return 0
  fi
  print "[$(date '+%F %T')] start $tag"
  local t0=$SECONDS
  if ! $PY $T10 "$@" --tag $tag --workers $W $extra > $LOG/$tag.log 2>&1; then
    print "[$(date '+%F %T')] FAILED $tag -- see $LOG/$tag.log"; exit 1
  fi
  print "[$(date '+%F %T')] done  $tag ($(( SECONDS - t0 ))s)"
}

check() {  # check <kind> <tag> <expected strict overlapping runs>: comparison of results/t10/<tag>.csv with the priors
  local kind=$1 tag=$2 want=$3
  local out=cmp_$tag
  local -a minr
  [[ $SMOKE == 1 ]] || minr=(--min-runs $want)
  if [[ -f $OUT/delegated/$tag ]]; then print "[$(date '+%F %T')] deleg $out (checked on the delegate machine)"; return 0; fi
  if [[ -f $OUT/$out.csv ]]; then print "[$(date '+%F %T')] skip  $out"; return 0; fi
  if ! $PY $T10 compare --kind $kind --csv $OUT/$tag.csv --tag $out --outdir $OUT $minr > $LOG/$out.log 2>&1; then
    rm -f $OUT/$out.csv
    print "[$(date '+%F %T')] FAILED $out (mismatch / missing coverage vs prior results) -- see $LOG/$out.log"; exit 1
  fi
  print "[$(date '+%F %T')] ok    $out: $(tail -1 $LOG/$out.log)"
}

analyze() {  # analyze <tag> [extra args]: the frozen pre-registered analysis
  local tag=$1; shift
  if [[ -f $OUT/$tag.txt ]]; then print "[$(date '+%F %T')] skip  $tag"; return 0; fi
  if ! $PY $T10 analyze --tag $tag --t10-dir $OUT --outdir $OUT "$@" > $LOG/$tag.log 2>&1; then
    print "[$(date '+%F %T')] FAILED $tag -- see $LOG/$tag.log"; exit 1
  fi
  print "[$(date '+%F %T')] done  $tag"
}

print "T10 run plan: $EXP  (python: $PY, workers: $W, smoke: $SMOKE)"
print "load average at start: $(sysctl -n vm.loadavg 2>/dev/null)  (idle machine assumed by the runtime projection)"

# =====================================================================================================================
# 0. Self-test (seeds 0-1, 10-11, 0-4 for the audit check, 100); re-run whenever the code hash changed    ~5-10 min
# =====================================================================================================================
if [[ $SMOKE != 1 ]]; then
  if [[ ! -f $OUT/selftest/SELFTEST_PASSED ]] || [[ "$(code_hash)" != "$(cat $OUT/selftest/SELFTEST_PASSED)" ]]; then
    print "[$(date '+%F %T')] start selftest (no sentinel or the code changed since it last passed)"
    rm -f $OUT/selftest/SELFTEST_PASSED
    if ! $PY $SCRIPT_DIR/T10_selftest.py > $LOG/selftest.log 2>&1; then
      print "FAILED selftest -- see $LOG/selftest.log"; exit 1
    fi
    code_hash > $OUT/selftest/SELFTEST_PASSED
    print "[$(date '+%F %T')] done  selftest"
  else
    print "[$(date '+%F %T')] skip  selftest (passed on the current code)"
  fi
fi

# =====================================================================================================================
# 0b. Like-for-like regression of the earlier L=1 GF blocks (T9_L1_gf s0-9, T9_L1_confirm s10-19 were run WITHOUT
#     early stopping): post_footprint_on, L=1, --no-earlystop, seeds 0-19, compared in ALL fields (strict). Cheap
#     (~5 min); run first so that a provenance problem stops the plan early. Also the seeds 0-19 arm of H5-noES.
# =====================================================================================================================
step gf_L1_noES_s00-19            cells --L 1 --cells post_footprint_on --no-earlystop --seed-start 0 --seeds 20   # ~5 min
check cells gf_L1_noES_s00-19 280

# =====================================================================================================================
# 1. Calibrations (seed 100, label-free rule of T9b: finite and J <= 5 Jc, min |J-Jc|/Jc)
# =====================================================================================================================
# E-D: SC-FFCM at L=1, R=250 on the extended grid eta_l {0.05..16} x eta_g {0.5..32} (63 settings), all 14 configs
step ed_scffcm_cal_L1_R250        scffcm-calibrate --L 1 --rounds 250 --grid ed   # ~10 min
# E-E: Pedrycz gradient FCM, alpha grid {0.0005..0.5} (10 values), R=50, L in {1,5,10}, all 14 configs
step ee_pedrycz_cal_L1            pedrycz-calibrate --L 1  --grid ee   # ~2 min
step ee_pedrycz_cal_L5            pedrycz-calibrate --L 5  --grid ee   # ~4 min
step ee_pedrycz_cal_L10           pedrycz-calibrate --L 10 --grid ee   # ~7 min

# =====================================================================================================================
# 2. E-A / E-B: the 16 cells (12 of T3 + pre/post x fpmask/massmask, personalization off), m*, R=50, seeds 0-19;
#    each block is compared bit for bit with the prior CSVs (mean_acc, worst_acc, min_dist, fcm_obj). Expected strict
#    overlapping runs (every prior row of those seeds): L5 s0-9 2240 (main 960 + mstar 720 + T6 560), L5 s10-19 840,
#    L1 s0-9 700 (L1all 560 + T9_L1_gf gate-only 140), L1 s10-19 560. Advisory (acc only, not fatal): the
#    personalized rows of T9_noES / T9_noES_confirm (L5) and T9_L1_gf / T9_L1_confirm (L1), run without early stop.
# =====================================================================================================================
step cells16_L5_s00-09            cells --L 5 --cells all16 --seed-start 0  --seeds 10   # ~1 h
check cells cells16_L5_s00-09 2240
step cells16_L5_s10-19            cells --L 5 --cells all16 --seed-start 10 --seeds 10   # ~1 h
check cells cells16_L5_s10-19 840
step cells16_L1_s00-09            cells --L 1 --cells all16 --seed-start 0  --seeds 10   # ~27 min
check cells cells16_L1_s00-09 700
step cells16_L1_s10-19            cells --L 1 --cells all16 --seed-start 10 --seeds 10   # ~27 min
check cells cells16_L1_s10-19 560

# =====================================================================================================================
# 3. E-C: split-sample personalization check, 80/20 per client, L=5, seeds 0-19
# =====================================================================================================================
step ec_split_L5_s00-19           cells --L 5 --split 0.2 --cells post_footprint_off,post_footprint_on,pre_mass_off,pre_mass_on --seed-start 0 --seeds 20   # ~36 min

# =====================================================================================================================
# 4. SC-FFCM at the calibrated steps (t9b: L in {1,5}; t9c: L=50, non-MNIST), R=50, seeds 0-19 (+ compare with
#    t9b/t9c_scffcm_fair.csv) and centralized FCM (T6 protocol) seeds 0-19 (+ compare with T6, which has seeds 0-9)
# =====================================================================================================================
step scffcm_cal_L1_s00-19         scffcm --L 1  --steps-csv $T9B --seed-start 0 --seeds 20   # ~2 min
check scffcm scffcm_cal_L1_s00-19 280
step scffcm_cal_L5_s00-19         scffcm --L 5  --steps-csv $T9B --seed-start 0 --seeds 20   # ~3 min
check scffcm scffcm_cal_L5_s00-19 280
step scffcm_cal_L50_s00-19        scffcm --L 50 --steps-csv $T9C --datasets nonmnist --seed-start 0 --seeds 20   # ~5 min
check scffcm scffcm_cal_L50_s00-19 240
step cfcm_s00-19                  cfcm --seed-start 0 --seeds 20   # ~4 min
check cfcm cfcm_s00-19 140

# =====================================================================================================================
# 5. E-D: SC-FFCM released default steps (eta_l 0.2, eta_g 0.5) at L in {1,5,50}, R=50, seeds 0-19, ALL 14 configs
#    (no calibration is needed at the default steps; MNIST at L=50 measured ~100-160 s per run);
#    SC-FFCM (E-D calibration) and pre_mass_off / pre_none_off at L=1, R=250, seeds 10-19
# =====================================================================================================================
step ed_scffcm_default_L1_s00-19  scffcm --L 1  --eta-l 0.2 --eta-g 0.5 --seed-start 0 --seeds 20   # ~2 min
step ed_scffcm_default_L5_s00-19  scffcm --L 5  --eta-l 0.2 --eta-g 0.5 --seed-start 0 --seeds 20   # ~3 min
step ed_scffcm_default_L50_s00-19 scffcm --L 50 --eta-l 0.2 --eta-g 0.5 --seed-start 0 --seeds 20   # ~12 min
step ed_scffcm_L1_R250_s10-19     scffcm --L 1 --rounds 250 --steps-csv $OUT/ed_scffcm_cal_L1_R250.csv --seed-start 10 --seeds 10   # ~2 min
step ed_cells_L1_R250_s10-19      cells  --L 1 --rounds 250 --cells pre_mass_off,pre_none_off --seed-start 10 --seeds 10   # ~12 min

# =====================================================================================================================
# 6. E-E: Pedrycz gradient federated FCM at the calibrated alpha, L in {1,5,10}, R=50, seeds 0-19
# =====================================================================================================================
step ee_pedrycz_L1_s00-19         pedrycz --L 1  --alpha-csv $OUT/ee_pedrycz_cal_L1.csv  --seed-start 0 --seeds 20   # ~3 min
step ee_pedrycz_L5_s00-19         pedrycz --L 5  --alpha-csv $OUT/ee_pedrycz_cal_L5.csv  --seed-start 0 --seeds 20   # ~8 min
step ee_pedrycz_L10_s00-19        pedrycz --L 10 --alpha-csv $OUT/ee_pedrycz_cal_L10.csv --seed-start 0 --seeds 20   # ~14 min

# =====================================================================================================================
# 7. E-F: m-grid audit, ALL 14 configs x m in {2,1.5,1.3,1.2,1.1} x seeds 0-4; inits paper / k-means++ best of 10 /
#    oracle + federated lossless PRE one-step (300 iterations); init (a) compared with paper2 degeneracy_data.csv
#    (8 real configs x 5 m x 5 seeds = 200 runs)
# =====================================================================================================================
step ef_audit_s00-04              audit --seed-start 0 --seeds 5   # ~1 h 45 min (upper bound)
check audit ef_audit_s00-04 200

# =====================================================================================================================
# 8. E-G: partial participation (half of the clients per round), seeds 0-9
# =====================================================================================================================
step eg_cells_P05_L5_s00-09       cells --L 5 --participation 0.5 --cells post_none_off,pre_none_off,pre_mass_off,post_footprint_on --seed-start 0 --seeds 10   # ~14 min
step eg_cells_P05_L1_s00-09       cells --L 1 --participation 0.5 --cells post_none_off,pre_none_off,pre_mass_off,post_footprint_on --seed-start 0 --seeds 10   # ~5 min
step eg_scffcm_cf05_L1_s00-09     scffcm --L 1 --steps-csv $T9B --cfraction 0.5 --seed-start 0 --seeds 10   # ~1 min
step eg_scffcm_cf05_L5_s00-09     scffcm --L 5 --steps-csv $T9B --cfraction 0.5 --seed-start 0 --seeds 10   # ~2 min

# =====================================================================================================================
# 8b. The pre-registered analysis, run on the seeds 0-19 CSVs of this plan (re-analysis of the earlier blocks with the
#     exact test; also the last test of the analysis code), then FROZEN before the fresh block is touched.
# =====================================================================================================================
for f in $OUT/delegated/*(N); do            # delegated steps must have delivered their CSV (and its cmp_ check, if any)
  t=${f:t}; [[ $t == README ]] && continue
  while [[ ! -f $OUT/$t.csv ]]; do sleep 60; done
  print "[$(date '+%F %T')] recv  $t (delegated output present)"
done
analyze analysis_s00-19 --blocks s00-09,s10-19
if [[ -f $OUT/ANALYSIS_FROZEN ]]; then
  if [[ "$(cd $EXP && shasum -a 256 scripts/T10_review.py)" != "$(cat $OUT/ANALYSIS_FROZEN)" ]]; then
    print "FAILED: scripts/T10_review.py changed after the analysis was frozen ($OUT/ANALYSIS_FROZEN)"; exit 1
  fi
else
  (cd $EXP && shasum -a 256 scripts/T10_review.py) > $OUT/ANALYSIS_FROZEN
  print "[$(date '+%F %T')] analysis frozen: $(cat $OUT/ANALYSIS_FROZEN)"
fi

# =====================================================================================================================
# 9. FRESH CONFIRMATORY BLOCK, seeds 20-29 (pre-registered section 1; run once, never re-run: steps are skipped when
#    their CSV exists). SC-FFCM uses the steps ALREADY calibrated on seed 100 (t9b / t9c) -- no re-calibration.
#    Stopping rule: T3's default (personalized cells early-stop at tol 1e-5), as §1 specifies ("identical to the paper").
#    FRESH_gf_L1_noES is the PRIMARY arm of H5 (GF at L=1 without early stop = the protocol of the earlier L=1 GF
#    blocks; pre-registration addendum 2); H5-ES (secondary) uses FRESH_cells16_L1 (T3 default). Decided before any fresh run.
# =====================================================================================================================
step FRESH_cells16_L5_s20-29        cells --L 5 --cells all16 --seed-start 20 --seeds 10 --allow-fresh   # ~1 h
step FRESH_cells16_L1_s20-29        cells --L 1 --cells all16 --seed-start 20 --seeds 10 --allow-fresh   # ~27 min
step FRESH_gf_L1_noES_s20-29        cells --L 1 --cells post_footprint_on --no-earlystop --seed-start 20 --seeds 10 --allow-fresh   # ~3 min
step FRESH_scffcm_cal_L1_s20-29     scffcm --L 1  --steps-csv $T9B --seed-start 20 --seeds 10 --allow-fresh   # ~1 min
step FRESH_scffcm_cal_L5_s20-29     scffcm --L 5  --steps-csv $T9B --seed-start 20 --seeds 10 --allow-fresh   # ~2 min
step FRESH_scffcm_cal_L50_s20-29    scffcm --L 50 --steps-csv $T9C --datasets nonmnist --seed-start 20 --seeds 10 --allow-fresh   # ~3 min
# E-D on the fresh block (secondary): SC-FFCM (E-D steps) and pre_mass_off / pre_none_off at L=1, R=250
step FRESH_ed_scffcm_L1_R250_s20-29 scffcm --L 1 --rounds 250 --steps-csv $OUT/ed_scffcm_cal_L1_R250.csv --seed-start 20 --seeds 10 --allow-fresh   # ~2 min
step FRESH_ed_cells_L1_R250_s20-29  cells  --L 1 --rounds 250 --cells pre_mass_off,pre_none_off --seed-start 20 --seeds 10 --allow-fresh   # ~12 min
# DESCRIPTIVE ONLY, NOT PRE-REGISTERED: centralized FCM on seeds 20-29 (a reference line for tables; enters no test)
step FRESH_cfcm_s20-29_descriptive  cfcm --seed-start 20 --seeds 10 --allow-fresh   # ~2 min

# =====================================================================================================================
# 10. Final analysis (frozen code): §1 on s20-29, re-analysis of s00-09 / s10-19, E-A, E-C, decision analysis (10-29)
# =====================================================================================================================
if [[ "$(cd $EXP && shasum -a 256 scripts/T10_review.py)" != "$(cat $OUT/ANALYSIS_FROZEN)" ]]; then
  print "FAILED: scripts/T10_review.py changed after the analysis was frozen; final analysis not run"; exit 1
fi
analyze analysis_final --blocks s00-09,s10-19,s20-29 --allow-fresh

# =====================================================================================================================
# 11. E-H / E-I theory checks (separate module, resumable via results/t10/theory/_cache; no seed >= 5)
# =====================================================================================================================
if [[ $SMOKE != 1 ]]; then
  if ! (cd $EXP && $PY scripts/T10_theory.py run) > $LOG/theory_run.log 2>&1; then
    print "FAILED theory run -- see $LOG/theory_run.log"; exit 1
  fi
  if [[ ! -f $OUT/theory/SUMMARY.md ]]; then
    (cd $EXP && $PY scripts/T10_theory.py analyze) > $LOG/theory_analyze.log 2>&1 || { print "FAILED theory analyze"; exit 1; }
  fi
  [[ -f $OUT/theory/SUMMARY.md ]] || { print "FAILED: theory/SUMMARY.md missing"; exit 1; }
  print "[$(date '+%F %T')] ok    theory (E-H/E-I): $OUT/theory/SUMMARY.md"
fi

date '+%F %T' > $OUT/RUN_DONE
print "[$(date '+%F %T')] ALL DONE -> $OUT/RUN_DONE"
