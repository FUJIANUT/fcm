#!/bin/zsh
# T11 post hoc plan (2026-10-07): alpha_s calibration on seed 100, common 1e-5 stopping rule (U), calibrated alpha_s
# (C), decision analysis. Run from any directory:  caffeinate -i zsh scripts/T11_run_all.sh
set -e
cd "${0:A:h}/.."
PY=python3
W=${T11_WORKERS:-12}
mkdir -p results/t11_posthoc/logs
step() { local tag=$1; shift; echo "[$(date +%H:%M:%S)] start $tag"; $PY scripts/T11_posthoc.py "$@" --workers $W > results/t11_posthoc/logs/$tag.log 2>&1; echo "[$(date +%H:%M:%S)] done  $tag"; }
step selftest selftest
step cal_L5 calibrate --L 5
step cal_L1 calibrate --L 1
step U_L5 runs --variant U --L 5
step U_L1 runs --variant U --L 1
step C_L5 runs --variant C --L 5
step C_L1 runs --variant C --L 1
step analyze analyze --force
touch results/t11_posthoc/RUN_DONE
echo "[$(date +%H:%M:%S)] all done"
