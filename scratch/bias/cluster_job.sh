#!/bin/bash
# One job of the cluster bias gate (examples/mesh/cluster/study.py --group biasgate), run
# from the repository root by job.sbatch:  bash scratch/bias/cluster_job.sh <job>
#
#   kn_seed{i}      Kn 0.5 paired gate at s = 1, PRODUCTION settings (fnum x1, S x K = 4 x 8,
#                   captured 300 / window 200, 10-step timing band): every grad_x-lambda
#                   variant, the along-flight identity with and without L* lambda, the volume
#                   term, the CRN FD (h = 0.2) and the FD with frozen cut volumes.  ~1 h/seed.
#   kn_fd{j}        Kn 0.5 FD only, fnum x1, 16 CRN seeds per job (j = 0..3 -> seeds 16 j ..),
#                   h = 0.1 and 0.2: the precise arbiter of kn_seed*.  ~1 h/job.
#   kn_hor_seed{i}  Kn 0.5 horizon: the same forward captured for the last 300 steps and for
#                   the WHOLE run (warm-up 0, 1000 steps: no truncation at all), fnum x1,
#                   S x K = 4 x 8.  ~4 h/seed, ~3.3x the memory of kn_seed (A100 40 GB / H200).
#   fm_hor_s{s}     free-molecular horizon 300 vs the whole run (1000), fnum x1, 8 seeds at
#                   s = 1 or 0.35: is the -2 % of the FM gate the truncation?  ~3 h/job.
#
# Every job appends one json line per seed to scratch/bias/cluster/<job>.jsonl and skips
# nothing (re-running a finished job repeats seeds: delete or move its file first).
# SMOKE=1: the same command at a toy size (fnum x 16, 30 + 30 steps, one seed) into
# scratch/bias/cluster_smoke/ (examples/mesh/cluster/preflight.sh).  Since 2026-10-11 the
# backward's field fit is the shipped per-cell CELL_BASIS (ensemble_backward.configure's
# default); the kn_seed gate is superseded by cellgate's sq_s1_seed* (fits paired).
set -euo pipefail
job="$1"
out=scratch/bias/cluster
n1=1; n8=8; n16=16
# the toy size, appended LAST to each json (a repeated key takes the last value)
SMALL=''
if [ "${SMOKE:-0}" = 1 ]; then
  out=scratch/bias/cluster_smoke; n8=1; n16=1
  SMALL=', "fnum_scale": 16, "warmup": 30, "steps": 30, "window": 20'
fi
# the horizon jobs' toy: 60 steps in all, captured 30 and 60 (the window 20 fits both)
HOR=', "total": 60, "horizons": [30, 60]' 
# CHECK=1: only resolve the job name (the preflight's pass over every job)
if [ "${CHECK:-0}" = 1 ]; then
  case "${job}" in kn_seed*|kn_fd*|kn_hor_seed*|fm_hor_s*) exit 0 ;; *) echo "unknown job ${job}"; exit 1 ;; esac
fi
mkdir -p "${out}"
KN='"fnum_scale": 1, "knudsen": 0.5, "a_max": 64, "virtual": [4, 8]'
case "${job}" in
  kn_seed*)
    i="${job#kn_seed}"
    python3 scratch/bias/kn_gate.py "${out}/${job}.jsonl" 1.0 "${i}" "$((i + 1))" \
      "{${KN}, \"fd_steps\": [0.2], \"timing_band\": 10${SMALL}}" ;;
  kn_fd*)
    j="${job#kn_fd}"
    python3 scratch/bias/fd_only.py "${out}/${job}.jsonl" 1.0 "$((16 * j))" "$((16 * j + n16))" \
      "{\"fnum_scale\": 1, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [0.1, 0.2]${SMALL}}" ;;
  kn_hor_seed*)
    i="${job#kn_hor_seed}"
    python3 scratch/bias/horizon.py "${out}/${job}.jsonl" 1.0 "${i}" "$((i + n1))" \
      "{${KN}, \"fd_steps\": [0.2], \"horizons\": [300, 1000]${SMALL}${SMALL:+${HOR}}}" ;;
  fm_hor_s*)
    s="${job#fm_hor_s}"
    python3 scratch/bias/horizon.py "${out}/${job}.jsonl" "${s}" 0 "${n8}" \
      "{\"fnum_scale\": 1, \"virtual\": [1, 2], \"fd_steps\": [0.05], \"horizons\": [300, 1000]${SMALL}${SMALL:+${HOR}}}" ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
