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
set -euo pipefail
job="$1"
out=scratch/bias/cluster
mkdir -p "${out}"
KN='"fnum_scale": 1, "knudsen": 0.5, "a_max": 64, "virtual": [4, 8]'
case "${job}" in
  kn_seed*)
    i="${job#kn_seed}"
    python3 scratch/bias/kn_gate.py "${out}/${job}.jsonl" 1.0 "${i}" "$((i + 1))" \
      "{${KN}, \"fd_steps\": [0.2], \"timing_band\": 10}" ;;
  kn_fd*)
    j="${job#kn_fd}"
    python3 scratch/bias/fd_only.py "${out}/${job}.jsonl" 1.0 "$((16 * j))" "$((16 * j + 16))" \
      '{"fnum_scale": 1, "knudsen": 0.5, "a_max": 64, "fd_steps": [0.1, 0.2]}' ;;
  kn_hor_seed*)
    i="${job#kn_hor_seed}"
    python3 scratch/bias/horizon.py "${out}/${job}.jsonl" 1.0 "${i}" "$((i + 1))" \
      "{${KN}, \"fd_steps\": [0.2], \"horizons\": [300, 1000]}" ;;
  fm_hor_s*)
    s="${job#fm_hor_s}"
    python3 scratch/bias/horizon.py "${out}/${job}.jsonl" "${s}" 0 8 \
      '{"fnum_scale": 1, "virtual": [1, 2], "fd_steps": [0.05], "horizons": [300, 1000]}' ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
