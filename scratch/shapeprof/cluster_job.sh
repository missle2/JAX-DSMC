#!/bin/bash
# One job of the 2026-10-19 shape-gradient profile (examples/mesh/cluster/study.py --group
# shapeprof), from the repository root:  bash scratch/shapeprof/cluster_job.sh <job>
# Output scratch/shapeprof/cluster/<job>.json (+ <job>.trace/, a perfetto trace of one seed).
#   optm_it{k}   the optm Kn 0.5 M 10 run's iterate k with its own config (4 x 8 pull-back),
#                seed 0: per-phase seconds of the shipped setting, the unguarded cubic and the
#                plain k32 slope fit (scratch/shapeprof/profile.py).  H200, ~30 min.
set -euo pipefail
job="$1"
out=scratch/shapeprof/cluster
OPTM=examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_adam_tan_sm8_free
case "${job}" in
  optm_it*)
    it="${job#optm_it}"
    f="${OPTM}/coef_${it}.npy"
    [ -f "${f}" ] || { echo "missing ${f}"; exit 1; }
    [ "${CHECK:-0}" = 1 ] && exit 0
    mkdir -p "${out}"
    python3 scratch/shapeprof/profile.py "${out}/${job}.json" "${OPTM}" "${it}" ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
