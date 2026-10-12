#!/bin/bash
# One job of the per-cell backward fit's cluster gates (examples/mesh/cluster/study.py
# --group cellgate), run from the repository root by job.sbatch:
#     bash scratch/cellfit/cluster_job.sh <job>
# CHECK=1 only resolves the job (its name, shape file) and exits: the preflight's pass over
# every job.  Every job appends one json line per seed to scratch/cellfit/cluster/<job>.jsonl (its own
# file: concurrent jobs never share one).  Re-running a finished job repeats its seeds:
# move its file away first.  SMOKE=1 runs the same command at a toy size (fnum x 16, 30 + 30
# steps, one seed) into scratch/cellfit/cluster_smoke/ -- the local preflight
# (examples/mesh/cluster/preflight.sh) runs every job kind that way before submission.
#
#   sq_s{s}_seed{i}   Kn 0.5 squash at s, PRODUCTION settings (fnum x1, S x K = 4 x 8, 300
#                     captured / window 200, 10-step timing band), one seed: the backward
#                     fits per-query k = 32 (the old default), the shipped per-cell
#                     CELL_BASIS, cell:cubicv:adapt1.5 and the unguarded cell:quad (its
#                     instability at production density), all paired, + the CRN FD (h = 0.2).
#                     ~1 h/seed (the per-query fit is most of it).
#   optc_{shape}_kn_b{j}  an optc iterate's level set at Kn 0.5, fnum x1, seeds 4j .. 4j+3:
#                     two-sided per-query k = 32, two-sided and one-sided (split, complex
#                     cells two-sided) CELL_BASIS; block 0 also the one-sided per-query
#                     k = 32 (unstable on these shapes, 2026-10-11).  CRN FD along the
#                     uniform shift and the cos 2 theta mode, h = 0.05.  ~1.5 h/seed.
#   optc_{shape}_fm_b{j}  the same shape free-molecular (the regime the optc runs optimised
#                     in; the fit does no work there), 1 x 2 pull-back, seeds 8j .. 8j+7.
set -euo pipefail
job="$1"
out=scratch/cellfit/cluster
first=0
if [ "${SMOKE:-0}" = 1 ]; then out=scratch/cellfit/cluster_smoke; fi
mkdir -p "${out}"
# the toy size of SMOKE=1, appended LAST to each json (a repeated key takes the last value):
# every other key as in production
SMALL=''
if [ "${SMOKE:-0}" = 1 ]; then SMALL=', "fnum_scale": 16, "warmup": 30, "steps": 30, "window": 20'; fi
OPT=examples/mesh/results/optimize
declare -A SHAPE=(
  [c0]="${OPT}/circle_coarse_S1_adam 0" [a50]="${OPT}/circle_coarse_S1_adam 50"
  [a100]="${OPT}/circle_coarse_S1_adam 100" [a200]="${OPT}/circle_coarse_S1_adam 200"
  [s8a30]="${OPT}/circle_coarse_S8_adam 30" [g200]="${OPT}/circle_coarse_S2_gd 200"
)
CELL="$(python3 -c 'from jax_dsmc.ensemble_backward import CELL_BASIS; print(CELL_BASIS)')"
case "${job}" in
  sq_s*_seed*)
    rest="${job#sq_s}"; s="${rest%%_seed*}"; i="${rest##*_seed}"
    [ "${CHECK:-0}" = 1 ] && exit 0
    SET="[{\"S\":4,\"K\":8,\"k\":32,\"basis\":\"quadratic_full\"},{\"S\":4,\"K\":8,\"k\":32,\"basis\":\"${CELL}\"},{\"S\":4,\"K\":8,\"k\":32,\"basis\":\"cell:cubicv:adapt1.5\"},{\"S\":4,\"K\":8,\"k\":32,\"basis\":\"cell:quad\"}]"
    python3 scratch/collconv/sweep.py "${out}/${job}.jsonl" "${s}" "${i}" "$((i + 1))" \
      "{\"fnum_scale\": 1, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [0.2], \"timing_band\": 10, \"extra_variants\": false, \"settings\": ${SET}${SMALL}}" ;;
  optc_*_kn_b*)
    rest="${job#optc_}"; shape="${rest%%_kn_b*}"; j="${rest##*_kn_b}"
    read -r run it <<< "${SHAPE[${shape}]}"
    [ -f "${run}/coef_${it}.npy" ] || { echo "missing ${run}/coef_${it}.npy"; exit 1; }
    [ "${CHECK:-0}" = 1 ] && exit 0
    SET="[{\"basis\":\"quadratic_full\",\"k\":32,\"sided\":false},{\"basis\":\"${CELL}\",\"k\":32,\"sided\":false},{\"basis\":\"${CELL}\",\"k\":32,\"sided\":true}"
    if [ "${j}" = 0 ]; then SET="${SET},{\"basis\":\"quadratic_full\",\"k\":32,\"sided\":true}"; fi
    SET="${SET}]"
    n=4; [ "${SMOKE:-0}" = 1 ] && n=1
    python3 scratch/cellfit/geom_gate.py "${out}/${job}.jsonl" "${run}" "${it}" "$((4 * j))" \
      "$((4 * j + n))" "{\"fnum_scale\": 1, \"knudsen\": 0.5, \"a_max\": 64, \"settings\": ${SET}, \"fd_h\": [0.05]${SMALL}}" ;;
  optc_*_fm_b*)
    rest="${job#optc_}"; shape="${rest%%_fm_b*}"; j="${rest##*_fm_b}"
    read -r run it <<< "${SHAPE[${shape}]}"
    [ -f "${run}/coef_${it}.npy" ] || { echo "missing ${run}/coef_${it}.npy"; exit 1; }
    [ "${CHECK:-0}" = 1 ] && exit 0
    n=8; [ "${SMOKE:-0}" = 1 ] && n=1
    python3 scratch/cellfit/geom_gate.py "${out}/${job}.jsonl" "${run}" "${it}" "$((8 * j))" \
      "$((8 * j + n))" "{\"fnum_scale\": 1, \"knudsen\": 0.0, \"backward_normals\": false, \"virtual\": [1, 2], \"settings\": [{\"basis\": \"${CELL}\", \"k\": 32}], \"fd_h\": [0.05]${SMALL}}" ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
