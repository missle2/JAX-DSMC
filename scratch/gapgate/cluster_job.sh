#!/bin/bash
# One job of the 2026-10-12 hypothesis gates (examples/mesh/cluster/study.py --group gapgate),
# run from the repository root by job.sbatch:  bash scratch/gapgate/cluster_job.sh <job>
# Each job appends one json line per seed to scratch/gapgate/cluster/<job>.jsonl; CHECK=1
# only resolves the job; SMOKE=1 runs it at toy size (fnum x16, 30 + 30 steps, horizons
# 30 / 60, one seed) into scratch/gapgate/cluster_smoke/ (the preflight).
#
# The Kn 0.5 gap (shipped fit -3.7 +- 0.8 % vs FD; cubic-in-v fit +3.1 %; whole-run horizon
# +2.8 % on 4 seeds -- do they add, and is the horizon part the fill transient?):
#   sqh_t{T}_seed{i}   squash s = 1, Kn 0.5, fnum x1, 4 x 8, T = warm-up + captured steps
#                      (1000: the production run; 2000: the box filled), the SAME forward
#                      captured for its last 300 and 1000 steps, shipped CELL and CUBIC
#                      paired; CRN FD h = 0.2 of the T-step run.  ~10-15 min.
#   circh_t{T}_b{j}    the circle (optc iterate 0) the same way, Kn 0.5, g . d along the shift
#                      and cos 2 theta vs CRN FD h = 0.05, seeds 4j .. 4j+3.  ~1 h.
#   cub_{shape}_b{j}   CUBIC (split, as it would ship) vs CELL on the optc iterates at Kn 0.5
#                      (stability and agreement), seeds 4j .. 4j+3.  ~20 min.
# The free-molecular remainder and the S8 cos 2 theta bias (both in the tally term), and the
# heavy-tailed gradients of the elongated iterates:
#   fmgz_{shape}_b{j}  optc iterate, free-molecular: the grazing cut-off 0.05 (shipped), 0.01,
#                      0.2 and 0 (exact) paired on one forward (it is derivative-only); FD
#                      h = 0.025, 0.05, 0.1 (linearity); every term's projection recorded;
#                      seeds 8j .. 8j+7.  ~1 h.
#   fmsm_{shape}_b{j}  the iterate smoothed (Gaussian sigma 2 knots = 12.5 cm on the
#                      coefficients, reinitialised, same area): the sub-cell features where the
#                      per-seed spikes sit removed -- does the variance go?  seeds 8j .. 8j+7.
#   sqfm_s{s}_b{j}     free-molecular squash at s, the grazing cut-offs as fmgz, FD h = 0.05,
#                      seeds 8j .. 8j+7.
set -euo pipefail
job="$1"
out=scratch/gapgate/cluster
n4=4; n8=8; H1=300; H2=1000
SMALL=''
if [ "${SMOKE:-0}" = 1 ]; then
  out=scratch/gapgate/cluster_smoke; n4=1; n8=1; H1=30; H2=60
  SMALL=', "fnum_scale": 16, "warmup": 30, "steps": 30, "window": 20'
fi
OPT=examples/mesh/results/optimize
declare -A SHAPE=(
  [c0]="${OPT}/circle_coarse_S1_adam 0" [a50]="${OPT}/circle_coarse_S1_adam 50"
  [a100]="${OPT}/circle_coarse_S1_adam 100" [a200]="${OPT}/circle_coarse_S1_adam 200"
  [s8a30]="${OPT}/circle_coarse_S8_adam 30" [g200]="${OPT}/circle_coarse_S2_gd 200"
)
CELL="$(python3 -c 'from jax_dsmc.ensemble_backward import CELL_BASIS; print(CELL_BASIS)')"
CUBIC="cell:cubicv:adapt1.5:align0.9:both"
KN='"fnum_scale": 1, "knudsen": 0.5, "a_max": 64'
FM='"fnum_scale": 1, "knudsen": 0.0, "backward_normals": false, "virtual": [1, 2]'
GZ=(0.05 0.01 0.2 0)
shape_of() {  # "<run> <iterate>" of a shape name; exits on a missing coefficient file
  read -r run it <<< "${SHAPE[$1]:-}"
  [ -n "${run:-}" ] || { echo "unknown shape $1"; exit 1; }
  [ -f "${run}/coef_${it}.npy" ] || { echo "missing ${run}/coef_${it}.npy"; exit 1; }
}
case "${job}" in
  sqh_t*_seed*)
    rest="${job#sqh_t}"; t="${rest%%_seed*}"; i="${rest##*_seed}"
    [ "${CHECK:-0}" = 1 ] && exit 0
    SET=""
    for b in "${CELL}" "${CUBIC}"; do
      for h in "${H1}" "${H2}"; do
        SET="${SET:+${SET},}{\"S\":4,\"K\":8,\"k\":32,\"basis\":\"${b}\",\"horizon\":${h}}"
      done
    done
    mkdir -p "${out}"
    python3 scratch/collconv/sweep.py "${out}/${job}.jsonl" 1.0 "${i}" "$((i + 1))" \
      "{${KN}, \"warmup\": $((t - 300)), \"fd_steps\": [0.2], \"timing_band\": 10, \"extra_variants\": false, \"settings\": [${SET}]${SMALL}}" ;;
  circh_t*_b*)
    rest="${job#circh_t}"; t="${rest%%_b*}"; j="${rest##*_b}"
    shape_of c0
    [ "${CHECK:-0}" = 1 ] && exit 0
    SET=""
    for b in "${CELL}" "${CUBIC}"; do
      for h in "${H1}" "${H2}"; do
        SET="${SET:+${SET},}{\"basis\":\"${b}\",\"k\":32,\"horizon\":${h}}"
      done
    done
    mkdir -p "${out}"
    python3 scratch/cellfit/geom_gate.py "${out}/${job}.jsonl" "${run}" "${it}" "$((4 * j))" \
      "$((4 * j + n4))" "{${KN}, \"warmup\": $((t - 300)), \"settings\": [${SET}], \"fd_h\": [0.05]${SMALL}}" ;;
  cub_*_b*)
    rest="${job#cub_}"; shape="${rest%%_b*}"; j="${rest##*_b}"
    shape_of "${shape}"
    [ "${CHECK:-0}" = 1 ] && exit 0
    mkdir -p "${out}"
    python3 scratch/cellfit/geom_gate.py "${out}/${job}.jsonl" "${run}" "${it}" "$((4 * j))" \
      "$((4 * j + n4))" "{${KN}, \"settings\": [{\"basis\":\"${CELL}\",\"k\":32},{\"basis\":\"${CUBIC}\",\"k\":32}], \"fd_h\": [0.05]${SMALL}}" ;;
  fmgz_*_b*|fmsm_*_b*)
    kind="${job%%_*}"; rest="${job#*_}"; shape="${rest%%_b*}"; j="${rest##*_b}"
    shape_of "${shape}"
    [ "${CHECK:-0}" = 1 ] && exit 0
    if [ "${kind}" = fmgz ]; then
      SET=""
      for g in "${GZ[@]}"; do SET="${SET:+${SET},}{\"basis\":\"${CELL}\",\"k\":32,\"grazing\":${g}}"; done
      EXTRA="\"settings\": [${SET}], \"fd_h\": [0.025, 0.05, 0.1]"
    else
      EXTRA="\"settings\": [{\"basis\":\"${CELL}\",\"k\":32}], \"fd_h\": [0.05], \"smooth_shape\": 2"
    fi
    mkdir -p "${out}"
    python3 scratch/cellfit/geom_gate.py "${out}/${job}.jsonl" "${run}" "${it}" "$((8 * j))" \
      "$((8 * j + n8))" "{${FM}, ${EXTRA}${SMALL}}" ;;
  sqfm_s*_b*)
    rest="${job#sqfm_s}"; s="${rest%%_b*}"; j="${rest##*_b}"
    [ "${CHECK:-0}" = 1 ] && exit 0
    SET=""
    for g in "${GZ[@]}"; do SET="${SET:+${SET},}{\"S\":1,\"K\":2,\"k\":32,\"basis\":\"${CELL}\",\"grazing\":${g}}"; done
    mkdir -p "${out}"
    python3 scratch/collconv/sweep.py "${out}/${job}.jsonl" "${s}" "$((8 * j))" "$((8 * j + n8))" \
      "{\"fnum_scale\": 1, \"fd_steps\": [0.05], \"timing_band\": 10, \"extra_variants\": false, \"settings\": [${SET}]${SMALL}}" ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
