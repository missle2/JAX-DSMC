#!/bin/bash
# One job of the 2026-10-18 leverage-guard gates (examples/mesh/cluster/study.py --group
# guardgate), run from the repository root by job.sbatch:  bash scratch/guardgate/cluster_job.sh <job>
# Rows to scratch/guardgate/cluster/<job>.jsonl, per (seed, fit) term maps and the guard's
# per-cell order counts to scratch/guardgate/cluster/maps/<job>/.  CHECK=1 only resolves the
# job.  Summary: python3 scratch/thingate/summarise.py scratch/guardgate/cluster.
#
# thingate (2026-10-18) found the cubic-in-v fit's thin-body M 10 blow-up is extrapolation:
# post-collision queries in the empty velocity gap between the beam and the wall-reflected gas
# where the cubic's leverage reaches ~10^3 (scratch/thingate/leverage_toy.py).  guard<h>
# (jax_dsmc/cell_fit.py) takes, per query, the highest order whose leverage there is <= h.
# Every job pairs the fits on ONE forward per seed against CRN central FDs:
#   C    cell:cubicv:adapt1.5:align0.9:both          the shipped fit
#   Q    cell:quad:adapt1.5:align0.9:both            its predecessor (2-4 % low)
#   G<h> cell:cubicv:adapt1.5:guard<h>:align0.9:both  h = 8, 32, 128 (unit, 2026-10-18: h = 2
#        moved the sparse inlet gates (N128) -13 / -20 % off C, h 8 -3 / -7 %, h 32 within 0.6 %;
#        relax / Couette unchanged at every h)
# Expected (stated before the runs): on the blunt squash G within 0.3 % of C (paired) and the
# guard switching < ~5 % of queries; on the thin M 10 squash and the optm spike iterates G within
# 2 s.e. of the FD, max |lambda| near Q's; the stored device gates unchanged.
#
#   sq_M032_s0.6_b{j}  blunt Kn 0.5, the deck's M 0.32, s = 0.6 (where C beat Q by 3.1 +- 0.2 %
#                      paired, 2026-10-12); seeds 4j .. 4j+3, j = 0..3 (16 seeds).  FD h 0.1.
#   sq_M10_s{s}_b{j}   thin Kn 0.5 M 10 (thingate's sq_M10), s = 0.193 / 0.25 / 0.3 / 0.45.
#   om_it{k}_b0        thingate's optm Kn 0.5 M 10 iterates 30 (blunt), 58, 156, 246 (h200).
#   unit               scratch/cellfit/gates.py: the stored relax / Couette / inlet device
#                      gates, 4 seeds, C and G<h> paired with the stored per-query gradient.
set -euo pipefail
job="$1"
# the guard's order counts (cell_fit.TRACE) are a jax.debug.callback, which needs a CPU device
# beside the GPU (env.sh sets JAX_PLATFORMS=cuda); the GPU stays the default device
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda},cpu"
out=scratch/guardgate/cluster
Q="cell:quad:adapt1.5:align0.9:both"
C="cell:cubicv:adapt1.5:align0.9:both"
G8="cell:cubicv:adapt1.5:guard8:align0.9:both"
G32="cell:cubicv:adapt1.5:guard32:align0.9:both"
G128="cell:cubicv:adapt1.5:guard128:align0.9:both"
FITS=("${C}" "${Q}" "${G8}" "${G32}" "${G128}")
OPTM=examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_adam_tan_sm8_free
KN_M10='"fnum_scale": 1, "knudsen": 0.5, "a_max": 256, "mach": 10.0, "dt": 2.5e-5, "slots": 160000'
KN_M032='"fnum_scale": 1, "knudsen": 0.5, "a_max": 64'
OM_CFG='"case": "circle_long", "fnum_scale": 2, "warmup": 2800, "boundary": "free", "knudsen": 0.5, "mach": 10.0, "dt": 2.5e-5, "a_max": 256, "backward_normals": false, "levelset_h": 0.0625'
s_value() {  # the squash label -> s (0.193: the squashkn2 S2 spike's s, exactly)
  case "$1" in
    0.193) echo 0.19285115115165288 ;;
    0.25|0.3|0.45|0.6) echo "$1" ;;
    *) echo "unknown s $1" >&2; exit 1 ;;
  esac
}
settings() {
  local set=""
  for b in "${FITS[@]}"; do set="${set:+${set},}{\"S\":4,\"K\":8,\"k\":32,\"basis\":\"${b}\"}"; done
  echo "[${set}]"
}
geom_settings() {
  local set=""
  for b in "${FITS[@]}"; do set="${set:+${set},}{\"basis\":\"${b}\",\"k\":32}"; done
  echo "[${set}]"
}
squash() {  # job s seed0 seed1 config fd
  mkdir -p "${out}"
  python3 scratch/collconv/sweep.py "${out}/$1.jsonl" "$2" "$3" "$4" \
    "{$5, \"fd_steps\": [$6], \"timing_band\": 10, \"extra_variants\": false, \"settings\": $(settings), \"maps\": \"${out}/maps/$1\"}"
}
case "${job}" in
  sq_M032_s0.6_b[0-3]|sq_M10_s*_b[01])
    kind="${job%%_s*}"; rest="${job#*_s}"; label="${rest%%_b*}"; j="${rest##*_b}"
    s="$(s_value "${label}")"
    [ "${CHECK:-0}" = 1 ] && exit 0
    case "${kind}" in
      sq_M032) squash "${job}" "${s}" "$((4 * j))" "$((4 * j + 4))" "${KN_M032}" 0.1 ;;
      sq_M10) squash "${job}" "${s}" "$((4 * j))" "$((4 * j + 4))" "${KN_M10}" 0.02 ;;
    esac ;;
  om_it*_b0)
    rest="${job#om_it}"; it="${rest%%_b*}"
    f="${OPTM}/coef_${it}.npy"
    [ -f "${f}" ] || { echo "missing ${f} (pull the optm results first)"; exit 1; }
    [ "${CHECK:-0}" = 1 ] && exit 0
    mkdir -p "${out}"
    python3 scratch/cellfit/geom_gate.py "${out}/${job}.jsonl" "${OPTM}" "${it}" 0 4 \
      "{${OM_CFG}, \"virtual\": [4, 8], \"settings\": $(geom_settings), \"directions\": [\"shift\", \"nose\"], \"fd_h\": [0.05], \"maps\": \"${out}/maps/${job}\"}" ;;
  unit)
    [ "${CHECK:-0}" = 1 ] && exit 0
    mkdir -p "${out}"
    python3 scratch/cellfit/gates.py "${out}/unit.jsonl" 4 "${C},${G8},${G32},${G128}" ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
