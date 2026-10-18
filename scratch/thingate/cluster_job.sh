#!/bin/bash
# One job of the 2026-10-17 thin-body adjoint gates (examples/mesh/cluster/study.py --group
# thingate), run from the repository root by job.sbatch:  bash scratch/thingate/cluster_job.sh <job>
# Each job appends one json line per seed to scratch/thingate/cluster/<job>.jsonl and, per
# (seed, backward setting), the shipped variant's per-coefficient terms to
# scratch/thingate/cluster/maps/<job>/*.npz.  CHECK=1 only resolves the job; SMOKE=1 runs it at
# toy size (fnum x16, a short warm-up, 30 captured steps, one seed, two settings) into
# scratch/thingate/cluster_smoke/ (the preflight).  Summary: scratch/thingate/summarise.py.
#
# The error: the adjoint of bodies under ~2 DSMC cells (1 m) thick, at M 10 and Kn 0.5 --
#   squash, cubic-in-v fit (squashkn2 S2, iterate 20, s = 0.1929, 0.58 m thick): dD/ds = -29.9
#     against +1.5, all of it the pathwise v part (grad_v lambda at the hits); |pathwise_v|
#     grows below s ~ 0.30 for the cubic fit only (quadratic runs never got that thin);
#   optm Kn 0.5 M 10 (circle_long, free space): the nose's gradient spikes 70-14 000x from
#     iterate ~36 and the drag saw-tooths (scratch/optm/analysis_optm.py).
# Free-molecular M 10 (no collision fit) shows neither.  Every job pairs the backward fits ON
# ONE FORWARD per seed (only the backward and the fits differ), against CRN central FDs:
#   Q   cell:quad:adapt1.5:align0.9:both            the shipped fit until 2026-10-12
#   C   cell:cubicv:adapt1.5:align0.9:both          the shipped fit (CELL_BASIS)
#   CR  cell:cubicv:adapt1.5:ridge1e-3:align0.9:both  C, ridged: is it extrapolation in v?
#   CA  cell:cubicv:adapt4:align0.9:both            C where a cell has >= 4 nodes a column
#   CL  cell:cubicv:adapt1.5:loo:align0.9:both      C, order chosen per cell by leave-one-out
# (backward_normals is off in production, so align / both are inert: one unsplit fit a cell.)
#
#   sq_M10_s{s}_b{j}     squash, Kn 0.5 M 10 (squashkn2's config), s = 0.193 (the spike's exact
#                        s; seeds 0, 1 of b0 ARE its two seeds), 0.25, 0.3, 0.45 (0.58 / 0.75 /
#                        0.9 / 1.35 m thick); 5 fits; FD h = 0.02; seeds 4j .. 4j+3.  ~1 h.
#   sq_M10off_s{s}_b{j}  the same with the body a quarter cell off the cell faces (centre y
#                        5.25).  At s = 0.193 (y 4.71..5.29 centred) each face's cut cell is 42 %
#                        flow (a 0.21 m sliver, few particles); off-centre (y 4.96..5.54) 92 %
#                        flow and the row between is all body.  Cut-cell occupancy at equal
#                        thickness.
#   sqg2_M10_s{s}_b{j}   grid_scale 2 (0.25 m cells, 4x the particles, same per cell): the body
#                        2-4 cells thick.  Q and C; seeds 2j .. 2j+1.  H200, ~2-3 h.
#   sq_M032_s{s}_b{j}    Kn 0.5 at the deck's M 0.32 at s = 0.193, 0.3: does the speed ratio
#                        (cubic in v over a hypersonic cell's velocity spread) matter?  5 fits.
#   sq_FM10_s{s}_b0      free-molecular M 10 (1 x 2 pull-back): the shape fits' share, no
#                        collision fit to blame.  Q and C.
#   om_it{k}_b0          the optm Kn 0.5 M 10 iterate k (circle_long, free space) with the
#                        optimiser's own config and seeds 0-3 (its common random numbers: C
#                        reproduces its gradient), 5 fits; FDs along the uniform shift and a
#                        Gaussian bump (0.5 m) at the nose, h = 0.05.  k = 30 (before the
#                        spikes, blunt), 58 (spike), 109, 156 (largest spike), 232 (spike),
#                        246 (the sting).  H200, ~1 h.
#   omg2_it{k}_b0        om at grid_scale 2, Q and C, seeds 0-1, nose FD.  H200, ~3 h.
set -euo pipefail
job="$1"
out=scratch/thingate/cluster
n4=4; n2=2; SMALL=''; SMALL_OM=''; WARM_OM=''
Q="cell:quad:adapt1.5:align0.9:both"
C="cell:cubicv:adapt1.5:align0.9:both"
CR="cell:cubicv:adapt1.5:ridge1e-3:align0.9:both"
CA="cell:cubicv:adapt4:align0.9:both"
CL="cell:cubicv:adapt1.5:loo:align0.9:both"
FITS=("${C}" "${Q}" "${CR}" "${CA}" "${CL}")
if [ "${SMOKE:-0}" = 1 ]; then
  out=scratch/thingate/cluster_smoke; n4=1; n2=1; FITS=("${C}" "${Q}")
  # M 10 crosses 3 m (the inflow face to the squash body) in 40 steps; circle_long's free box
  # puts the body 9 m from the faces
  SMALL=', "fnum_scale": 16, "warmup": 120, "steps": 30, "window": 20'
  SMALL_OM=', "fnum_scale": 32, "steps": 30, "window": 20'
  WARM_OM=200
fi
OPTM=examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_adam_tan_sm8_free
KN_M10='"fnum_scale": 1, "knudsen": 0.5, "a_max": 256, "mach": 10.0, "dt": 2.5e-5, "slots": 160000'
KN_M032='"fnum_scale": 1, "knudsen": 0.5, "a_max": 64'
FM_M10='"fnum_scale": 1, "knudsen": 0.0, "mach": 10.0, "dt": 2.5e-5, "slots": 160000'
OM_CFG='"case": "circle_long", "fnum_scale": 2, "warmup": 2800, "boundary": "free", "knudsen": 0.5, "mach": 10.0, "dt": 2.5e-5, "a_max": 256, "backward_normals": false, "levelset_h": 0.0625'
s_value() {  # the squash label -> s (0.193: the squashkn2 S2 spike's s, exactly)
  case "$1" in
    0.193) echo 0.19285115115165288 ;;
    0.25|0.3|0.45) echo "$1" ;;
    *) echo "unknown s $1" >&2; exit 1 ;;
  esac
}
settings() {  # sweep.py settings: "S K" then the fits
  local sk="$1"; shift
  local set=""
  for b in "$@"; do set="${set:+${set},}{\"S\":${sk% *},\"K\":${sk#* },\"k\":32,\"basis\":\"${b}\"}"; done
  echo "[${set}]"
}
geom_settings() {
  local set=""
  for b in "$@"; do set="${set:+${set},}{\"basis\":\"${b}\",\"k\":32}"; done
  echo "[${set}]"
}
squash() {  # job s seed0 seed1 config settings
  mkdir -p "${out}"
  python3 scratch/collconv/sweep.py "${out}/$1.jsonl" "$2" "$3" "$4" \
    "{$5, \"fd_steps\": [0.02], \"timing_band\": 10, \"extra_variants\": false, \"settings\": $6, \"maps\": \"${out}/maps/$1\"${7:-}${SMALL}}"
}
optm_iterate() {  # job iterate seed0 seed1 settings directions [extra config]
  local f="${OPTM}/coef_$2.npy"
  [ -f "${f}" ] || { echo "missing ${f} (pull the optm results first)"; exit 1; }
  [ "${CHECK:-0}" = 1 ] && exit 0
  local warm=""
  [ -n "${WARM_OM}" ] && warm=", \"warmup\": ${WARM_OM}"
  mkdir -p "${out}"
  python3 scratch/cellfit/geom_gate.py "${out}/$1.jsonl" "${OPTM}" "$2" "$3" "$4" \
    "{${OM_CFG}, \"virtual\": [4, 8], \"settings\": $5, \"directions\": $6, \"fd_h\": [0.05], \"maps\": \"${out}/maps/$1\"${7:-}${SMALL_OM}${warm}}"
}
case "${job}" in
  sq_M10_s*_b*|sq_M10off_s*_b*|sq_M032_s*_b*)
    kind="${job%%_s*}"; rest="${job#*_s}"; label="${rest%%_b*}"; j="${rest##*_b}"
    s="$(s_value "${label}")"
    [ "${CHECK:-0}" = 1 ] && exit 0
    case "${kind}" in
      sq_M10) squash "${job}" "${s}" "$((4 * j))" "$((4 * j + n4))" "${KN_M10}" "$(settings "4 8" "${FITS[@]}")" ;;
      sq_M10off) squash "${job}" "${s}" "$((4 * j))" "$((4 * j + n4))" "${KN_M10}" \
        "$(settings "4 8" "${FITS[@]}")" ', "case_centre": [5.0, 5.25]' ;;
      sq_M032) squash "${job}" "${s}" "$((4 * j))" "$((4 * j + n4))" "${KN_M032}" "$(settings "4 8" "${FITS[@]}")" ;;
    esac ;;
  sqg2_M10_s*_b*)
    rest="${job#sqg2_M10_s}"; label="${rest%%_b*}"; j="${rest##*_b}"
    s="$(s_value "${label}")"
    [ "${CHECK:-0}" = 1 ] && exit 0
    squash "${job}" "${s}" "$((2 * j))" "$((2 * j + n2))" "${KN_M10}" "$(settings "4 8" "${C}" "${Q}")" \
      ', "grid_scale": 2, "slots": 640000' ;;
  sq_FM10_s*_b0)
    rest="${job#sq_FM10_s}"; label="${rest%%_b*}"
    s="$(s_value "${label}")"
    [ "${CHECK:-0}" = 1 ] && exit 0
    squash "${job}" "${s}" 0 "${n4}" "${FM_M10}" "$(settings "1 2" "${C}" "${Q}")" ;;
  om_it*_b0)
    rest="${job#om_it}"; it="${rest%%_b*}"
    optm_iterate "${job}" "${it}" 0 "${n4}" "$(geom_settings "${FITS[@]}")" '["shift", "nose"]' ;;
  omg2_it*_b0)
    rest="${job#omg2_it}"; it="${rest%%_b*}"
    optm_iterate "${job}" "${it}" 0 "${n2}" "$(geom_settings "${C}" "${Q}")" '["nose"]' ', "grid_scale": 2' ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
