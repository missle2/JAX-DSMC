#!/bin/bash
# MI100 SPARTA sphere abort (device-side "Surf_collide tsurf <= 0.0", job 36514785): build three
# variants on acompile, then run the sphere deck on each (and on the current default) on an MI100.
#
#   cd /projects/$USER/JAX-DSMC-sparta && bash bench/jobs/mi100_sphere_diag.sh
#
#   tgt  default flags + explicit GPU_TARGETS=gfx908 (acompile has no GPU to detect)
#   o1   -O1 instead of -O3 (a code-generation bug shows as o1 running)
#   mk   Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS (Kokkos' large-functor launch path)
# The test job runs whatever built (afterany).  Logs: build_mi100_diag_<id>.out (repo root, picked
# up by bench/sync_logs.sh) and joblogs/sphtest-alp-mi100_<id>.out.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$PWD"
mkdir -p joblogs bench/plans/alp-mi100
B=$(sbatch --parsable --job-name=build_mi100_diag --partition=acompile --qos=compile \
  --account="${ACCOUNT:-ucb856_asc1}" --ntasks=4 --time=04:00:00 --output="${REPO}/build_mi100_diag_%j.out" \
  --wrap "cd ${REPO} && for V in default:tgt o1:o1 multikernel:mk; do \
            echo; echo \"===== \${V##*:} (\$(date -Iseconds)) =====\"; \
            PLATFORM=alp-mi100 VARIANT=\${V%%:*} BUILD_TAG=\${V##*:} bash bench/env/build_sparta.sh \
              || echo \"===== \${V##*:} FAILED\"; \
          done")
echo "build job ${B}"
PLAN="${REPO}/bench/plans/alp-mi100/sphtest.plan"
for b in default tgt o1 mk; do
  echo "[ -x \"\${SPARTA_BUILD_DIR}/sparta-\${PLATFORM}-${b}/spa\" ] || { echo 'no build ${b}'; exit 1; }; python3 bench/run_bench.py sparta --exe \"\${SPARTA_BUILD_DIR}/sparta-\${PLATFORM}-${b}/spa\" --deck sphere --particles 64K --steps 200 --reps 1 --machine \"\${MACHINE}\" --variant sph_${b} --out \"\${RESULTS_DIR}/check\" --procs 1 --launch \"\$(pf_launch 1 1 bind)\" --accel '-k on g 1 -sf kk'"
done > "${PLAN}"
RES="$(PLATFORM=alp-mi100 bash -c 'source bench/env/platforms.sh > /dev/null && pf_resources_shared 1')"
# shellcheck disable=SC2086
T=$(sbatch --parsable ${RES} --dependency=afterany:"${B}" --time=00:30:00 --job-name=sphtest-alp-mi100 \
  --output="${REPO}/joblogs/%x_%j.out" --export=ALL,PLATFORM=alp-mi100,PLAN="${PLAN}" bench/jobs/bench.sbatch)
echo "sphere test job ${T} (after ${B})"
