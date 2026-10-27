#!/bin/bash
# The local band-vs-Helmholtz comparison (2026-10-26): Newtonian drag at fixed area from the
# circle_long circle, serially in one CPU slot, the gradients saved for the offline step replay.
set -uo pipefail
PY=/home/aakash/envs/jax-env/bin/python
COMMON="--surrogate newtonian --case circle_long --levelset-h 0.0625 --levelset-samples 16 \
  --iterations 80 --step 0.05 --smooth 8 --band 1.0 --keepout 1.0 --area-mode tangent \
  --recentre 1 --boundary free --save-gradient 1 --frames-every 0"
for arm in "band ox_adam" "helmholtz ox_adam" "band mma_s" "helmholtz mma_s" "helmholtz ox_gd"; do
  set -- $arm
  echo "[$(date +%T)] $1 $2"
  PYTHONPATH=. JAX_PLATFORMS=cpu /home/aakash/jax_dsmc/run_locked.sh cpu "${PY}" -m examples.mesh.optimize \
    --tag "lsf_newton_$1_$2" --filter "$1" --optimizer "$2" ${COMMON} 2>&1 | grep -E "^iterate (0|20|40|60|79):|Error|Traceback"
done
echo done
