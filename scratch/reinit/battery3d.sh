#!/bin/bash
# 3-D Newtonian surrogate battery (2026-10-26): does dropping the per-iterate reinitialisation
# (--reinit-every 0) and / or a larger Helmholtz radius keep the nose and the L / D plate
# clean?  sphere_long at h = 0.2 (knot units as sph3d's h = 0.1: smooth 3, band 4 knots, step
# 0.2 knots), ox_adam, 150 iterates, fixed step (as the DSMC runs), one run at a time.
cd /home/aakash/jax_dsmc/JAX-DSMC || exit 1
COMMON="--surrogate newtonian --case sphere_long --levelset-h 0.2 --levelset-samples 16
 --iterations 150 --step 0.04 --smooth 3 --band 0.8 --keepout 0.3 --boundary free
 --recentre 1 --frames-every 0 --optimizer ox_adam"
declare -A PROB=(
  [drag]="--area-mode tangent"
  [ld]="--objective lift_drag --area-mode floor --area-floor 0.5"
)
# arm -> flags
ARMS=(
  "band_re1|--filter band --reinit-every 1"
  "band_re0|--filter band --reinit-every 0"
  "hz_re1|--filter helmholtz --reinit-every 1"
  "hz_re0|--filter helmholtz --reinit-every 0"
  "hz2_re0|--filter helmholtz --filter-radius 0.85 --reinit-every 0"
)
for prob in drag ld; do
  for arm in "${ARMS[@]}"; do
    name=${arm%%|*}; flags=${arm#*|}
    tag="n3d_${prob}_${name}"
    echo "[$(date +%H:%M:%S)] $tag"
    PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu /home/aakash/envs/jax-env/bin/python \
      -m examples.mesh.optimize --tag "$tag" $COMMON ${PROB[$prob]} $flags 2>&1 \
      | grep -E '^iterate (0|25|50|75|100|125|149):|converged|Error|Traceback'
  done
done
echo done
