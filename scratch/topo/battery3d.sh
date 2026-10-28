#!/bin/bash
# 3-D Newtonian L / D battery of the topology guard and the thickness constraint (2026-10-27):
# scratch/reinit/battery3d.sh's L / D problem (sphere_long at h = 0.2, Helmholtz, ox_adam, 150
# iterates, volume >= 0.5 V0), against its n3d_ld_hz_re0 (no reinitialisation: L/D 4.63, one
# piece, a 0.2-0.9-knot plate) and n3d_ld_hz_re1 (reinitialised every iterate: 10 pieces).
# Thickness: the constraint T <= 0.02 at 2 knots, multiplier from 1 at rate 10.  One run at a
# time on the GPU.
cd /home/aakash/jax_dsmc/JAX-DSMC || exit 1
COMMON="--surrogate newtonian --case sphere_long --levelset-h 0.2 --levelset-samples 16
 --iterations 150 --step 0.04 --smooth 3 --band 0.8 --keepout 0.3 --boundary free
 --recentre 1 --frames-every 0 --optimizer ox_adam --filter helmholtz
 --objective lift_drag --area-mode floor --area-floor 0.5"
THICK="--thickness 1 --thickness-rate 10"
ARMS=(
  "hz_re0_gt|--reinit-every 0 --topology-guard 1 ${THICK}"
  "hz_re0_g|--reinit-every 0 --topology-guard 1"
  "hz_re0_t|--reinit-every 0 ${THICK}"
  "hz_re1_g|--reinit-every 1 --topology-guard 1"
)
for arm in "${ARMS[@]}"; do
  name=${arm%%|*}; flags=${arm#*|}
  tag="n3d_ld_${name}"
  echo "[$(date +%H:%M:%S)] $tag"
  PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu /home/aakash/envs/jax-env/bin/python \
    -m examples.mesh.optimize --tag "$tag" $COMMON $flags 2>&1 \
    | grep -E '^iterate (0|25|50|75|100|125|149):|converged|Error|Traceback'
done
echo done
