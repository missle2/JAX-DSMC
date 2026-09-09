#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
# wait for the 2-D ensemble viscosity run to release the GPU
while [ ! -f scratch/bird/visc_d2e.done ]; do sleep 30; done
sleep 20
# Broadband IC, Ma = 1.0.  Re ~ Ma, so this buys Re ~ 200 (k_d ~ 34) at LOWER cost
# than the Ma = 0.3 run: same N, and 3.3x fewer steps for the same number of eddy
# times.  Tests both fixes at once -- the comb should be gone and a range should exist.
python3 -u -m examples.turbulence --dim 1 --side 1024 --npc 200 --mach 1.0 \
  --n-eddy 1.5 --stride 8 --a-max 400 --spectrum --ic broadband --k-max 4 \
  --ensemble 8 --save scratch/bird/turb_d1b \
  > scratch/bird/turb_d1b.log 2>&1
echo DONE > scratch/bird/turb_d1b.done
