#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
# 1. broadband IC, Ma = 1.0 -> Re ~ 200, k_d ~ 34.  Tests both fixes at once.
python3 -u -m examples.turbulence --dim 1 --side 1024 --npc 200 --mach 1.0 \
  --n-eddy 1.5 --stride 8 --a-max 2400 --spectrum --ic broadband --k-max 4 \
  --ensemble 8 --save scratch/bird/turb_d1b \
  > scratch/bird/turb_d1b.log 2>&1
echo DONE > scratch/bird/turb_d1b.done
# 2. 2-D viscosity re-run at 2.5 e-folds, MATCHING the original single-seed run so
#    the corrected/uncorrected comparison is not confounded by window length.
python3 -u -m examples.turbulence --dim 2 --side 128 --npc 100 --efolds 2.5 \
  --stride 8 --a-max 128 --ensemble 16 --save scratch/bird/visc_d2e2 \
  > scratch/bird/visc_d2e2.log 2>&1
echo DONE > scratch/bird/visc_d2e2.done
