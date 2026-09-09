#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
# wait for the running 1-D ensemble to release the GPU
while pgrep -f "turbulence --dim 1 --side 1024" > /dev/null; do sleep 30; done
# 2-D viscosity at npc_eff = 1600: enough for a ~2 e-fold window with NO correction
python3 -u -m examples.turbulence --dim 2 --side 128 --npc 100 --efolds 2.0 \
  --stride 8 --a-max 128 --ensemble 16 --save scratch/bird/visc_d2e \
  > scratch/bird/visc_d2e.log 2>&1
echo DONE > scratch/bird/visc_d2e.done
