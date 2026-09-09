#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
while pgrep -f "probe_variance|shock_adjoint --mode" > /dev/null; do sleep 20; done
sleep 15
JAX_PLATFORMS=cuda python3 -u scratch/adjoint/probe_split_variance.py \
  > scratch/adjoint/split_var.log 2>&1
echo DONE > scratch/adjoint/split_var.done
