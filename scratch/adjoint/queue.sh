#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
while pgrep -f "shock_adjoint --mode" > /dev/null; do sleep 20; done
sleep 10
JAX_PLATFORMS=cuda python3 -u scratch/adjoint/probe_cost.py > scratch/adjoint/cost.log 2>&1
echo DONE > scratch/adjoint/cost.done
JAX_PLATFORMS=cuda python3 -u scratch/adjoint/probe_variance.py > scratch/adjoint/variance.log 2>&1
echo DONE > scratch/adjoint/variance.done
