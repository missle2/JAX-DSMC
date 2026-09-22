#!/bin/bash
# Diagnose the -8 % on the reservoir temperature: exact last-step ghosts, and particle count.
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 192 --exact-last > scratch/bench/ensemble_temp_exactlast.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 128 --particles 512 > scratch/bench/ensemble_temp_N512.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 64 --particles 2048 > scratch/bench/ensemble_temp_N2048.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 192 --steps 1 > scratch/bench/ensemble_temp_M1.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 192 --steps 2 > scratch/bench/ensemble_temp_M2.log 2>&1
echo done
