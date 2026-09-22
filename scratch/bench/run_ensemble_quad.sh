#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 384 > scratch/bench/ensemble_temp.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter nrho --seeds 384 > scratch/bench/ensemble_nrho.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 128 --particles 512 > scratch/bench/ensemble_temp_N512.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 64 --particles 2048 > scratch/bench/ensemble_temp_N2048.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 192 --particles 256 --steps 8 > scratch/bench/ensemble_temp_N256_M8.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter nrho --seeds 192 --particles 256 --steps 8 > scratch/bench/ensemble_nrho_N256_M8.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 192 --no-collision-term > scratch/bench/ensemble_temp_nocoll.log 2>&1
echo done
