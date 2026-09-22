#!/bin/bash
# Re-run every ensemble case under the sequential (exact-step) collision pull-back.
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
python3 -m examples.ensemble_relaxation --parameter diameter --seeds 192 > scratch/bench/ensemble_relax_diam.log 2>&1
python3 -m examples.ensemble_relaxation --parameter omega --seeds 192 > scratch/bench/ensemble_relax_omega.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 384 > scratch/bench/ensemble_temp.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter nrho --seeds 384 > scratch/bench/ensemble_nrho.log 2>&1
python3 -m examples.ensemble_couette --parameter speed --seeds 384 > scratch/bench/ensemble_couette_speed.log 2>&1
python3 -m examples.ensemble_couette --parameter temperature_left --seeds 384 > scratch/bench/ensemble_couette_tleft.log 2>&1
python3 -m examples.ensemble_couette --parameter accommodation --seeds 384 > scratch/bench/ensemble_couette_acc.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 192 --particles 256 --steps 8 > scratch/bench/ensemble_temp_N256_M8.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter nrho --seeds 192 --particles 256 --steps 8 > scratch/bench/ensemble_nrho_N256_M8.log 2>&1
python3 -m examples.ensemble_couette --parameter speed --seeds 192 --epsilon 0 > scratch/bench/ensemble_couette_speed_eps0.log 2>&1
echo done
