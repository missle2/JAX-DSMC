#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
python3 -m examples.ensemble_couette --parameter speed --seeds 384 > scratch/bench/ensemble_couette_speed.log 2>&1
python3 -m examples.ensemble_couette --parameter temperature_left --seeds 384 > scratch/bench/ensemble_couette_tleft.log 2>&1
python3 -m examples.ensemble_couette --parameter accommodation --seeds 384 > scratch/bench/ensemble_couette_acc.log 2>&1
python3 -m examples.ensemble_couette --parameter speed --seeds 192 --epsilon 0 > scratch/bench/ensemble_couette_speed_eps0.log 2>&1
echo done
