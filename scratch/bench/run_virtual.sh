#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
for S in 1 2 4 8; do python3 -m examples.ensemble_relaxation --parameter diameter --seeds 48 --pullback virtual --substeps $S --partners 8 > scratch/bench/virt_relax_S$S.log 2>&1; done
for K in 2 32; do python3 -m examples.ensemble_relaxation --parameter diameter --seeds 48 --pullback virtual --substeps 4 --partners $K > scratch/bench/virt_relax_K$K.log 2>&1; done
python3 -m examples.ensemble_relaxation --parameter omega --seeds 48 --pullback virtual --substeps 4 --partners 8 > scratch/bench/virt_relax_omega.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 32 --particles 2048 --dt 0.5 --attempts 2048 --steps 4 --pullback virtual --substeps 4 --partners 8 > scratch/bench/virt_inlet_2048.log 2>&1
python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 64 --pullback virtual --substeps 4 --partners 8 > scratch/bench/virt_inlet_128.log 2>&1
echo done
