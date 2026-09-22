#!/bin/bash
# Rankine-Hugoniot acceptance case on the device, after the staged queue.
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
echo "=== rh $(date '+%F %T')"
python3 -m examples.ensemble_inlet_shock --parameter mach --objective density-ratio --particles 3200 --cells 80 --steps 300 --dt 0.2 --attempts 192 --seeds 16 --fd-steps 0.1,0.05 --device --pullback virtual --substeps 2 --partners 8 --nodes staged > scratch/bench/gpu_rh_mach.log 2>&1
echo "done $(date '+%F %T')"
