#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
while kill -0 628041 2>/dev/null; do sleep 30; done
JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/bench/relax_fd_arbiter.py --parameter omega --particles 2000 --attempts 2048 --seeds 1600 --output scratch/bench/relax_fd_arbiter_N2000_omega.json > scratch/bench/relax_fd_arbiter_N2000_omega.log 2>&1
