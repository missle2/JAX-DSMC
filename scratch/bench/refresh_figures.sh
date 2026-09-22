#!/bin/bash
# Re-generate the forward/pathwise figures whose random streams changed on this branch.
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
stage() { echo "=== $1  $(date '+%F %T')"; }
stage shock_adjoint-pathwise; python3 -m examples.shock_adjoint --mode pathwise > scratch/bench/refresh_shock_adjoint_pathwise.log 2>&1; echo "    exit=$?"
stage shock-steepening; python3 -m examples.shock --dims 1 2 3 --plot > scratch/bench/refresh_shock_steepening.log 2>&1; echo "    exit=$?"
stage shock-bird; python3 -m examples.shock.normal --dims 1 > scratch/bench/refresh_shock_bird.log 2>&1; echo "    exit=$?"
stage couette-gradients; python3 -m examples.couette.gradients --seeds 512 --horizon-seeds 128 > scratch/bench/refresh_couette_gradients.log 2>&1; echo "    exit=$?"
stage DONE
