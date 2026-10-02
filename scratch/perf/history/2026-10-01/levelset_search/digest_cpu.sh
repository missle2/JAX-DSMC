#!/bin/bash
export JAX_PLATFORMS=cpu
D=/home/aakash/jax_dsmc/wt_levelset/scratch/perf/forward_digest.py
for code in /home/aakash/jax_dsmc/wt_levelset/scratch/ls/varB; do
  echo "=== $code"
  (cd $code && PYTHONPATH=$code:/home/aakash/jax_dsmc/wt_levelset taskset -c 5 python3 $D --compare new_cpu --bodies levelset 2>&1 | grep -v "^WARNING\|Tfrt\|E external")
done
