#!/bin/bash
D=/home/aakash/jax_dsmc/wt_levelset/scratch/perf/forward_digest.py
for code in /home/aakash/jax_dsmc/wt_levelset/scratch/ls/base /home/aakash/jax_dsmc/wt_levelset/scratch/ls/varA /home/aakash/jax_dsmc/wt_levelset; do
  echo "=== $code"
  (cd $code && PYTHONPATH=$code:/home/aakash/jax_dsmc/wt_levelset python3 $D --compare base_gpu --bodies levelset 2>&1 | grep -v "^WARNING\|Tfrt\|E external")
done
