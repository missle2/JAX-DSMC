#!/bin/bash
B=/home/aakash/jax_dsmc/wt_levelset/scratch/perf/bench_forward.py
for code in "$@"; do
  echo "=== $code"
  (cd $code && PYTHONPATH=$code:/home/aakash/jax_dsmc/wt_levelset python3 $B --bodies levelset 2>&1 | grep -v "^WARNING\|Tfrt\|E external")
done
