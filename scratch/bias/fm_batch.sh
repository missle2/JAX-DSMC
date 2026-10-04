#!/bin/bash
# FM squash bias study: s = 1.0 and 0.35, 8 seeds each, fnum x2 (fits the 4 GB GPU)
cd /home/aakash/jax_dsmc/JAX-DSMC
for s in 1.0 0.35; do
  PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/fm_exact.py \
    scratch/bias/fm_f2.jsonl $s 0 8 '{"fd_steps": [0.05], "fnum_scale": 2}'
done
