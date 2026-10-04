#!/bin/bash
# Kn 0.5 squash gate at s = 1.0: fnum x2 (~75 particles/cell), S x K = 2 x 4, 8 CRN seeds
cd /home/aakash/jax_dsmc/JAX-DSMC
PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/kn_gate.py \
  scratch/bias/kn_f2.jsonl 1.0 "${1:-0}" "${2:-8}" \
  '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "virtual": [2, 4], "fd_steps": [0.05], "timing_band": 10}'
