#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/kn_gate.py \
  scratch/bias/kn_f2_seq.jsonl 1.0 0 4 \
  '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "pullback": "sequential", "fd_steps": [0.2], "timing_band": 10}'
