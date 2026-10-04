#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/restitution.py \
  scratch/bias/restitution_seq_f2.jsonl 0 4 '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "pullback": "sequential"}'
