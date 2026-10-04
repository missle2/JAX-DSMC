#!/bin/bash
# after after_batch.sh: Kn 0.5 volume-scaling gate (fnum x4, eps 0.2, 8 seeds) and FD-only
# Kn 0.5 at h = 0.1, 0.2 (fnum x2, 32 seeds)
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/volume_scale.py scratch/bias/volume_scale_f4.jsonl 1.0 0 8 \
  '{"fnum_scale": 4, "knudsen": 0.5, "a_max": 64, "virtual": [2, 4], "eps": 0.2}'
run scratch/bias/fd_only.py scratch/bias/kn_fd_f2.jsonl 1.0 0 32 \
  '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "fd_steps": [0.1, 0.2]}'
