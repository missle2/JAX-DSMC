#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/kn_gate.py scratch/bias/fm_gate_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "timing_band": 10}'
run scratch/bias/kn_gate.py scratch/bias/diffuse_fm_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "acc": 1.0, "silhouette_band": 0.03, "timing_band": 10}'
