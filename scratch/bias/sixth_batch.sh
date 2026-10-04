#!/bin/bash
# Kn 0.5 horizon at fnum x2 with S x K = 2 x 8 (stable over long horizons), M 300 vs 450;
# then the FM leftovers
cd /home/aakash/jax_dsmc/JAX-DSMC
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/horizon.py scratch/bias/horizon_kn_f2.jsonl 1.0 0 6 \
  '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "virtual": [2, 8], "fd_steps": [0.2], "horizons": [300, 450]}'
run scratch/bias/kn_gate.py scratch/bias/fm_gate_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "timing_band": 10}'
run scratch/bias/kn_gate.py scratch/bias/diffuse_fm_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "acc": 1.0, "silhouette_band": 0.03, "timing_band": 10}'
