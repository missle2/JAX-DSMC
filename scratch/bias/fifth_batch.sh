#!/bin/bash
# after the FD-only batch: Kn 0.5 suspects for the -5 % gap
# (1) one-sided backward fits (backward_normals), fnum x2, 6 seeds (paired with kn_f2.jsonl)
# (2) horizon M 300 vs 600, Kn 0.5, fnum x4, 6 seeds
# then the FM leftovers (s = 1 with the 10-step band, FM horizon, diffuse 10-step band)
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/kn_gate.py scratch/bias/kn_f2_normals.jsonl 1.0 0 6 \
  '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "virtual": [2, 4], "fd_steps": [0.2], "timing_band": 10, "backward_normals": true}'
run scratch/bias/horizon.py scratch/bias/horizon_kn_f4.jsonl 1.0 0 6 \
  '{"fnum_scale": 4, "knudsen": 0.5, "a_max": 64, "virtual": [2, 4], "fd_steps": [0.2]}'
run scratch/bias/kn_gate.py scratch/bias/fm_gate_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "timing_band": 10}'
for s in 0.6 1.0; do
  run scratch/bias/horizon.py scratch/bias/horizon_fm_f4.jsonl $s 0 8 \
    '{"fnum_scale": 4, "virtual": [1, 2], "fd_steps": [0.05]}'
done
run scratch/bias/kn_gate.py scratch/bias/diffuse_fm_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "acc": 1.0, "silhouette_band": 0.03, "timing_band": 10}'
