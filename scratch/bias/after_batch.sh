#!/bin/bash
# queued after next_batch.sh: (1) specular FM s = 1, all variants, 10-step timing band;
# (2) diffuse FM s = 1, 10-step band; (3) horizon M 300 vs 600, FM s = 0.6 and 1, fnum x4
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/kn_gate.py scratch/bias/fm_gate_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "timing_band": 10}'
run scratch/bias/kn_gate.py scratch/bias/diffuse_fm_f2_tb10.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "acc": 1.0, "silhouette_band": 0.03, "timing_band": 10}'
for s in 0.6 1.0; do
  run scratch/bias/horizon.py scratch/bias/horizon_fm_f4.jsonl $s 0 8 \
    '{"fnum_scale": 4, "virtual": [1, 2], "fd_steps": [0.05]}'
done
