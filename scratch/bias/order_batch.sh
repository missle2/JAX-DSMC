#!/bin/bash
# after the grazing run: (1) Kn 0.5 gate with the collision-order term, same seeds and
# settings as kn_f2.jsonl (fnum x2, S x K = 2 x 4, 10-step band); (2) FM horizon 300 vs 600
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/kn_gate.py scratch/bias/kn_f2_order.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64, "virtual": [2, 4], "fd_steps": [0.2], "timing_band": 10}'
for s in 1.0 0.35; do
  run scratch/bias/horizon.py scratch/bias/horizon_fm_f4.jsonl $s 0 8 \
    '{"fnum_scale": 4, "virtual": [1, 2], "fd_steps": [0.05], "horizons": [300, 600]}'
done
