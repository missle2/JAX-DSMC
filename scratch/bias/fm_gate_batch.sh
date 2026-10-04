#!/bin/bash
# FM squash gate (kn_gate in FM mode): all six grad_x-lambda variants, fnum x2, 8 CRN seeds
cd /home/aakash/jax_dsmc/JAX-DSMC
for s in 0.35 0.6; do
  PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/kn_gate.py \
    scratch/bias/fm_gate_f2.jsonl $s 0 8 '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05]}'
done
