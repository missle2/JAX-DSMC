#!/bin/bash
# after the FM gate: (1) diffuse FM gate at s = 1 (acc 1, silhouette band 0.03), 8 seeds;
# (2) Kn 0.5 specular gate at s = 1, fnum x2, S x K = 2 x 4, 8 seeds
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/kn_gate.py \
  scratch/bias/diffuse_fm_f2.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "acc": 1.0, "silhouette_band": 0.03}'
scratch/bias/kn_batch.sh 0 8
