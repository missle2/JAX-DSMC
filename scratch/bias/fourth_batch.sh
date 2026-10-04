#!/bin/bash
# after third_batch.sh: the volume-scaling gate with one-sided backward fits (normals)
cd /home/aakash/jax_dsmc/JAX-DSMC
while ps -p "$1" > /dev/null; do sleep 20; done
PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/bias/volume_scale.py \
  scratch/bias/volume_scale_f4_normals.jsonl 1.0 0 8 \
  '{"fnum_scale": 4, "knudsen": 0.5, "a_max": 64, "virtual": [2, 4], "eps": 0.2, "backward_normals": true}'
