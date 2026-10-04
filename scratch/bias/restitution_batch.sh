#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/restitution.py scratch/bias/restitution_fm_f2.jsonl 0 3 '{"fnum_scale": 2, "virtual": [1, 2]}'
run scratch/bias/restitution.py scratch/bias/restitution_f2.jsonl 1 6 '{"fnum_scale": 2, "knudsen": 0.5, "a_max": 64}'
