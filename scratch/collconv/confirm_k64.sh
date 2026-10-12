#!/bin/bash
# 12-seed confirmation of the backward fit k=64 (whole-cell-equivalent) against the per-seed
# FD h=0.2 already in ppc_f4.jsonl (same keys -> same forward/FD; deterministic to ~1e-11).
# Seeds 0-2 have k64 in sweep_f4.jsonl; seeds 3-6 ran 2026-10-10 (then the session ended);
# this runs seeds 7-11.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
SET='[{"S":4,"K":8,"k":64}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/collconv/confirm_k64_f4.jsonl 1.0 7 12 "{\"fnum_scale\": 4, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [], \"settings\": $SET}"
echo CONFIRM_K64_DONE
