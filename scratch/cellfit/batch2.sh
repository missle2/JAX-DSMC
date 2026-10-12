#!/bin/bash
# The per-cell backward fits on fnum x4 seeds 3-11 (seeds 0-2: squash_cell1.jsonl): the
# 12-seed comparison against the per-seed FD of ppc_f4.jsonl.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
cd $R
SET='[{"S":4,"K":8,"k":32,"basis":"cell:quad"},{"S":4,"K":8,"k":32,"basis":"cell:quad:rb"},{"S":4,"K":8,"k":32,"basis":"cell:cubicv:adapt1.5"},{"S":4,"K":8,"k":32,"basis":"cell:quad:adapt1.5"}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/cellfit/squash_cell1.jsonl 1.0 3 12 "{\"fnum_scale\": 4, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [], \"settings\": $SET}"
echo BATCH2_DONE
