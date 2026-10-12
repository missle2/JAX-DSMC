#!/bin/bash
# After confirm_k64.sh: the k64 field snapshot of seed 0 (offline CV study), then the
# per-cell backward fits on fnum x4 seeds 0-2 (paired with sweep_f4 / sweep_k_f4 rows).
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
cd $R
$L gpu env PYTHONPATH=. python3 $R/scratch/cellfit/snapshot.py $R/scratch/cellfit/snap_f4_s0_k64.npz 0 '{"S":4,"K":8,"k":64}'
SET='[{"S":4,"K":8,"k":32,"basis":"cell:quad"},{"S":4,"K":8,"k":32,"basis":"cell:quad:adapt1.5"},{"S":4,"K":8,"k":32,"basis":"cell:cubicv:adapt1.5"},{"S":4,"K":8,"k":32,"basis":"cell:quad:pool1"},{"S":4,"K":8,"k":32,"basis":"cell:quad:rb"}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/cellfit/squash_cell1.jsonl 1.0 0 3 "{\"fnum_scale\": 4, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [], \"settings\": $SET}"
echo BATCH1_DONE
