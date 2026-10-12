#!/bin/bash
# Path 2b: paired hyper-parameter sweep at fnum x4 (37 ppc; same gradient as fnum x2), 3 seeds.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
SET='[{"S":4,"K":8,"k":32},{"S":1,"K":8,"k":32},{"S":2,"K":8,"k":32},{"S":8,"K":8,"k":32},{"S":4,"K":4,"k":32},{"S":4,"K":16,"k":32},{"S":4,"K":8,"k":16},{"S":4,"K":8,"k":64}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/collconv/sweep_f4.jsonl 1.0 0 3 "{\"fnum_scale\": 4, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [], \"settings\": $SET}"
echo SWEEP_BATCH_DONE
