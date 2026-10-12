#!/bin/bash
# Path 2c: the backward fit's neighbour count k (k32 -> k64 moved the squash gradient +3.4 %
# +- 0.3 %, paired): does it level off?  Same fnum x4 seeds 0-2 as sweep_f4.jsonl (paired).
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
SET='[{"S":4,"K":8,"k":48},{"S":4,"K":8,"k":96},{"S":4,"K":8,"k":128}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/collconv/sweep_k_f4.jsonl 1.0 0 3 "{\"fnum_scale\": 4, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [], \"settings\": $SET}"
echo SWEEP_K_DONE
