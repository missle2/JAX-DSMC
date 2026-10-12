#!/bin/bash
# Path 2a: particles-per-cell scaling of the Kn 0.5 squash gap at S x K = 4 x 8 (paired FD
# h = 0.2 per seed), then the FD-only references (h = 0.1, 0.2) per resolution.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
KN='"knudsen": 0.5, "a_max": 64'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/collconv/ppc_f8.jsonl 1.0 0 16 "{\"fnum_scale\": 8, $KN}"
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/sweep.py $R/scratch/collconv/ppc_f4.jsonl 1.0 0 12 "{\"fnum_scale\": 4, $KN}"
$L gpu env PYTHONPATH=. python3 $R/scratch/bias/fd_only.py $R/scratch/collconv/fd_f8.jsonl 1.0 0 64 "{\"fnum_scale\": 8, $KN, \"fd_steps\": [0.1, 0.2]}"
$L gpu env PYTHONPATH=. python3 $R/scratch/bias/fd_only.py $R/scratch/collconv/fd_f4.jsonl 1.0 0 48 "{\"fnum_scale\": 4, $KN, \"fd_steps\": [0.1, 0.2]}"
echo PPC_BATCH_DONE
