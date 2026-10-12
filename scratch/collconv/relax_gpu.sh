#!/bin/bash
# Path 1b on the GPU: the M40 grad_v lambda gate with wider FD steps (the 0.05 / 0.025 CRN FD
# decorrelates over 40 steps of collisions).  Seeds 0-8 of relax_tx_M40.jsonl ran on the CPU.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
FD='"fd_steps": [0.2, 0.1, 0.05, 0.025]'
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/relax_ic.py $R/scratch/collconv/relax_tx_M40_gpu.jsonl 0 24 "{\"steps\": 40, \"virtual\": [[4, 8], [2, 4]], \"neighbors\": [32], $FD}"
$L gpu env PYTHONPATH=. python3 $R/scratch/collconv/relax_ic.py $R/scratch/collconv/relax_ux_M40_gpu.jsonl 0 24 "{\"control\": \"ux\", \"steps\": 40, \"virtual\": [[4, 8]], \"neighbors\": [32], $FD}"
echo RELAX_GPU_DONE
