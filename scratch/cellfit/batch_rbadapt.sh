#!/bin/bash
# rb (sigma-point angle expectation) combined with the order guard, paired on the 12 squash
# seeds of squash_cell1.jsonl (fnum x4, Kn 0.5): rb was only ever tested unguarded.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
SET='[{"S":4,"K":8,"k":32,"basis":"cell:quad:adapt1.5"},{"S":4,"K":8,"k":32,"basis":"cell:quad:adapt1.5:rb"},{"S":4,"K":2,"k":32,"basis":"cell:quad:adapt1.5:rb"}]'
XLA_FLAGS=--xla_gpu_enable_command_buffer= $L gpu env PYTHONPATH=$R python3 $R/scratch/collconv/sweep.py $R/scratch/cellfit/squash_rbadapt.jsonl 1.0 0 12 "{\"fnum_scale\": 4, \"knudsen\": 0.5, \"a_max\": 64, \"fd_steps\": [], \"settings\": $SET}"
echo RBADAPT_DONE
