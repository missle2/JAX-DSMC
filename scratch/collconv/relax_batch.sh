#!/bin/bash
# Path 1b: grad_v lambda gate (0-D relaxation, reparameterised initial condition).
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
$L cpu env PYTHONPATH=. python3 $R/scratch/collconv/relax_ic.py $R/scratch/collconv/relax_tx_M10.jsonl 1 24 '{"virtual": [[4, 8], [2, 4]], "neighbors": [32]}'
$L cpu env PYTHONPATH=. python3 $R/scratch/collconv/relax_ic.py $R/scratch/collconv/relax_tx_M40.jsonl 0 24 '{"steps": 40, "virtual": [[4, 8], [2, 4]], "neighbors": [32]}'
$L cpu env PYTHONPATH=. python3 $R/scratch/collconv/relax_ic.py $R/scratch/collconv/relax_ux_M40.jsonl 0 24 '{"control": "ux", "steps": 40, "virtual": [[4, 8]], "neighbors": [32]}'
echo RELAX_BATCH_DONE
