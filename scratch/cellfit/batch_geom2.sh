#!/bin/bash
# optc S1-adam iterate 100, Kn 0.5: the order-guarded split per-cell fit (everywhere, and
# with its complex cells two-sided) against the two-sided fits -- the one-sided per-query
# fit (and so the per-query fallback) is unstable there (batch_geomdiag.sh).
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=$R/examples/mesh/results/optimize/circle_coarse_S1_adam
cd $R
SET='[{"basis":"cell:quad:adapt1.5","k":32},{"basis":"cell:quad:adapt1.5:align0.9:both","k":32},{"basis":"cell:quad:adapt1.5","k":32,"sided":false}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/cellfit/geom_gate.py $R/scratch/cellfit/geom_diag2.jsonl $O 100 1 3 "{\"settings\": $SET, \"fd_h\": [0.05]}"
SET='[{"basis":"quadratic_full","k":32,"sided":false},{"basis":"cell:quad","k":32,"sided":false}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/cellfit/geom_gate.py $R/scratch/cellfit/geom_diag2.jsonl $O 100 2 3 "{\"settings\": $SET, \"fd_h\": []}"
echo GEOM2_DONE
