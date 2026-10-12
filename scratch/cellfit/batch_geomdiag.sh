#!/bin/bash
# Why is the per-seed coefficient gradient on optc S1-adam iterate 100 (Kn 0.5) so noisy
# that g . 1 (+0.2 .. +1.3 across backward fits) swamps its FD (-0.007)?  Control: the
# circle (iterate 0) at the same settings; and two-sided vs one-sided backwards.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=$R/examples/mesh/results/optimize/circle_coarse_S1_adam
cd $R
SET='[{"basis":"quadratic_full","k":32},{"basis":"quadratic_full","k":32,"sided":false},{"basis":"cell:quad","k":32,"sided":false}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/cellfit/geom_gate.py $R/scratch/cellfit/geom_diag.jsonl $O 0 0 2 "{\"settings\": $SET, \"fd_h\": [0.05]}"
SET='[{"basis":"quadratic_full","k":32,"sided":false},{"basis":"cell:quad","k":32,"sided":false},{"basis":"quadratic_full","k":32},{"basis":"cell:quad:align0.9","k":32}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/cellfit/geom_gate.py $R/scratch/cellfit/geom_diag.jsonl $O 100 1 2 "{\"settings\": $SET, \"fd_h\": [0.05]}"
echo GEOMDIAG_DONE
