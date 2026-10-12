#!/bin/bash
# The split per-cell backward on a complicated real geometry: optc S1-adam iterate 100
# (4.5 % of the flow in cells with R < 0.9, 24 cells with opposed normals), Kn 0.5,
# one-sided backward fits, fnum x4 on the 20 x 20 grid; seeds 0-2 paired + CRN FD of the
# uniform level-set shift.
R=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
cd $R
SET='[{"basis":"quadratic_full","k":32},{"basis":"quadratic_full","k":64},{"basis":"cell:quad","k":32},{"basis":"cell:quad:align0.9","k":32},{"basis":"cell:quad:adapt1.5:align0.9","k":32}]'
$L gpu env PYTHONPATH=. python3 $R/scratch/cellfit/geom_gate.py $R/scratch/cellfit/geom_S1a100.jsonl $R/examples/mesh/results/optimize/circle_coarse_S1_adam 100 0 3 "{\"settings\": $SET, \"fd_h\": [0.05, 0.1]}"
echo GEOM_DONE
