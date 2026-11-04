#!/bin/bash
# Local smoke of optimize --constraints project / mma (2026-11-03): the diffuse M 10 circle at
# 1/32 of its particles, 150 + 40 steps, 2 iterates (the second has the lagged heat gradient),
# the lift floor high enough that the lift constraint is violated (active rows + restoration).
# Run from the repo root with PYTHONPATH=.; tags smoke_cons_<mode> under results/optimize.
set -e
PY=/home/aakash/envs/jax-env/bin/python
COMMON="--case circle_long --levelset-h 0.0625 --levelset-samples 16 --knudsen 0.5 --mach 10
 --dt 2.5e-5 --a-max 256 --boundary free --fnum-scale 32 --warmup 150 --steps 40 --window 20
 --restart-steps 0 --acc 1 --t-wall 273.15 --silhouette-band 0.03 --backward-normals true
 --grazing 0.01 --timing-band 10 --heat-patch 0.2 --objective drag_lift --lift-min 0.3
 --heat-cap 1.0 --heat-cap-relative 1 --area-mode floor --area-floor 0.8 --filter helmholtz
 --reinit-every 0 --variant cell_along_pool_plane --iterations 2 --substeps 4 --partners 8
 --step 0.02 --save-gradient 1"
$PY -m examples.mesh.optimize $COMMON --tag smoke_cons_project --optimizer ox_adam --constraints project
$PY -m examples.mesh.optimize $COMMON --tag smoke_cons_mma --optimizer mma_s --constraints mma
