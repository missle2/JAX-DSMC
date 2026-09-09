#!/bin/bash
# Matched cross-dimension Bird normal shock.  Same x resolution, same npc, same
# settle/sample distance in mean free paths -- only D changes.
set -x
cd /home/aakash/jax_dsmc/JAX-DSMC
COMMON="--mach 2.0 --settle-mfp 25 --sample-mfp 40 --domain-mfp 320 --cells-per-mfp 4 --npc 120 --a-max 160 --sample-stride 4 --transverse-cells 3"
python3 -u scratch/bird/run_case.py --out scratch/bird/m2_d1 --dim 1 --seeds 2 $COMMON > scratch/bird/m2_d1.log 2>&1
python3 -u scratch/bird/run_case.py --out scratch/bird/m2_d2 --dim 2 --seeds 1 $COMMON > scratch/bird/m2_d2.log 2>&1
python3 -u scratch/bird/run_case.py --out scratch/bird/m2_d3 --dim 3 --seeds 1 $COMMON > scratch/bird/m2_d3.log 2>&1
echo DONE > scratch/bird/queue_dims.done
