#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
B="--dim 1 --seeds 2 --mach 2.0 --settle-mfp 25 --sample-mfp 40 --domain-mfp 320 --sample-stride 4 --a-max 256"
# baseline
python3 -u scratch/bird/run_case.py --out scratch/bird/sw_base  $B --cells-per-mfp 4 --npc 120 > scratch/bird/sw_base.log 2>&1
# cell size
python3 -u scratch/bird/run_case.py --out scratch/bird/sw_cpm2  $B --cells-per-mfp 2 --npc 120 > scratch/bird/sw_cpm2.log 2>&1
python3 -u scratch/bird/run_case.py --out scratch/bird/sw_cpm8  $B --cells-per-mfp 8 --npc 120 > scratch/bird/sw_cpm8.log 2>&1
# particles per cell (noise -> max-gradient bias)
python3 -u scratch/bird/run_case.py --out scratch/bird/sw_npc30 $B --cells-per-mfp 4 --npc 30  > scratch/bird/sw_npc30.log 2>&1
python3 -u scratch/bird/run_case.py --out scratch/bird/sw_npc480 $B --cells-per-mfp 4 --npc 480 > scratch/bird/sw_npc480.log 2>&1
echo DONE > scratch/bird/sweep.done
