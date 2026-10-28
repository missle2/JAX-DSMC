#!/bin/bash
# the lsfilter / lsfilter3d movies (2026-10-27), one process at a time
cd /home/aakash/jax_dsmc/JAX-DSMC
export PYTHONPATH=.
for m in lsf_m10 lsf_kn lsf_knld lsf_m10ld; do
  /home/aakash/jax_dsmc/run_locked.sh cpu /home/aakash/envs/jax-env/bin/python scratch/ldopt/movies.py $m
done
/home/aakash/jax_dsmc/run_locked.sh cpu /home/aakash/envs/jax-env/bin/python scratch/sph3d/movie3d.py \
  drag_hz0 ld_hz0
