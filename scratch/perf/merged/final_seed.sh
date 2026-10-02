#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
until [ -f scratch/perf/merged/final/done ]; do sleep 30; done
PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 scratch/perf/profile_seed.py --tag final_seed > scratch/perf/merged/final/seed.log 2>&1
python3 scratch/perf/profile_report.py final_seed > scratch/perf/merged/final/report.log 2>&1
echo DONE > scratch/perf/merged/final/seed_done
