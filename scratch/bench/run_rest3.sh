#!/bin/bash
# R2048 measured at ~80 min/seed (reconstruction is O(queries x retained)); 8 seeds is 9 h.
# Re-run it with 2 seeds after the main chain (pid $1) is done, then rebuild the reports.
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "=== p1_ballistic_R2048 (2 seeds) $(date +%T)"
rm -f examples/p3_replication/results/p1_ballistic_R2048.json
python3 -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000 --retained 2048 --source-samples 1995 --partners 8 --neighbors 32 --seeds 2 --fd-step 0.1 --output examples/p3_replication/results/p1_ballistic_R2048.json > scratch/bench/p3/p1_ballistic_R2048.log 2>&1; echo "    exit=$? $(date +%T)"
rm -f examples/p3_replication/results/p1_ballistic_P4.json examples/p3_replication/results/p1_ballistic_K16.json
python3 -m examples.p3_replication.report > scratch/bench/p3_report.log 2>&1
echo "=== DONE-R2048 $(date +%T)"
