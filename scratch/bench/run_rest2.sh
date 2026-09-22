#!/bin/bash
# Trimmed continuation: 4 min/seed measured for the N4k JAX inlet job, so the remaining
# P1 jobs run at 8 seeds and N64k is dropped (est. >1 h/seed).  Waits for the running
# N4k job (pid $1) first.
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
stage() { echo "=== $1  $(date '+%F %T')"; }
while kill -0 "$1" 2>/dev/null; do sleep 30; done
stage P3-trimmed
sed -i 's/^run p1_ballistic_N64k/: skip p1_ballistic_N64k/' scratch/bench/run_p3.sh
SEEDS=8 bash scratch/bench/run_p3.sh >> scratch/bench/P3_chain.log 2>&1; echo "    exit=$?"
stage report; python3 -m examples.gradient_bench.report > scratch/bench/bench_report.log 2>&1; python3 -m examples.p3_replication.report > scratch/bench/p3_report.log 2>&1; echo "    exit=$?"
stage DONE
