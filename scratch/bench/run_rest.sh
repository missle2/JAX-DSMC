#!/bin/bash
# Resume after the session restart: finish inlet2_temp, then the P3 chain, then the report.
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
stage() { echo "=== $1  $(date '+%F %T')"; }
stage bench-inlet2_temp-resume; python3 -m examples.gradient_bench --study inlet2_temp --seeds-scale 0.75 >> scratch/bench/bench_inlet2_temp.log 2>&1; echo "    exit=$?"
stage P3; SEEDS=16 bash scratch/bench/run_p3.sh > scratch/bench/P3_chain.log 2>&1; echo "    exit=$?"
stage report; python3 -m examples.gradient_bench.report > scratch/bench/bench_report.log 2>&1; python3 -m examples.p3_replication.report > scratch/bench/p3_report.log 2>&1; echo "    exit=$?"
stage DONE
