#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/perf/merged
$L gpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/JAX-DSMC --save merged_tally > $O/gate_tally_run.log 2>&1
python3 scratch/perf/noise_gate.py forward --compare base merged_tally > $O/gate_tally.txt 2>&1
PYTHONPATH=. XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $L cpu taskset -c 5 python3 scratch/perf/bench_forward.py --bodies polygon > $O/bench_cpu_tally.log 2>&1
echo DONE >> $O/bench_cpu_tally.log
