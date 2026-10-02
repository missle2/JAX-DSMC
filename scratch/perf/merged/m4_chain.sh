#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/perf/merged
PYTHONPATH=. XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $L cpu taskset -c 5 python3 scratch/perf/bench_forward.py > $O/m4_bench_cpu.log 2>&1
PYTHONPATH=. $L gpu python3 scratch/perf/bench_forward.py > $O/m4_bench_gpu.log 2>&1
$L gpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/JAX-DSMC --save merged4 > $O/m4_fwd_run.log 2>&1
python3 scratch/perf/noise_gate.py forward --compare base merged4 > $O/m4_fwd.txt 2>&1
$L gpu python3 scratch/perf/noise_gate.py gradient --code /home/aakash/jax_dsmc/JAX-DSMC --save merged4 > $O/m4_grad_run.log 2>&1
python3 scratch/perf/noise_gate.py gradient --compare base merged4 > $O/m4_grad.txt 2>&1
echo DONE > $O/m4_gpu_done
