#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/perf/merged
$L gpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/JAX-DSMC --save merged > $O/gate_fwd_run.log 2>&1
python3 scratch/perf/noise_gate.py forward --compare base merged > $O/gate_fwd.txt 2>&1
$L gpu python3 scratch/perf/noise_gate.py gradient --code /home/aakash/jax_dsmc/JAX-DSMC --save merged > $O/gate_grad_run.log 2>&1
python3 scratch/perf/noise_gate.py gradient --compare base merged > $O/gate_grad.txt 2>&1
PYTHONPATH=. $L gpu python3 scratch/perf/regress.py --compare batch_fwd --save merged > $O/regress.log 2>&1
PYTHONPATH=. $L gpu python3 scratch/perf/bench_forward.py > $O/bench_gpu.log 2>&1
echo GPU_DONE >> $O/bench_gpu.log
