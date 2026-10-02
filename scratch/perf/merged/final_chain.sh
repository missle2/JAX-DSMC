#!/bin/bash
# final verification + timings + profiles of the 2026-10-01 code (sequential: timings unloaded)
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/perf/merged/final; mkdir -p $O
PYTHONPATH=. XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $L cpu taskset -c 5 python3 scratch/perf/bench_forward.py > $O/bench_cpu.log 2>&1
mv scratch/perf/bench_forward_cpu.json scratch/perf/bench_forward_cpu_serial.json
PYTHONPATH=. $L gpu python3 scratch/perf/bench_forward.py > $O/bench_gpu.log 2>&1
$L cpu bash scratch/perf/profile_cpu.sh polygon "" > $O/prof_cpu.log 2>&1
rm -rf /tmp/hlo_poly
PYTHONPATH=. XLA_FLAGS="--xla_dump_to=/tmp/hlo_poly --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*_run.*" $L gpu python3 scratch/perf/dump_run.py polygon > $O/dump.log 2>&1
PYTHONPATH=. $L gpu python3 scratch/perf/profile_forward.py > $O/prof_gpu.log 2>&1
PYTHONPATH=. $L gpu python3 scratch/perf/regress.py --compare batch_fwd --save final > $O/regress.log 2>&1
$L gpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/JAX-DSMC --save final > $O/gate_fwd_run.log 2>&1
python3 scratch/perf/noise_gate.py forward --compare base final > $O/gate_fwd.txt 2>&1
$L cpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/JAX-DSMC --save final_cpu > $O/gate_fwd_cpu_run.log 2>&1
python3 scratch/perf/noise_gate.py forward --compare base final_cpu > $O/gate_fwd_cpu.txt 2>&1
$L gpu python3 scratch/perf/noise_gate.py gradient --code /home/aakash/jax_dsmc/JAX-DSMC --save final > $O/gate_grad_run.log 2>&1
python3 scratch/perf/noise_gate.py gradient --compare base final > $O/gate_grad.txt 2>&1
echo DONE > $O/done
