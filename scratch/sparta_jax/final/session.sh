#!/bin/bash
# one idle-machine timing session: SPARTA, sparta_jax (CPU-1, CPU-all, GPU cmdbuf off/on), main code
S=/home/aakash/jax_dsmc/JAX-DSMC-sparta
M=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=$S/scratch/sparta_jax/final
t0=$(date +%s)
while [ $(awk '{print ($1 < 1.0)}' /proc/loadavg) -eq 0 ] && [ $(( $(date +%s) - t0 )) -lt 1200 ]; do sleep 30; done
echo "start $(date) load $(cat /proc/loadavg)" > $O/session.log
res=$S/examples/sparta_jax/results
out=$res/sparta_timing_final.txt
echo "build ranks deck repeat loop_seconds moves particles loadavg1" > $out
cd /tmp/spbench
for r in 1 2 3 4 5; do for deck in in.circle_ar_bench in.circle_ar; do
  ~/sparta/sparta/build/src/spa_serial -var seed 12345 -in $deck -log log.s -screen none
  echo "serial 1 $deck $r $(awk '/^Loop time/{print $4}' log.s) $(awk '/^Particle moves/{print $4}' log.s) $(awk '/^Loop time/{print $(NF-1)}' log.s) $(cut -d' ' -f1 /proc/loadavg)" >> $out
  mpirun --oversubscribe -np 8 ~/sparta/sparta/build_mpi/src/spa_ -var seed 12345 -in $deck -log log.m -screen none
  echo "mpi 8 $deck $r $(awk '/^Loop time/{print $4}' log.m) $(awk '/^Particle moves/{print $4}' log.m) $(awk '/^Loop time/{print $(NF-1)}' log.m) $(cut -d' ' -f1 /proc/loadavg)" >> $out
done; done
cd $S
$L cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 examples/sparta_jax/bench.py --tag cpu_1 --repeats 5 >> $O/session.log 2>&1
$L cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 examples/sparta_jax/bench.py --tag cpu_1_tally --repeats 3 --set tally=1 >> $O/session.log 2>&1
$L cpu python3 examples/sparta_jax/bench.py --tag cpu_all --repeats 3 >> $O/session.log 2>&1
$L gpu python3 examples/sparta_jax/bench.py --tag gpu --repeats 5 --set sort=full >> $O/session.log 2>&1
flock /tmp/jax_gpu.lock env XLA_PYTHON_CLIENT_PREALLOCATE=false python3 examples/sparta_jax/bench.py --tag gpu_cmdbuf --repeats 5 --set sort=full >> $O/session.log 2>&1
bash scratch/sparta_jax/prof.sh cpu1_final >> $O/session.log 2>&1
cd $M
PYTHONPATH=. XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $L cpu taskset -c 3 python3 scratch/perf/bench_forward.py > $O/main_cpu1.log 2>&1
PYTHONPATH=. $L gpu python3 scratch/perf/bench_forward.py > $O/main_gpu.log 2>&1
$L cpu bash scratch/perf/profile_cpu.sh polygon _final > $O/main_prof.log 2>&1
echo "end $(date) load $(cat /proc/loadavg)" >> $O/session.log
echo DONE > $O/done
