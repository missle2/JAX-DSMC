#!/bin/bash
# idle-machine re-timing: SPARTA serial / MPI x8 (both decks) and sparta_jax CPU-1 / CPU-all
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
res=examples/sparta_jax/results
t0=$(date +%s)
# wait for an idle machine (1-min load < 1.0), at most 60 min
while [ $(awk '{print ($1 < 1.0)}' /proc/loadavg) -eq 0 ] && [ $(( $(date +%s) - t0 )) -lt 3600 ]; do sleep 30; done
echo "start $(date) load $(cat /proc/loadavg)"
out=$res/sparta_timing_idle.txt
echo "build ranks deck repeat loop_seconds moves particles loadavg1" > $out
cd /tmp/spbench
for r in 1 2 3 4 5; do for deck in in.circle_ar_bench in.circle_ar; do
  ~/sparta/sparta/build/src/spa_serial -var seed 12345 -in $deck -log log.s -screen none
  echo "serial 1 $deck $r $(awk '/^Loop time/{print $4}' log.s) $(awk '/^Particle moves/{print $4}' log.s) $(awk '/^Loop time/{print $(NF-1)}' log.s) $(cut -d' ' -f1 /proc/loadavg)" >> /home/aakash/jax_dsmc/JAX-DSMC-sparta/$out
  mpirun --oversubscribe -np 8 ~/sparta/sparta/build_mpi/src/spa_ -var seed 12345 -in $deck -log log.m -screen none
  echo "mpi 8 $deck $r $(awk '/^Loop time/{print $4}' log.m) $(awk '/^Particle moves/{print $4}' log.m) $(awk '/^Loop time/{print $(NF-1)}' log.m) $(cut -d' ' -f1 /proc/loadavg)" >> /home/aakash/jax_dsmc/JAX-DSMC-sparta/$out
done; done
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
echo "load before jax $(cat /proc/loadavg)"
/home/aakash/jax_dsmc/run_locked.sh cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 examples/sparta_jax/bench.py --tag cpu_1 --repeats 5
/home/aakash/jax_dsmc/run_locked.sh cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 examples/sparta_jax/bench.py --tag cpu_1_fullsort --repeats 3 --set sort=full
/home/aakash/jax_dsmc/run_locked.sh cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 examples/sparta_jax/bench.py --tag cpu_1_tally --repeats 3 --set tally=1
/home/aakash/jax_dsmc/run_locked.sh cpu python3 examples/sparta_jax/bench.py --tag cpu_all --repeats 3
bash scratch/sparta_jax/prof.sh cpu1
echo "end $(date) load $(cat /proc/loadavg)"
