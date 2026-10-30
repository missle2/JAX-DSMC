#!/bin/bash
# sparta_jax on 1..16 hardware threads (taskset), XLA's intra-op (Eigen) threading on, and the
# whole machine with it off: is the all-cores slowdown the threading or the core count?
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
L=/home/aakash/jax_dsmc/run_locked.sh
for n in 1 2 4 8 16; do
  $L cpu taskset -c 0-$((n - 1)) python3 examples/sparta_jax/bench.py --tag cpu_threads_$n --repeats 3 | grep '"seconds"'
done
$L cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false python3 examples/sparta_jax/bench.py --tag cpu_all_eigen_off --repeats 3 | grep '"seconds"'
$L cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 examples/sparta_jax/bench.py --tag cpu_1_recheck --repeats 3 | grep '"seconds"'
