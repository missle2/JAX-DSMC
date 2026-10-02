#!/bin/bash
# serial-CPU A/B of the forward bench: baseline code (/tmp/trk_base = this tree with the
# wt_base library files) vs this tree, interleaved
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_PLATFORMS=cpu
for rep in 1 2; do
for tree in /tmp/trk_base /home/aakash/jax_dsmc/wt_tracker; do
  cd $tree
  PYTHONPATH=$tree taskset -c ${CORE:-7} python3 $tree/scratch/perf/bench_forward.py --bodies ${BODIES:-polygon levelset} --repeats 1 2>&1 | grep '^{' | sed "s|^|$(basename $tree) |"
done
done
