#!/bin/bash
# serial-CPU A/B of the forward bench: baseline code (/tmp/trk_base = this tree with the
# wt_base library files) vs this tree, interleaved
# GPU flags come from run_locked.sh gpu
for rep in 1 2; do
for tree in /tmp/trk_base /home/aakash/jax_dsmc/wt_tracker; do
  cd $tree
  PYTHONPATH=$tree python3 $tree/scratch/perf/bench_forward.py --bodies ${BODIES:-polygon levelset} --repeats 1 2>&1 | grep '^{' | sed "s|^|$(basename $tree) |"
done
done
