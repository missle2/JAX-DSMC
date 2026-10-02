#!/bin/bash
# usage: prof_tree.sh TREE BODY TAG  -- profile_forward_cpu.py (this tree's copy) against TREE's jax_dsmc
TREE=$1; BODY=$2; TAG=$3
cd $TREE
D=/tmp/hlo_cpu_$BODY$TAG
rm -rf $D
HLO_DUMP=$D JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_dump_to=$D --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*_run.*" PYTHONPATH=$TREE taskset -c ${CORE:-3} python3 /home/aakash/jax_dsmc/wt_misc/scratch/perf/profile_forward_cpu.py $BODY $TAG
