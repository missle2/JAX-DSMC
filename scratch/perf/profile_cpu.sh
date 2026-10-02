#!/bin/bash
# usage: profile_cpu.sh BODY TAG   (serial CPU per-op profile, fresh HLO dump)
cd /home/aakash/jax_dsmc/JAX-DSMC
D=/tmp/hlo_cpu_$1$2
rm -rf $D
HLO_DUMP=$D JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_dump_to=$D --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*_run.*" PYTHONPATH=. taskset -c 3 python3 scratch/perf/profile_forward_cpu.py $1 $2
