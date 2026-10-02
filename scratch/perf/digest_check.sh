#!/usr/bin/env bash
# 1) the pre-batch snapshot reproduces the clean GPU baseline; 2) it records the CPU
# baseline; 3) the working tree against both
set -u
repo=/home/aakash/jax_dsmc/JAX-DSMC
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
cd /tmp/prebatch
PYTHONPATH=/tmp/prebatch python3 scratch/perf/forward_digest.py --compare base_gpu
PYTHONPATH=/tmp/prebatch JAX_PLATFORMS=cpu taskset -c 3 python3 scratch/perf/forward_digest.py --save base_cpu
cp scratch/perf/digest/base_cpu_*.npz $repo/scratch/perf/digest/
cd $repo
PYTHONPATH=. python3 scratch/perf/forward_digest.py --compare base_gpu
PYTHONPATH=. JAX_PLATFORMS=cpu taskset -c 3 python3 scratch/perf/forward_digest.py --compare base_cpu
echo check done
