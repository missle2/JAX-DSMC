#!/usr/bin/env bash
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= PYTHONPATH=.
python3 scratch/perf/forward_digest.py --compare base_gpu
JAX_PLATFORMS=cpu taskset -c 3 python3 scratch/perf/forward_digest.py --compare base_cpu
python3 scratch/perf/compile_time.py
echo check2 done
