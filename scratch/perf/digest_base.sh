#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
python3 scratch/perf/forward_digest.py --save base_gpu
JAX_PLATFORMS=cpu taskset -c 3 python3 scratch/perf/forward_digest.py --save base_cpu
echo digest base done
