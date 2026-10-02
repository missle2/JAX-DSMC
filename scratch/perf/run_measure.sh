#!/usr/bin/env bash
# the measurement chain after the speed-ups: GPU forward (level set, fnum 1), the GPU profile
# of one optimiser seed, the CPU forward (both bodies); sequential, one JAX process at a time
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
python3 scratch/perf/bench_forward.py --bodies levelset > scratch/perf/bench_gpu_levelset.log 2>&1
python3 scratch/perf/profile_seed.py --tag gpu_after > scratch/perf/profile_gpu_after.log 2>&1
JAX_PLATFORMS=cpu python3 scratch/perf/bench_forward.py --repeats 1 > scratch/perf/bench_cpu.log 2>&1
echo chain done
