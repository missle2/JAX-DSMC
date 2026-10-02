#!/usr/bin/env bash
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC
export PYTHONPATH=.
JAX_PLATFORMS=cpu taskset -c 4-11 python3 -m pytest -q -p no:cacheprovider tests/test_mesh_levelset.py tests/test_mesh_cut.py > scratch/perf/test_batch_b1.log 2>&1
JAX_PLATFORMS=cpu taskset -c 4-11 python3 -m pytest -q -p no:cacheprovider tests/test_mesh_adjoint.py > scratch/perf/test_batch_b2.log 2>&1
JAX_PLATFORMS=cpu taskset -c 4-11 python3 -m pytest -q -p no:cacheprovider tests/test_ensemble_backward.py tests/test_ensemble_device.py > scratch/perf/test_batch_b3.log 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
python3 scratch/perf/bench_forward.py > scratch/perf/bench_gpu_batch.log 2>&1
cp scratch/perf/bench_forward_gpu.json scratch/perf/bench_forward_gpu_batch.json
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 python3 scratch/perf/bench_forward.py --repeats 1 > scratch/perf/bench_cpu_batch.log 2>&1
cp scratch/perf/bench_forward_cpu.json scratch/perf/bench_forward_cpu_serial_batch.json
echo after done
