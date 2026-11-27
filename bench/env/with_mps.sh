#!/bin/bash
# Run a command with the CUDA MPS daemon on every node of the allocation (several MPI ranks per
# GPU share it through MPS instead of time-slicing):
#   bash bench/env/with_mps.sh <nodes> -- <command...>
set -uo pipefail
N="$1"; shift; [ "$1" = "--" ] && shift
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-${SLURM_JOB_ID:-$$}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log-${SLURM_JOB_ID:-$$}}"
srun -N "${N}" --ntasks-per-node=1 --overlap bash -c \
  'mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" && nvidia-cuda-mps-control -d' \
  || { echo "MPS did not start"; exit 1; }
"$@"; rc=$?
srun -N "${N}" --ntasks-per-node=1 --overlap bash -c 'echo quit | nvidia-cuda-mps-control' || true
exit "${rc}"
