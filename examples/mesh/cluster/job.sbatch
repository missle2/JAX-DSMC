#!/bin/bash
# One study job on CURC Alpine.  Resources (--gres, --cpus-per-task, --mem, --time,
# --job-name) come from submit_curc.sh on the sbatch command line; the command comes in
# the environment variable CMD (--export=ALL,CMD=...).  Runs from the repository root.
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb856_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=examples/mesh/cluster/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=aakash.jain@colorado.edu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source examples/mesh/cluster/env.sh
echo "[$(date)] ${SLURM_JOB_NAME} on $(hostname): gpus=${SLURM_GPUS_ON_NODE:-?} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L 2>&1 | sed "s/^/  /" || true
# a multi-GPU job (optimize --workers N: one seed per GPU) must have all N GPUs on this node
# (--nodes=1 above; --gres counts per node): check what Slurm actually handed over
if [ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]; then
  echo "[$(date)] ${SLURM_JOB_NUM_NODES} nodes allocated, the workers need one: abort"; exit 1
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  n_visible=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  if [ "${n_visible}" != "${SLURM_GPUS_ON_NODE}" ]; then
    echo "[$(date)] ${n_visible} visible GPUs, ${SLURM_GPUS_ON_NODE} allocated: abort"; exit 1
  fi
fi
# GPU monitor (GPU_MONITOR seconds, 0 = off): per-GPU utilisation / memory and the processes
# on each GPU, beside the job's log -- a 2-GPU worker job shows two python processes, one per
# GPU UUID, both busy (examples/mesh/cluster/gpu_usage.py summarises the file)
GPU_LOG="examples/mesh/cluster/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.log"
if [ "${GPU_MONITOR:-60}" != "0" ] && command -v nvidia-smi > /dev/null; then
  (
    set +e  # a failed query must not end the monitor (the job runs under set -e)
    while true; do
      echo "# $(date +%FT%T)"
      nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader 2>&1 | sed "s/^/gpu, /"
      nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>&1 \
        | sed "s/^/app, /"
      sleep "${GPU_MONITOR:-60}"
    done
  ) > "${GPU_LOG}" 2>&1 &
  MONITOR_PID=$!
  trap 'kill "${MONITOR_PID}" 2> /dev/null || true' EXIT
  echo "GPU monitor every ${GPU_MONITOR:-60} s -> ${GPU_LOG}"
fi
# Preflight (seconds): a node whose driver refuses cuInit would otherwise run the whole
# job on the CPU.  A failure records the node in bad_nodes.txt, which submit_curc.sh
# passes to sbatch as --exclude, so the queue stops feeding the same broken node.
if ! python3 -m examples.mesh.cluster.check_env --quick; then
  echo "$(hostname)" >> examples/mesh/cluster/bad_nodes.txt
  echo "[$(date)] no usable GPU on $(hostname): recorded in bad_nodes.txt, resubmit"
  exit 1
fi
# the package must import from anywhere a job's script lives (env.sh's PYTHONPATH): fail
# here in seconds rather than at the job's own first import (the 2026-10-04 biasgate jobs)
if ! (cd /tmp && python3 -c "import jax_dsmc, examples.ensemble_mesh"); then
  echo "[$(date)] jax_dsmc does not import with PYTHONPATH=${PYTHONPATH:-}"
  exit 1
fi
echo "${CMD}"
eval "${CMD}"
echo "[$(date)] done"
