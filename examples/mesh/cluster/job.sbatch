#!/bin/bash
# One study job on CURC Alpine.  Resources (--gres, --cpus-per-task, --mem, --time,
# --job-name) come from submit_curc.sh on the sbatch command line; the command comes in
# the environment variable CMD (--export=ALL,CMD=...).  Runs from the repository root.
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb-general
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
# Preflight (seconds): a node whose driver refuses cuInit would otherwise run the whole
# job on the CPU.  A failure records the node in bad_nodes.txt, which submit_curc.sh
# passes to sbatch as --exclude, so the queue stops feeding the same broken node.
if ! python3 -m examples.mesh.cluster.check_env --quick; then
  echo "$(hostname)" >> examples/mesh/cluster/bad_nodes.txt
  echo "[$(date)] no usable GPU on $(hostname): recorded in bad_nodes.txt, resubmit"
  exit 1
fi
echo "${CMD}"
eval "${CMD}"
echo "[$(date)] done"
