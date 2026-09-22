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
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source examples/mesh/cluster/env.sh
echo "[$(date)] ${SLURM_JOB_NAME} on $(hostname): gpus=${SLURM_GPUS_ON_NODE:-?} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "${CMD}"
eval "${CMD}"
echo "[$(date)] done"
