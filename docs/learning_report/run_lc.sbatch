#!/bin/bash
# One job of reproduce.sh on a generic SLURM GPU cluster; the command arrives in $CMD
# (sbatch --export=ALL,CMD="..."). Edit the #SBATCH lines for your site (account, partition,
# GPU type). Every job fits one GPU with >= 24 GB, except the two fine cylinders (>= 48 GB:
# pass e.g. SBATCH_EXTRA="--gres=gpu:a100_80gb:1" to reproduce.sh for those).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
##SBATCH --account=<your allocation>
##SBATCH --partition=<your gpu partition>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source "${JAX_VENV:-${HOME}/jax-env}/bin/activate"
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
echo "[$(date)] ${SLURM_JOB_NAME} on $(hostname)"
nvidia-smi -L || true
eval "${CMD}"
echo "[$(date)] ${SLURM_JOB_NAME} done"
