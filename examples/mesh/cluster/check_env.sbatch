#!/bin/bash
# 20-minute check on the testing QOS before anything is submitted: JAX version, devices,
# f64, a small level-set forward + gradient on the GPU.
#SBATCH --job-name=jaxdsmc-check
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=00:20:00
#SBATCH --output=examples/mesh/cluster/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source examples/mesh/cluster/env.sh
python3 -m examples.mesh.cluster.check_env
