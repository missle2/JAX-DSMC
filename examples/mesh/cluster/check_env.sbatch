#!/bin/bash
# 20-minute check before anything is submitted: JAX version, devices, f64, a small
# level-set forward + gradient on the GPU.  Runs on gpu-normal (the ah200 partition takes
# gpu-normal / gpu-long).  For the A100s the 1-hour testing QOS works, but only with a MIG
# slice (Alpine gives no full A100 on gpu-testing; verified 2026-09-21, 88 s):
#   --partition=aa100 --qos=gpu-testing --gres=gpu:a100_3g.20gb:1
#SBATCH --job-name=jaxdsmc-check
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:20:00
#SBATCH --output=examples/mesh/cluster/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source examples/mesh/cluster/env.sh
python3 -m examples.mesh.cluster.check_env
