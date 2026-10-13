#!/bin/bash
# 20-minute check before anything is submitted: JAX version, devices, f64, a small
# level-set forward + gradient on the GPU.  Default: one A100 (proven 2026-09-21, 88 s).
# The 1-hour testing QOS works too, but Alpine gives no full A100 on it -- only a MIG
# slice: --qos=gpu-testing --gres=gpu:a100_3g.20gb:1.  Before using the H200s run this on
# them (--partition=ah200 --gres=gpu:h200:1): on 2026-09-22 every ah200 job landed on
# c3gpu-g5-u5, whose driver refuses cuInit (CUDA_ERROR_UNKNOWN).
#SBATCH --job-name=jaxdsmc-check
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb856_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=00:20:00
#SBATCH --output=examples/mesh/cluster/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=aakash.jain@colorado.edu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source examples/mesh/cluster/env.sh
python3 -m examples.mesh.cluster.check_env
