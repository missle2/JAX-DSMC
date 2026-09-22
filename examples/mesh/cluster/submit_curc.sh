#!/bin/bash
# Submit the study to CURC Alpine, one right-sized SLURM job per line of a group:
#
#   cd JAX-DSMC-adjoint                    # on a login node
#   module load slurm/alpine
#   sbatch examples/mesh/cluster/check_env.sbatch          # first: environment check
#   bash examples/mesh/cluster/submit_curc.sh sphere       # 8 jobs x 1 GPU (one per seed)
#   bash examples/mesh/cluster/submit_curc.sh mode2        # 32 jobs x 1 GPU
#   bash examples/mesh/cluster/submit_curc.sh opt          # 8 jobs x {1,2,3} GPUs
#   bash examples/mesh/cluster/submit_curc.sh all
#
# Sizing (aa100 / ami100: 3 GPUs per node, 64 cores, 243 GB): 21 cores and 70 GB per GPU.
# Wall time: sphere 8 h/seed, mode2 4 h/seed, optimisation 12 h (S <= 2) / 24 h (S >= 4,
# the gpu-normal maximum); every job resumes from its artifact, so a job that hits its
# limit is simply resubmitted (`submit_curc.sh <group>` again: finished rows/iterates are
# skipped).  GPU_KIND=a100 (default, CUDA venv) -> aa100 / gpu:a100-40gb; GPU_KIND=mi100
# -> ami100 / gpu:mi100 (needs a jax >= 0.4.36 module, see env.sh).  GRES_TYPE overrides
# the gres name (a100-80gb, h200 with PART=ah200, ...).
set -euo pipefail
GROUPS_TO_RUN=("$@")
[ ${#GROUPS_TO_RUN[@]} -eq 0 ] && { echo "usage: $0 sphere|mode2|opt|all"; exit 1; }
[ "${GROUPS_TO_RUN[0]}" = "all" ] && GROUPS_TO_RUN=(sphere mode2 opt)
GPU_KIND="${GPU_KIND:-a100}"
if [ "${GPU_KIND}" = "mi100" ]; then PART="${PART:-ami100}"; GRES="${GRES_TYPE:-mi100}"; else PART="${PART:-aa100}"; GRES="${GRES_TYPE:-a100-40gb}"; fi
mkdir -p examples/mesh/cluster/logs
for group in "${GROUPS_TO_RUN[@]}"; do
  case "${group}" in
    sphere) HOURS=8 ;;
    mode2) HOURS=4 ;;
    opt) HOURS=24 ;;
    *) echo "unknown group ${group}"; exit 1 ;;
  esac
  mapfile -t NAMES < <(python3 -m examples.mesh.cluster.study --group "${group}" --names)
  mapfile -t CMDS < <(python3 -m examples.mesh.cluster.study --group "${group}" --list)
  mapfile -t GPUS < <(python3 -m examples.mesh.cluster.study --group "${group}" --gpus | awk '{print $2}')
  for i in "${!NAMES[@]}"; do
    n="${GPUS[$i]}"
    hours="${HOURS}"
    if [ "${group}" = "opt" ] && [ "${n}" -le 2 ]; then hours=12; fi
    sbatch --job-name="${group}_${NAMES[$i]}" \
      --partition="${PART}" --gres="gpu:${GRES}:${n}" \
      --cpus-per-task=$((21 * n)) --mem="$((70 * n))G" --time="${hours}:00:00" \
      --export=ALL,CMD="${CMDS[$i]}",GPU_KIND="${GPU_KIND}" \
      examples/mesh/cluster/job.sbatch
  done
done
