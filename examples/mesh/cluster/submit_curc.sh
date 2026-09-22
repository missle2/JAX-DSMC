#!/bin/bash
# Submit the study to CURC Alpine, one right-sized SLURM job per line of a group:
#
#   cd JAX-DSMC-adjoint                    # on a login node
#   module load slurm/alpine
#   sbatch examples/mesh/cluster/check_env.sbatch          # first: environment check
#   bash examples/mesh/cluster/submit_curc.sh sphere       # 8 jobs x 1 GPU (one per seed)
#   bash examples/mesh/cluster/submit_curc.sh mode2        # 32 jobs x 1 GPU
#   bash examples/mesh/cluster/submit_curc.sh opt          # 8 jobs x {1,2,4} GPUs
#   bash examples/mesh/cluster/submit_curc.sh all
#
# Sizing per GPU = the node divided by its GPUs (CURC "Alpine hardware"): ah200 (4 H200,
# 128 cores, 1.5 TB) -> 32 cores, 128 GB (a quarter of the memory would be 384 GB; the
# jobs need < 20 GB); aa100 / ami100 (3 GPUs, 64 cores, 243 GB) -> 21 cores, 70 GB.
# Wall time: sphere 8 h/seed, mode2 4 h/seed, optimisation 12 h (S <= 2) / 24 h (S >= 4,
# the gpu-normal maximum); every job resumes from its artifact, so a job that hits its
# limit is simply resubmitted (`submit_curc.sh <group>` again: finished rows/iterates are
# skipped).  GPU_KIND=h200 (default, CUDA venv) -> ah200 / gpu:h200; GPU_KIND=a100 (CUDA
# venv) -> aa100 / gpu:a100-40gb; GPU_KIND=mi100 -> ami100 / gpu:mi100 (needs a jax >=
# 0.4.36 module, see env.sh).  PART / GRES_TYPE override the partition / gres name
# (GRES_TYPE=a100-80gb, ...); the partition's GPUs per node cap the GPUs of an opt job.
set -euo pipefail
GROUPS_TO_RUN=("$@")
[ ${#GROUPS_TO_RUN[@]} -eq 0 ] && { echo "usage: $0 sphere|mode2|opt|all"; exit 1; }
[ "${GROUPS_TO_RUN[0]}" = "all" ] && GROUPS_TO_RUN=(sphere mode2 opt)
GPU_KIND="${GPU_KIND:-h200}"
case "${GPU_KIND}" in
  h200) PART="${PART:-ah200}"; GRES="${GRES_TYPE:-h200}"; CORES=32; MEM=128; PER_NODE=4 ;;
  a100) PART="${PART:-aa100}"; GRES="${GRES_TYPE:-a100-40gb}"; CORES=21; MEM=70; PER_NODE=3 ;;
  mi100) PART="${PART:-ami100}"; GRES="${GRES_TYPE:-mi100}"; CORES=21; MEM=70; PER_NODE=3 ;;
  *) echo "unknown GPU_KIND ${GPU_KIND} (h200|a100|mi100)"; exit 1 ;;
esac
# study.py needs python >= 3.8 (annotations); the login node's /usr/bin/python3 is 3.6, so
# use the jobs' venv interpreter (PYTHON overrides).
PY="${PYTHON:-${JAX_VENV:-/projects/${USER}/jax-env}/bin/python3}"
[ -x "${PY}" ] || PY=python3
"${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 8))' || { echo "${PY} is $("${PY}" -V); need >= 3.8 (set PYTHON or JAX_VENV)"; exit 1; }
mkdir -p examples/mesh/cluster/logs
for group in "${GROUPS_TO_RUN[@]}"; do
  case "${group}" in
    sphere) HOURS=8 ;;
    mode2) HOURS=4 ;;
    opt) HOURS=24 ;;
    *) echo "unknown group ${group}"; exit 1 ;;
  esac
  STUDY=("${PY}" -m examples.mesh.cluster.study --group "${group}" --gpus-per-node "${PER_NODE}")
  mapfile -t NAMES < <("${STUDY[@]}" --names)
  mapfile -t CMDS < <("${STUDY[@]}" --list)
  mapfile -t GPUS < <("${STUDY[@]}" --gpus | awk '{print $2}')
  [ ${#NAMES[@]} -gt 0 ] && [ ${#NAMES[@]} -eq ${#CMDS[@]} ] || { echo "no jobs listed for ${group}"; exit 1; }
  for i in "${!NAMES[@]}"; do
    n="${GPUS[$i]}"
    hours="${HOURS}"
    if [ "${group}" = "opt" ] && [ "${n}" -le 2 ]; then hours=12; fi
    sbatch --job-name="${group}_${NAMES[$i]}" \
      --partition="${PART}" --gres="gpu:${GRES}:${n}" \
      --cpus-per-task=$((CORES * n)) --mem="$((MEM * n))G" --time="${hours}:00:00" \
      --export=ALL,CMD="${CMDS[$i]}",GPU_KIND="${GPU_KIND}" \
      examples/mesh/cluster/job.sbatch
  done
done
