#!/bin/bash
# Submit the study to CURC Alpine, one right-sized SLURM job per line of a group:
#
#   cd JAX-DSMC-adjoint                    # on a login node
#   module load slurm/alpine
#   sbatch examples/mesh/cluster/check_env.sbatch          # first: environment check
#   bash examples/mesh/cluster/submit_curc.sh sphere       # 8 jobs x 1 GPU (one per seed)
#   bash examples/mesh/cluster/submit_curc.sh mode2        # 32 jobs x 1 GPU
#   bash examples/mesh/cluster/submit_curc.sh opt          # 8 jobs x {1,2,3} GPUs (h200: {1,2,4})
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh optc   # coarser circle, fits rtx
#   bash examples/mesh/cluster/submit_curc.sh all
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh check   # check_env on that kind
#   GPU_KIND=h200 bash examples/mesh/cluster/submit_curc.sh preflight optckn optc2 squashkn2
#       # one job (PREFLIGHT_HOURS, default 2: rtx compiles slowly) on two GPUs of that kind: preflight.sh of each group on the GPUs
#       # (PREFLIGHT_GPU=1), the optimisers' toys with a two-worker pool
#
# Sizing per GPU = the node divided by its GPUs (CURC "Alpine hardware"): ah200 (4 H200,
# 128 cores, 1.5 TB) -> 32 cores, 128 GB (a quarter of the memory would be 384 GB; the
# jobs need < 20 GB); aa100 / ami100 (3 GPUs, 64 cores, 243 GB) -> 21 cores, 70 GB.
# Wall time: sphere 8 h/seed, mode2 4 h/seed, optimisation 12 h (S <= 2) / 24 h (S >= 4,
# the gpu-normal maximum; optc always 24 h); every job resumes from its artifact, so a job that hits its
# limit is simply resubmitted (`submit_curc.sh <group>` again: finished rows/iterates are
# skipped).  GPU_KIND=a100 (default, CUDA venv) -> aa100 / gpu:a100-40gb; GPU_KIND=h200
# -> ah200 / a 2g.35gb MIG slice of an H200 (GPU_KIND=h200_full: the whole card, gpu:h200,
# 32 cores, 128 GB -- the 3-D runs) (faster, but run check_env there first: on 2026-09-22 the ah200
# jobs all landed on one node whose driver refuses cuInit); GPU_KIND=mi100 -> ami100 /
# gpu:mi100 (needs a jax >= 0.4.36 module, see env.sh).  Nodes listed in bad_nodes.txt
# (written by a job's preflight) and in EXCLUDE_NODES are excluded from every submission.
# GPU_KIND=rtx -> the RTX PRO 6000 nodes: partition, gres name and per-GPU cores/memory are
# read from `sinfo` (the first partition whose gres mentions "rtx"; RTX_MATCH overrides the
# pattern), so nothing about those nodes is hard-coded.  They are Blackwell (sm_120): run
# `check` on them first.  The group `check` submits check_env.sbatch on the chosen kind.
# The nodes are MIG-partitioned (a 1g.24gb slice = a quarter card) and gpu-normal allows one
# slice per job, so rtx jobs get 1 GPU.  Jobs whose name is already in squeue are skipped.
# ONLY=<regex> submits just the group's jobs whose name matches (e.g. the biasgate group:
#   GPU_KIND=rtx  ONLY='^kn_(seed|fd)' bash examples/mesh/cluster/submit_curc.sh biasgate
#   GPU_KIND=h200 ONLY='hor'           bash examples/mesh/cluster/submit_curc.sh biasgate).
# Before submitting a group, run its local preflight (examples/mesh/cluster/preflight.sh
# <group>): every job kind at toy size, as job.sbatch runs it.
# QOS overrides the QOS of job.sbatch / check_env.sbatch (gpu-normal), e.g. for a new
# allocation's priority QOS.  PART / GRES_TYPE override the partition / gres name
# (GRES_TYPE=a100-80gb, ...); the partition's GPUs per node cap the GPUs of an opt job.
set -euo pipefail
GROUPS_TO_RUN=("$@")
[ ${#GROUPS_TO_RUN[@]} -eq 0 ] && { echo "usage: $0 sphere|mode2|opt|optc|squash|squashkn|biasgate|cellgate|gapgate|thingate|guardgate|shapeprof|speedprof|boxcheck|opt3dcheck|optrules3|adjbias|optrules|optrules2|optrules1seed|optax1seed|optcm|optct|optckn|optc2|optab|optm|squashkn2|all|check|preflight <group>..."; exit 1; }
[ "${GROUPS_TO_RUN[0]}" = "all" ] && GROUPS_TO_RUN=(sphere mode2 opt)
[ ${#GROUPS_TO_RUN[@]} -eq 1 ] && [ "${GROUPS_TO_RUN[0]}" = "check" ] && CHECK_ONLY=1
[ "${GROUPS_TO_RUN[0]}" = "preflight" ] && PREFLIGHT_GROUPS=("${GROUPS_TO_RUN[@]:1}")
QOS_ARG=()
if [ -n "${QOS:-}" ]; then QOS_ARG=(--qos="${QOS}"); fi
GPU_KIND="${GPU_KIND:-a100}"
case "${GPU_KIND}" in
  h200) PART="${PART:-ah200}"; GRES="${GRES_TYPE:-h200_2g.35gb}"; CORES=10; MEM=40; PER_NODE=16 ;;
  # a whole H200 (141 GB), not a 2g.35gb MIG slice: the 3-D sphere ran out of memory on the
  # slice (opt3dcheck, 2026-10-24).  Counts against the same 4-concurrent H200 cap
  h200_full) PART="${PART:-ah200}"; GRES="${GRES_TYPE:-h200}"; CORES=32; MEM=128; PER_NODE=4 ;;
  a100) PART="${PART:-aa100}"; GRES="${GRES_TYPE:-a100-40gb}"; CORES=21; MEM=70; PER_NODE=3 ;;
  mi100) PART="${PART:-ami100}"; GRES="${GRES_TYPE:-mi100}"; CORES=21; MEM=70; PER_NODE=3 ;;
  rtx)
    # one line per node: "partition cpus memory_MB gres", e.g. "atesting 64 500000 gpu:rtx_pro_6000:4(S:0-1)"
    LINE="$(sinfo -h -N -o "%P %c %m %G" | grep -i "${RTX_MATCH:-rtx}" | head -1 || true)"
    [ -n "${LINE}" ] || { echo "no partition with an rtx gres in sinfo (set PART/GRES_TYPE or RTX_MATCH)"; exit 1; }
    read -r P_SINFO NCPU MEM_MB G_SINFO <<< "${LINE}"
    PART="${PART:-${P_SINFO%\*}}"
    GRES="${GRES_TYPE:-$(echo "${G_SINFO}" | cut -d: -f2)}"
    PER_NODE="$(echo "${G_SINFO}" | cut -d: -f3 | cut -d'(' -f1)"
    CORES=$((NCPU / PER_NODE)); MEM=$((MEM_MB / 1024 * 9 / 10 / PER_NODE))
    echo "rtx: partition ${PART}, gres gpu:${GRES}, ${PER_NODE}/node -> ${CORES} cores, ${MEM} GB per GPU"
    # gpu-normal allows ONE rtx MIG slice per job (sbatch Error 8, 2026-09-23): an opt job's
    # S seeds then run one after another on its slice (optimize.py --workers 1)
    PER_NODE=1 ;;
  *) echo "unknown GPU_KIND ${GPU_KIND} (h200|h200_full|a100|rtx|mi100)"; exit 1 ;;
esac
# a venv active in the submitting shell would ride into every job (--export=ALL) and break
# env.sh's `module purge` (2026-10-22, see env.sh): submit without it
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "dropping the active venv ${VIRTUAL_ENV} from the jobs' environment"
  PATH="$(echo "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONHOME
fi
# study.py needs python >= 3.8 (annotations); the login node's /usr/bin/python3 is 3.6, so
# use the jobs' venv interpreter (PYTHON overrides).
PY="${PYTHON:-${JAX_VENV:-/projects/${USER}/jax-env}/bin/python3}"
[ -x "${PY}" ] || PY=python3
"${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 8))' || { echo "${PY} is $("${PY}" -V); need >= 3.8 (set PYTHON or JAX_VENV)"; exit 1; }
mkdir -p examples/mesh/cluster/logs
if [ -n "${CHECK_ONLY:-}" ]; then
  sbatch "${QOS_ARG[@]}" --partition="${PART}" --gres="gpu:${GRES}:1" \
    --export=ALL,GPU_KIND="${GPU_KIND}" examples/mesh/cluster/check_env.sbatch
  exit 0
fi
if [ -n "${PREFLIGHT_GROUPS+x}" ]; then
  [ ${#PREFLIGHT_GROUPS[@]} -gt 0 ] || { echo "usage: $0 preflight <group>..."; exit 1; }
  n=$((PER_NODE < 2 ? PER_NODE : 2))
  # no commas in CMD: sbatch --export splits on them
  CMD="rc=0; for g in ${PREFLIGHT_GROUPS[*]}; do PREFLIGHT_GPU=1 PREFLIGHT_WORKERS=${n} bash examples/mesh/cluster/preflight.sh \${g} || rc=1; done; exit \${rc}"
  sbatch --job-name="preflight_${GPU_KIND}" "${QOS_ARG[@]}" \
    --partition="${PART}" --gres="gpu:${GRES}:${n}" \
    --cpus-per-task=$((CORES * n)) --mem="$((MEM * n))G" --time="0${PREFLIGHT_HOURS:-2}:00:00" \
    --export=ALL,CMD="${CMD}",GPU_KIND="${GPU_KIND}" \
    examples/mesh/cluster/job.sbatch
  exit 0
fi
# nodes a preflight found unusable, plus anything the caller names
BAD_FILE=examples/mesh/cluster/bad_nodes.txt
EXCLUDE="$(cat "${BAD_FILE}" 2>/dev/null | sort -u | paste -sd, -)"
if [ -n "${EXCLUDE_NODES:-}" ]; then EXCLUDE="${EXCLUDE:+${EXCLUDE},}${EXCLUDE_NODES}"; fi
EXCLUDE_ARG=()
if [ -n "${EXCLUDE}" ]; then EXCLUDE_ARG=(--exclude="${EXCLUDE}"); echo "excluding ${EXCLUDE}"; fi
for group in "${GROUPS_TO_RUN[@]}"; do
  case "${group}" in
    sphere) HOURS=8 ;;
    mode2) HOURS=4 ;;
    opt|optc) HOURS=24 ;;
    squash|squashkn|squashkn2|optckn|optc2|optab|optm|optrules|optrules2|optrules1seed|optax1seed|optcm|optct|optrules3) HOURS=24 ;;
    biasgate|cellgate|gapgate|thingate|guardgate|shapeprof|speedprof|boxcheck|opt3dcheck|adjbias) HOURS=12 ;;
    *) echo "unknown group ${group}"; exit 1 ;;
  esac
  # the finer circle (240 k slots x 300 captured steps) peaks above a 24 GB MIG slice: the
  # captured forward alone is 18.7 GB after rematerialisation and ran out of memory on every
  # rtx opt job (2026-09-23).  The sphere (50 k slots) fits.  optrules2 / optrules1seed
  # (2026-10-20): the optm free-box problems, ~18 GB on the H200 slices per the user's Slurm
  # reports -- allowed on rtx; an out-of-memory job fails in its first iterate and resumes
  # elsewhere (a100 / h200).
  if [ "${GPU_KIND}" = "rtx" ] && [ "${group}" != "sphere" ] && [ "${group}" != "optc" ] \
      && [ "${group}" != "squash" ] && [ "${group}" != "squashkn" ] \
      && [ "${group}" != "biasgate" ] && [ "${group}" != "cellgate" ] \
      && [ "${group}" != "gapgate" ] && [ "${group}" != "thingate" ] \
      && [ "${group}" != "guardgate" ] && [ "${group}" != "shapeprof" ] \
      && [ "${group}" != "speedprof" ] && [ "${group}" != "boxcheck" ] \
      && [ "${group}" != "opt3dcheck" ] && [ "${group}" != "adjbias" ] \
      && [ "${group}" != "optckn" ] && [ "${group}" != "optc2" ] && [ "${group}" != "optab" ] \
      && [ "${group}" != "squashkn2" ] && [ "${group}" != "optrules2" ] \
      && [ "${group}" != "optrules1seed" ] && [ "${group}" != "optax1seed" ] \
      && [ "${group}" != "optcm" ] && [ "${group}" != "optct" ] \
      && [ "${group}" != "optrules3" ]; then
    echo "skip ${group}: the finer circle needs > 24 GB, more than an rtx MIG slice (use optc or a100/h200)"
    continue
  fi
  STUDY=("${PY}" -m examples.mesh.cluster.study --group "${group}" --gpus-per-node "${PER_NODE}")
  mapfile -t NAMES < <("${STUDY[@]}" --names)
  mapfile -t CMDS < <("${STUDY[@]}" --list)
  mapfile -t GPUS < <("${STUDY[@]}" --gpus | awk '{print $2}')
  [ ${#NAMES[@]} -gt 0 ] && [ ${#NAMES[@]} -eq ${#CMDS[@]} ] || { echo "no jobs listed for ${group}"; exit 1; }
  # a job of the same name already queued or running writes the same artifact: skip it
  QUEUED="$(squeue -h -u "${USER}" -o "%j" 2>/dev/null || true)"
  for i in "${!NAMES[@]}"; do
    # ONLY=<regex>: submit just the matching jobs of the group (slot parts onto free queues)
    if [ -n "${ONLY:-}" ] && ! grep -Eq "${ONLY}" <<< "${NAMES[$i]}"; then continue; fi
    if grep -qx "${group}_${NAMES[$i]}" <<< "${QUEUED}"; then
      echo "skip ${group}_${NAMES[$i]}: already queued"; continue
    fi
    n="${GPUS[$i]}"
    hours="${HOURS}"
    if [ "${group}" = "opt" ] && [ "${n}" -le 2 ]; then hours=12; fi
    sbatch --job-name="${group}_${NAMES[$i]}" "${EXCLUDE_ARG[@]}" "${QOS_ARG[@]}" \
      --partition="${PART}" --gres="gpu:${GRES}:${n}" \
      --cpus-per-task=$((CORES * n)) --mem="$((MEM * n))G" --time="${hours}:00:00" \
      --export=ALL,CMD="${CMDS[$i]}",GPU_KIND="${GPU_KIND}" \
      examples/mesh/cluster/job.sbatch
  done
done
