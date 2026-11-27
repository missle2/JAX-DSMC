#!/bin/bash
# Platform table of the scaling study.  Sourced (not run) with PLATFORM set:
#
#   PLATFORM=pm-gpu   NERSC Perlmutter GPU node: 4x A100 (40/80 GB), 1x EPYC 7763 (64 cores)
#   PLATFORM=pm-cpu   NERSC Perlmutter CPU node: 2x EPYC 7763 (128 cores, 256 threads)
#   PLATFORM=alp-h200     CURC Alpine NVIDIA H200, partition ah200, gpu:h200:4 (whole cards;
#                         the MIG-sliced h200_2g/3g nodes are skipped)
#   PLATFORM=alp-a100     CURC Alpine NVIDIA A100, partition aa100, gpu:a100-40gb:3
#                         (GRES_MATCH=^gpu:a100_80gb: for the 80 GB nodes)
#   PLATFORM=alp-l40      CURC Alpine NVIDIA L40 (not L40S), partition al40, gpu:l40:3
#   PLATFORM=alp-rtx6000  CURC Alpine NVIDIA RTX PRO 6000 Blackwell, partition artxpro6000,
#                         gpu:rtx_pro_6000:4 (whole cards; the 1g/2g MIG nodes are skipped)
#   PLATFORM=alp-mi100    CURC Alpine AMD MI100, partition ami100, gpu:mi100:3
# (sinfo 2026-11-25: Alpine has no MI250 or L40S partition.)
#
# Sets: SITE, SB_ARGS (sbatch resource flags without node count / time), GPUS_PER_NODE
# (devices; 0 on CPU), CORES_PER_NODE, MODULES (loaded by `pf_modules`),
# KOKKOS_PRESET / KOKKOS_ARCH (SPARTA build), JAX_FLAVOR (build_jax.sh), BENCH_ROOT (where
# builds, envs, caches and results live: a shared filesystem), and MACHINE (log tag).
# Everything the clusters may name differently is an overridable variable (PART, GRES_TYPE,
# QOS, ACCOUNT, MODULES, ...): check the values with `bash bench/env/show_platform.sh`.
set -o pipefail

: "${PLATFORM:?set PLATFORM (pm-gpu pm-cpu alp-h200 alp-a100 alp-l40 alp-rtx6000 alp-mi100)}"
MACHINE="${MACHINE:-${PLATFORM}}"

# discover "partition cpus mem_MB gres" of the first node whose gres matches $1 and is a
# whole card (no MIG profile "<k>g.<m>gb" in the gres name)
pf_sinfo_gres() {
  sinfo -h -N -o "%P %c %m %G" 2>/dev/null | awk -v m="$1" '
    { n = split($4, g, ","); for (i = 1; i <= n; i++)
        if (tolower(g[i]) ~ tolower(m) && g[i] !~ /[0-9]g\.[0-9]+gb/) { print $1, $2, $3, g[i]; exit } }'
}

pf_alpine_gpu() {  # $1 = gres regex; fills PART GRES GRES_COUNT CORES_PER_NODE
  local line p ncpu mem g
  line="$(pf_sinfo_gres "$1" || true)"
  if [ -n "${line}" ]; then
    read -r p ncpu mem g <<< "${line}"
    PART="${PART:-${p%\*}}"
    GRES="${GRES_TYPE:-$(echo "${g}" | cut -d: -f2)}"
    GRES_COUNT="${GRES_COUNT:-$(echo "${g}" | cut -d: -f3 | cut -d'(' -f1)}"
    CORES_PER_NODE="${CORES_PER_NODE:-${ncpu}}"
  fi
  : "${PART:?no partition with a whole-card gres matching '$1' in sinfo: set PART and GRES_TYPE}"
  : "${GRES:=${GRES_TYPE:?set GRES_TYPE}}"
  : "${GRES_COUNT:?set GRES_COUNT (gres per node)}"
}

case "${PLATFORM}" in
  pm-gpu)
    SITE=nersc
    GPUS_PER_NODE=4; CORES_PER_NODE=64; SMT=2
    : "${ACCOUNT:?set ACCOUNT (NERSC project, e.g. m1234)}"
    SB_ARGS="-A ${ACCOUNT} -C ${CONSTRAINT:-gpu} -q ${QOS:-regular} --gpus-per-node=4"
    MODULES="${MODULES:-PrgEnv-gnu cudatoolkit craype-accel-nvidia80 cray-mpich cmake python}"
    KOKKOS_PRESET=kokkos_cuda; KOKKOS_ARCH="${KOKKOS_ARCH:-AMPERE80 ZEN3}"
    JAX_FLAVOR="${JAX_FLAVOR:-cuda12}"
    BENCH_ROOT="${BENCH_ROOT:-${PSCRATCH:-${SCRATCH}}/sparta-scaling}"
    ;;
  pm-cpu)
    SITE=nersc
    GPUS_PER_NODE=0; CORES_PER_NODE=128; SMT=2
    : "${ACCOUNT:?set ACCOUNT (NERSC project, e.g. m1234)}"
    SB_ARGS="-A ${ACCOUNT} -C ${CONSTRAINT:-cpu} -q ${QOS:-regular}"
    MODULES="${MODULES:-PrgEnv-gnu cray-mpich cmake python}"
    KOKKOS_PRESET=kokkos_omp; KOKKOS_ARCH="${KOKKOS_ARCH:-ZEN3}"
    JAX_FLAVOR="${JAX_FLAVOR:-cpu}"
    BENCH_ROOT="${BENCH_ROOT:-${PSCRATCH:-${SCRATCH}}/sparta-scaling}"
    ;;
  alp-h200|alp-a100|alp-l40|alp-rtx6000|alp-mi100)
    SITE=curc
    ACCOUNT="${ACCOUNT:-ucb856_asc1}"
    case "${PLATFORM}" in
      alp-h200)    pf_alpine_gpu "${GRES_MATCH:-^gpu:h200:}";         KOKKOS_PRESET=kokkos_cuda; KOKKOS_ARCH="${KOKKOS_ARCH:-HOPPER90}";     JAX_FLAVOR="${JAX_FLAVOR:-cuda12}" ;;
      alp-a100)    pf_alpine_gpu "${GRES_MATCH:-^gpu:a100-40gb:}";    KOKKOS_PRESET=kokkos_cuda; KOKKOS_ARCH="${KOKKOS_ARCH:-AMPERE80}";     JAX_FLAVOR="${JAX_FLAVOR:-cuda12}" ;;
      alp-l40)     pf_alpine_gpu "${GRES_MATCH:-^gpu:l40:}";          KOKKOS_PRESET=kokkos_cuda; KOKKOS_ARCH="${KOKKOS_ARCH:-ADA89}";        JAX_FLAVOR="${JAX_FLAVOR:-cuda12}" ;;
      alp-rtx6000) pf_alpine_gpu "${GRES_MATCH:-^gpu:rtx_pro_6000:}"; KOKKOS_PRESET=kokkos_cuda; KOKKOS_ARCH="${KOKKOS_ARCH:-BLACKWELL120}"; JAX_FLAVOR="${JAX_FLAVOR:-cuda12}" ;;
      alp-mi100)   pf_alpine_gpu "${GRES_MATCH:-^gpu:mi100:}";        KOKKOS_PRESET=kokkos_hip;  KOKKOS_ARCH="${KOKKOS_ARCH:-AMD_GFX908}";   JAX_FLAVOR="${JAX_FLAVOR:-rocm}" ;;
    esac
    SB_ARGS="-A ${ACCOUNT} -p ${PART} --qos=${QOS:-gpu-normal}"
    # MODULES: loaded by every job (runtime); BUILD_MODULES: by build_sparta.sh only.
    # NVIDIA: the JAX env is main's /projects/$USER/jax-env (`module load python` + pip
    # "jax[cuda12]", whose wheels bring their own CUDA; examples/mesh/cluster/env.sh), so jobs
    # load no cuda module that could shadow the wheels' libraries; nvcc (cuda module) is only
    # needed to build SPARTA/Kokkos, which links the CUDA runtime statically.
    if [ "${PLATFORM}" = alp-mi100 ]; then
      # ROCm: the system install /opt/rocm-6.4.4 (in the node images; 2026-11-27), not the
      # rocm/6.1.0 module, whose hipcc is older than Kokkos 5's minimum (6.2) and whose RCCL
      # lacks ncclCommRegister, which the jax-rocm60 plugin calls.  ROCM_ROOT overrides
      # (ROCM_ROOT= with MODULES/BUILD_MODULES naming rocm/6.1.0 restores the module).
      ROCM_ROOT="${ROCM_ROOT-/opt/rocm-6.4.4}"
      MODULES="${MODULES:-python gcc openmpi}"
      BUILD_MODULES="${BUILD_MODULES:-gcc cmake openmpi}"
      DEVICES_PER_GRES=1  # one HIP device per MI100 card
    else
      MODULES="${MODULES:-python gcc openmpi}"
      BUILD_MODULES="${BUILD_MODULES:-gcc cuda cmake openmpi}"
      JAX_VENV_DEFAULT="/projects/${USER}/jax-env"
      DEVICES_PER_GRES=1
    fi
    GPUS_PER_NODE=$(( GRES_COUNT * DEVICES_PER_GRES ))
    BENCH_ROOT="${BENCH_ROOT:-/projects/${USER}/sparta-scaling}"
    ;;
  *) echo "unknown PLATFORM ${PLATFORM}"; return 1 2>/dev/null || exit 1 ;;
esac

# sbatch resources for N nodes with all GPUs (or cores) of each node
pf_resources() {  # $1 = nodes
  if [ "${SITE}" = curc ]; then
    echo "${SB_ARGS} -N $1 --gres=gpu:${GRES}:${GRES_COUNT} --exclusive"
  else
    echo "${SB_ARGS} -N $1 --exclusive"
  fi
}

# sbatch resources for G GPUs of one node, not the whole node (NERSC: the shared QOS, charged
# by the GPU fraction; Alpine: a non-exclusive job with G whole cards)
pf_resources_shared() {  # $1 = GPUs
  if [ "${SITE}" = curc ]; then
    echo "${SB_ARGS} -N 1 -n $1 --gres=gpu:${GRES}:$1 -c $(( CORES_PER_NODE / GRES_COUNT ))"
  else
    echo "-A ${ACCOUNT} -C ${CONSTRAINT:-gpu} -q shared -N 1 -n $1 -c 32 --gpus-per-task=1"
  fi
}

pf_modules() {  # [$1 = module list; default MODULES]
  module purge 2>/dev/null || true
  local m
  for m in ${1:-${MODULES}}; do
    module load "${m}" 2>/dev/null || echo "[platforms.sh] module ${m} not available on $(hostname)"
  done
  # a ROCm install outside the module system (ROCM_ROOT, e.g. /opt/rocm-6.4.4)
  if [ -n "${ROCM_ROOT:-}" ]; then
    if [ -x "${ROCM_ROOT}/bin/hipcc" ]; then
      export ROCM_PATH="${ROCM_ROOT}" HIP_PATH="${ROCM_ROOT}"
      export PATH="${ROCM_ROOT}/bin:${PATH}"
      export LD_LIBRARY_PATH="${ROCM_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      export CMAKE_PREFIX_PATH="${ROCM_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    else
      echo "[platforms.sh] WARNING: ROCM_ROOT=${ROCM_ROOT} has no bin/hipcc on $(hostname)"
    fi
  fi
  # Alpine's rocm modules put hipcc on PATH but set no ROCM_PATH, which Kokkos' HIP setup and
  # the ROCm JAX plugin look for: derive it (2026-11-27)
  if [ -z "${ROCM_PATH:-}" ] && command -v hipcc > /dev/null 2>&1; then
    ROCM_PATH="$(dirname "$(dirname "$(command -v hipcc)")")"  # <rocm>/bin/hipcc
    export ROCM_PATH HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
  fi
}

# launch prefix of an MPI/JAX job, $3 = bind (default) | nobind | node:
#   bind    one rank per device, CUDA/ROCR_VISIBLE_DEVICES = its own GPU (sparta_jax: one
#           jax.distributed process per GPU)
#   nobind  every rank sees every GPU of its node (SPARTA/Kokkos: `-k on g <gpus>` picks the
#           device by local rank; GPU-aware MPI needs the peers visible for CUDA IPC -- NERSC's
#           recommendation for CUDA-aware MPI)
#   node    one process per node with all its GPUs (sparta_jax process-per-node variant)
# -c counts hardware threads: SMT=2 on Perlmutter
pf_launch() {  # $1 = nodes, $2 = ranks per node, [$3 = bind | nobind | node]
  local n=$(( $1 * $2 )) c=$(( ${SMT:-1} * CORES_PER_NODE / $2 ))
  if [ "${GPUS_PER_NODE}" -gt 0 ]; then
    case "${3:-bind}" in
      bind)   echo "srun -N $1 -n ${n} --ntasks-per-node=$2 -c ${c} --cpu-bind=cores --gpus-per-task=1 --gpu-bind=${GPU_BIND:-closest}" ;;
      nobind) echo "srun -N $1 -n ${n} --ntasks-per-node=$2 -c ${c} --cpu-bind=cores --gpus-per-node=${GRES_COUNT:-${GPUS_PER_NODE}} --gpu-bind=none" ;;
      node)   echo "srun -N $1 -n $1 --ntasks-per-node=1 -c $(( ${SMT:-1} * CORES_PER_NODE )) --cpu-bind=cores --gpus-per-node=${GRES_COUNT:-${GPUS_PER_NODE}} --gpu-bind=none" ;;
    esac
  else
    echo "srun -N $1 -n ${n} --ntasks-per-node=$2 -c ${c} --cpu-bind=cores"
  fi
}

SPARTA_SRC="${SPARTA_SRC:-${BENCH_ROOT}/src/sparta}"
SPARTA_BUILD_DIR="${BENCH_ROOT}/builds"
JAX_ENV_DIR="${BENCH_ROOT}/envs"
JAX_VENV_DEFAULT="${JAX_VENV_DEFAULT:-${JAX_ENV_DIR}/jax-${PLATFORM}}"
BUILD_MODULES="${BUILD_MODULES:-${MODULES}}"
RESULTS_DIR="${RESULTS_DIR:-${BENCH_ROOT}/results/${MACHINE}}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${BENCH_ROOT}/jax-cache/${PLATFORM}}"
export SPARTA_JAX_GEO_CACHE="${SPARTA_JAX_GEO_CACHE:-${BENCH_ROOT}/geo-cache}"
