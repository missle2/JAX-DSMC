#!/bin/bash
# Runtime environment of a benchmark job (sourced by bench/jobs/bench.sbatch with PLATFORM set).
# Loads the platform modules, activates the JAX env, and exports the settings both codes run
# with.  Tuning sweeps override single variables per run (bench/matrix.py writes them into the
# plan lines), so every default here is a `${VAR:-default}`.
HERE_RT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE_RT}/platforms.sh"
pf_modules

# ---- JAX ----------------------------------------------------------------------------------
if [ -n "${JAX_VARIANT:-}" ]; then JAX_VENV="${JAX_VENV:-${JAX_ENV_DIR}/jax-${PLATFORM}-${JAX_VARIANT}}"; fi
JAX_VENV="${JAX_VENV:-${JAX_VENV_DEFAULT}}"
if [ -f "${JAX_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${JAX_VENV}/bin/activate"
fi
export JAX_ENABLE_X64=1
if [ "${GPUS_PER_NODE}" -gt 0 ]; then
  if [ "${PLATFORM}" = alp-mi100 ]; then export JAX_PLATFORMS="${JAX_PLATFORMS:-rocm}"; else export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"; fi
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi
# one process per GPU owns it: preallocate most of it (no growth stalls); the tuning sweep
# tries the platform allocator and cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export XLA_FLAGS="${XLA_FLAGS:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${BENCH_REPO:-$(cd "${HERE_RT}/../.." && pwd)}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${SPARTA_JAX_GEO_CACHE}" 2>/dev/null || true

# ---- NCCL over Slingshot (Perlmutter): the libfabric (aws-ofi-nccl) plugin, else NCCL falls
# back to TCP sockets.  NERSC's nccl module ships the plugin; NCCL_DEBUG=INFO in the check job
# shows "NET/OFI" when it is used.  Overridable for the tuning sweep.
if [ "${SITE}" = nersc ] && [ "${GPUS_PER_NODE}" -gt 0 ]; then
  module load nccl 2>/dev/null || true
  export FI_PROVIDER="${FI_PROVIDER:-cxi}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
  export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
  export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
  export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
  # GPU-aware Cray MPICH for SPARTA/Kokkos
  export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
fi

# ---- SPARTA ---------------------------------------------------------------------------------
SPARTA_VARIANT="${SPARTA_VARIANT:-default}"
SPA="${SPA:-${SPARTA_BUILD_DIR}/sparta-${PLATFORM}-${SPARTA_VARIANT}/spa}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"
# Kokkos prints a warning per rank without these when OpenMP is not the backend
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# what plan lines use (they run in `bash -c`)
export PLATFORM MACHINE SITE GPUS_PER_NODE CORES_PER_NODE SMT GRES_COUNT RESULTS_DIR SPARTA_BUILD_DIR JAX_ENV_DIR BENCH_ROOT
export -f pf_launch
mkdir -p "${RESULTS_DIR}"

echo "[runtime] ${PLATFORM} on $(hostname): jax env ${JAX_VENV} ($(python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || echo none)), spa ${SPA}"
