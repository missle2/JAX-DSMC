#!/bin/bash
# Build everything the Alpine NVIDIA runs need, on acompile (4 tasks, 6 h):
#   1. a CUDA 12.8 toolkit in /projects/$USER/cuda-12.8 (once; Alpine's newest module is 12.1.1,
#      Blackwell needs >= 12.8)
#   2. a check that main's JAX env (/projects/$USER/jax-env) imports
#   3. SPARTA/Kokkos for alp-h200, alp-a100, alp-l40, alp-rtx6000 (default variant)
#   4. AMD MI100 (after the NVIDIA builds): its JAX env (jax-rocm60 plugin wheels on
#      rocm/6.1.0), an external Kokkos 4.7 (the bundled Kokkos 5 needs hipcc >= 6.2, Alpine has
#      6.1), and SPARTA against it (MI100=0 skips this part)
#
#   cd /projects/$USER/JAX-DSMC-sparta && sbatch bench/env/build_alpine.sbatch
#
# A failed platform is reported and the next one is built.  Log: build_alpine_<jobid>.out
# Overrides: PLATFORMS="alp-h200 ..." VARIANTS="default nomallocasync" BUILD_MODULES="gcc/... cmake openmpi"
#SBATCH --job-name=build_alpine
#SBATCH --partition=acompile
#SBATCH --ntasks=4
#SBATCH --time=06:00:00
#SBATCH --output=build_alpine_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
CUDA_ROOT="${CUDA_ROOT:-/projects/${USER}/cuda-12.8}"
CUDA_RUN="cuda_12.8.1_570.124.06_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${CUDA_RUN}"
PLATFORMS="${PLATFORMS:-alp-h200 alp-a100 alp-l40 alp-rtx6000}"
VARIANTS="${VARIANTS:-default}"
export BUILD_MODULES="${BUILD_MODULES:-gcc cmake openmpi}"
export JOBS=4
echo "[$(date -Iseconds)] build_alpine on $(hostname): platforms '${PLATFORMS}', variants '${VARIANTS}'"

# --- 1. CUDA toolkit ---------------------------------------------------------------------------
if [ -x "${CUDA_ROOT}/bin/nvcc" ]; then
  echo "CUDA toolkit present: $("${CUDA_ROOT}/bin/nvcc" --version | tail -1)"
else
  echo "[$(date -Iseconds)] installing the CUDA 12.8 toolkit into ${CUDA_ROOT}"
  tmp="/projects/${USER}/cuda-tmp-${SLURM_JOB_ID}"
  mkdir -p "${tmp}"
  if ( cd "${tmp}" && wget -q "${CUDA_URL}" ) && sh "${tmp}/${CUDA_RUN}" --silent --toolkit --no-drm \
       --no-man-page --installpath="${CUDA_ROOT}" --tmpdir="${tmp}"; then
    "${CUDA_ROOT}/bin/nvcc" --version | tail -1
  else
    echo "CUDA toolkit install FAILED (${CUDA_URL}; /tmp/cuda-installer.log): skipping the NVIDIA SPARTA builds"
    PLATFORMS=""
  fi
  rm -rf "${tmp}"
fi

# --- 2. the JAX env (main's) ---------------------------------------------------------------------
module purge 2>/dev/null || true
module load python 2>/dev/null || true
if /projects/"${USER}"/jax-env/bin/python3 -c 'import jax, scipy; print("jax", jax.__version__)'; then
  echo "JAX env ok"
else
  echo "WARNING: /projects/${USER}/jax-env does not import jax + scipy"
fi

# --- 3. SPARTA/Kokkos per platform ---------------------------------------------------------------
module purge 2>/dev/null || true
for m in ${BUILD_MODULES}; do module load "${m}"; done
echo "host compiler: $(g++ --version | head -1)   (CUDA 12.8 supports GCC <= 14)"
ok=(); bad=()
for P in ${PLATFORMS}; do
  for V in ${VARIANTS}; do
    echo; echo "[$(date -Iseconds)] ===== SPARTA ${P} / ${V} ====="
    if PLATFORM="${P}" VARIANT="${V}" CUDA_ROOT="${CUDA_ROOT}" bash bench/env/build_sparta.sh; then
      ok+=("${P}/${V}")
    else
      bad+=("${P}/${V}")
    fi
  done
done

# --- 4. AMD MI100 ----------------------------------------------------------------------------------
if [ "${MI100:-1}" = 1 ]; then
  echo; echo "[$(date -Iseconds)] ===== MI100: JAX env ====="
  if PLATFORM=alp-mi100 bash bench/env/build_jax.sh; then ok+=("alp-mi100/jax-env"); else bad+=("alp-mi100/jax-env"); fi
  echo; echo "[$(date -Iseconds)] ===== MI100: Kokkos 4.7 (HIP, gfx908) ====="
  KROOT="/projects/${USER}/sparta-scaling/kokkos/${KOKKOS_VERSION:-4.7.01}-alp-mi100"
  if PLATFORM=alp-mi100 bash bench/env/build_kokkos.sh; then
    ok+=("alp-mi100/kokkos")
    echo; echo "[$(date -Iseconds)] ===== SPARTA alp-mi100 / default (external Kokkos) ====="
    if PLATFORM=alp-mi100 VARIANT=default KOKKOS_ROOT="${KROOT}" bash bench/env/build_sparta.sh; then
      ok+=("alp-mi100/default")
    else
      bad+=("alp-mi100/default"); echo "SPARTA does not build against Kokkos 4.7 on ROCm 6.1: next is a ROCm >= 6.2 container"
    fi
  else
    bad+=("alp-mi100/kokkos")
  fi
fi

echo
echo "[$(date -Iseconds)] built: ${ok[*]:-none}"
echo "failed: ${bad[*]:-none}"
ls -l /projects/"${USER}"/sparta-scaling/builds/*/spa 2>/dev/null
