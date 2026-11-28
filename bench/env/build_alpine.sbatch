#!/bin/bash -l
# Build everything the Alpine NVIDIA runs need, on acompile (4 tasks, 6 h):
#   1. a CUDA 12.8 toolkit in /projects/$USER/cuda-12.8 (once; Alpine's newest module is 12.1.1,
#      Blackwell needs >= 12.8)
#   2. a check that main's JAX env (/projects/$USER/jax-env) imports
#   3. SPARTA/Kokkos for alp-h200, alp-a100, alp-l40, alp-rtx6000 (default variant)
#   4. AMD MI100 (after the NVIDIA builds): a user-space ROCm 6.4.4 on /scratch (rocm_user.sh:
#      the nodes' runtime + toolchain + libraries from their own repo), the JAX env (AMD's
#      ROCm 6.4.4 wheels, jax 0.4.35), SPARTA with the bundled Kokkos 5 (fallback: external
#      Kokkos 4.7).  MI100=0 skips this part.
#
#   cd /projects/$USER/JAX-DSMC-sparta && sbatch bench/env/build_alpine.sbatch
#
# A failed platform is reported and the next one is built.  Log: build_alpine_<jobid>.out
# Overrides are ARGUMENTS (VAR=value), not the environment:
#   sbatch bench/env/build_alpine.sbatch PLATFORMS= MI100_VARIANTS="default:tgt o1:o1"
#   (PLATFORMS="alp-h200 ..." VARIANTS="default nomallocasync" BUILD_MODULES="gcc/... cmake openmpi")
# Environment: --export=NONE + a login shell (bash -l) = what an interactive acompile shell has, on
# the compile node itself.  Jobs that inherited a LOGIN-node environment (--export=ALL, submitted
# from a login shell) could not load gcc/cmake/openmpi on the compile node -- builds 36547906,
# 36550537, 2026-11-28 -- while the same script submitted from inside acompile could (36503353).
#SBATCH --job-name=build_alpine
#SBATCH --partition=acompile
#SBATCH --qos=compile
#SBATCH --account=ucb856_asc1
#SBATCH --ntasks=4
#SBATCH --time=06:00:00
#SBATCH --output=build_alpine_%j.out
#SBATCH --export=NONE
set -uo pipefail
for kv in "$@"; do  # VAR=value overrides (the environment is not inherited)
  case "${kv}" in *=*) export "${kv?}" ;; *) echo "ignoring argument '${kv}' (want VAR=value)" ;; esac
done
echo "[$(date -Iseconds)] environment: login shell on $(hostname); module: $(type -t module || echo none)"
cd "${SLURM_SUBMIT_DIR}"
CUDA_ROOT="${CUDA_ROOT:-/projects/${USER}/cuda-12.8}"
CUDA_RUN="cuda_12.8.1_570.124.06_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${CUDA_RUN}"
# PLATFORMS= (empty) skips every NVIDIA build; unset means all four.  A platform whose spa
# exists is skipped unless REBUILD=1.
PLATFORMS="${PLATFORMS-alp-h200 alp-a100 alp-l40 alp-rtx6000}"
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
PLATFORM=alp-a100 source bench/env/platforms.sh > /dev/null  # for pf_modules
pf_modules "python"
if /projects/"${USER}"/jax-env/bin/python3 -c 'import jax, scipy; print("jax", jax.__version__)'; then
  echo "JAX env ok"
else
  echo "WARNING: /projects/${USER}/jax-env does not import jax + scipy"
fi

# --- 3. SPARTA/Kokkos per platform ---------------------------------------------------------------
pf_modules "${BUILD_MODULES}"
echo "host compiler: $(g++ --version | head -1)   (CUDA 12.8 supports GCC <= 14)"
ok=(); bad=()
for P in ${PLATFORMS}; do
  for V in ${VARIANTS}; do
    if [ -x "/projects/${USER}/sparta-scaling/builds/sparta-${P}-${V}/spa" ] && [ "${REBUILD:-0}" != 1 ]; then
      echo "[$(date -Iseconds)] SPARTA ${P}/${V}: already built (REBUILD=1 rebuilds)"; ok+=("${P}/${V}"); continue
    fi
    echo; echo "[$(date -Iseconds)] ===== SPARTA ${P} / ${V} ====="
    if PLATFORM="${P}" VARIANT="${V}" CUDA_ROOT="${CUDA_ROOT}" bash bench/env/build_sparta.sh; then
      ok+=("${P}/${V}")
    else
      bad+=("${P}/${V}")
    fi
  done
done

# --- 4. AMD MI100 ----------------------------------------------------------------------------------
# MI100_VARIANTS="default:tgt o1:o1 multikernel:mk": only these SPARTA builds (VARIANT:BUILD_TAG,
# next to the existing default), no ROCm/JAX steps -- the sphere-abort diagnosis.  (A `sbatch
# --wrap` job -- /bin/sh -- found no gcc/cmake/openmpi modules, 2026-11-28; this bash job does.)
if [ -n "${MI100_VARIANTS:-}" ]; then
  pf_modules "${BUILD_MODULES}"
  for VT in ${MI100_VARIANTS}; do
    V="${VT%%:*}"; T="${VT##*:}"
    echo; echo "[$(date -Iseconds)] ===== SPARTA alp-mi100 / ${V} -> ${T} ====="
    if PLATFORM=alp-mi100 VARIANT="${V}" BUILD_TAG="${T}" bash bench/env/build_sparta.sh; then
      ok+=("alp-mi100/${T}"); else bad+=("alp-mi100/${T}"); fi
  done
  MI100=0
fi
if [ "${MI100:-1}" = 1 ]; then
  echo; echo "[$(date -Iseconds)] ===== MI100: user-space ROCm 6.4.4 (toolchain + libraries) ====="
  RU="/scratch/alpine/${USER}/rocm-6.4.4-user"
  if bash bench/env/rocm_user.sh; then ok+=("alp-mi100/rocm-6.4.4"); else bad+=("alp-mi100/rocm-6.4.4"); fi
  XLIB="${RU}/usr/lib64:${RU}/usr/lib64/atlas"
  echo; echo "[$(date -Iseconds)] ===== MI100: JAX env ====="
  if PLATFORM=alp-mi100 ROCM_EXTRA_LIBS="${XLIB}" bash bench/env/build_jax.sh; then ok+=("alp-mi100/jax-env"); else bad+=("alp-mi100/jax-env"); fi
  # SPARTA: bundled Kokkos 5 with ROCm 6.4.4; if that fails, an external Kokkos 4.7
  echo; echo "[$(date -Iseconds)] ===== SPARTA alp-mi100 / default (bundled Kokkos, ROCm ${ROCM_ROOT:-/opt/rocm-6.4.4}) ====="
  if PLATFORM=alp-mi100 VARIANT=default bash bench/env/build_sparta.sh; then
    ok+=("alp-mi100/default")
  else
    echo; echo "[$(date -Iseconds)] ===== MI100 fallback: Kokkos 4.7 (HIP, gfx908) + SPARTA ====="
    KROOT="/projects/${USER}/sparta-scaling/kokkos/${KOKKOS_VERSION:-4.7.01}-alp-mi100"
    if PLATFORM=alp-mi100 bash bench/env/build_kokkos.sh \
       && PLATFORM=alp-mi100 VARIANT=default KOKKOS_ROOT="${KROOT}" bash bench/env/build_sparta.sh; then
      ok+=("alp-mi100/default(kokkos4.7)")
    else
      bad+=("alp-mi100/default")
    fi
  fi
fi

echo
echo "[$(date -Iseconds)] built: ${ok[*]:-none}"
echo "failed: ${bad[*]:-none}"
ls -l /projects/"${USER}"/sparta-scaling/builds/*/spa 2>/dev/null
