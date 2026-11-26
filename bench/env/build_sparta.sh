#!/bin/bash
# Build SPARTA for one platform and one build variant (login node, or a compute node for HIP
# builds that need the GPU toolchain present):
#
#   PLATFORM=pm-gpu ACCOUNT=m1234 bash bench/env/build_sparta.sh               # Kokkos CUDA, default
#   PLATFORM=pm-gpu VARIANT=nomallocasync bash bench/env/build_sparta.sh
#   PLATFORM=pm-cpu VARIANT=mpi bash bench/env/build_sparta.sh                  # plain MPI (no Kokkos)
#   PLATFORM=pm-cpu VARIANT=kokkos_omp bash bench/env/build_sparta.sh
#
# -> ${BENCH_ROOT}/builds/sparta-${PLATFORM}-${VARIANT}/src/spa_*  (+ BUILD_INFO.txt)
#
# Source: SPARTA_SRC (default ${BENCH_ROOT}/src/sparta; clone the fork there; its commit is
# recorded).  The bundled Kokkos (lib/kokkos, 5.0.x) needs C++20: GCC >= 10.4 / recent clang,
# CUDA >= 12.2 or ROCm >= 6.2 -- the configure step stops early on older toolchains.
#
# VARIANTs (build-side tuning knobs; runtime knobs are in bench/tune.py):
#   default          Release -O3, the preset's Kokkos settings, arch from platforms.sh
#   nomallocasync    -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF (CUDA)
#   march            + -march=native (host code; CPU builds and the host side of GPU builds)
#   multikernel      -D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON (HIP)
#   bigbig           -D SPARTA_SIZES=bigbig (64-bit ids/cells)
#   mpi              no Kokkos (SPARTA's own CPU code), MPI           [CPU platforms]
#   kokkos_omp       Kokkos OpenMP backend                             [CPU platforms]
#   kokkos_serial    Kokkos Serial backend (one rank per core)         [CPU platforms]
#   clang / cce      another compiler: CXX for host/MPI builds (cce = Cray CC wrapper on NERSC)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/platforms.sh"
pf_modules "${BUILD_MODULES}"
VARIANT="${VARIANT:-default}"
SRC="${SPARTA_SRC}"
BUILD="${SPARTA_BUILD_DIR}/sparta-${PLATFORM}-${VARIANT}"
[ -f "${SRC}/cmake/CMakeLists.txt" ] || { echo "no SPARTA source at ${SRC} (set SPARTA_SRC)"; exit 1; }

PRESET="${KOKKOS_PRESET}"
EXTRA=()
CXX_FLAGS="-O3"
case "${VARIANT}" in
  default) ;;
  nomallocasync) EXTRA+=(-D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF) ;;
  march) CXX_FLAGS="-O3 -march=native" ;;
  multikernel) EXTRA+=(-D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON) ;;
  bigbig) EXTRA+=(-D SPARTA_SIZES=bigbig) ;;
  mpi) PRESET=mpi ;;
  kokkos_omp) PRESET=kokkos_omp ;;
  kokkos_serial) PRESET=kokkos_mpi_only ;;
  clang) export CXX="${CXX:-clang++}" ;;
  cce) export CXX=CC ;;
  *) echo "unknown VARIANT ${VARIANT}"; exit 1 ;;
esac

# Kokkos architecture flags: KOKKOS_ARCH="AMPERE80 ZEN3" -> -D Kokkos_ARCH_AMPERE80=ON ...;
# the presets switch HOPPER90 / VEGA90A on, which we switch off unless asked for
ARCH=()
if [ "${PRESET}" != mpi ]; then
  for off in HOPPER90 VEGA90A; do ARCH+=(-D "Kokkos_ARCH_${off}=OFF"); done
  for a in ${KOKKOS_ARCH}; do ARCH+=(-D "Kokkos_ARCH_${a}=ON"); done
fi
# NERSC: the Cray wrappers are the MPI compilers; GPU-aware Cray MPICH needs the GTL library
if [ "${SITE}" = nersc ]; then
  export MPICH_GPU_SUPPORT_ENABLED=1
  [ "${PRESET}" = mpi ] && EXTRA+=(-D CMAKE_CXX_COMPILER="${CXX:-CC}")
  [ "${PRESET}" = kokkos_omp ] && EXTRA+=(-D CMAKE_CXX_COMPILER="${CXX:-CC}")
  if [ "${PRESET}" = kokkos_cuda ]; then
    export NVCC_WRAPPER_DEFAULT_COMPILER="${CXX:-CC}"
    EXTRA+=(-D MPI_CXX_COMPILER=CC)
  fi
fi
if [ "${PRESET}" = kokkos_hip ]; then
  EXTRA+=(-D CMAKE_CXX_COMPILER="${CXX:-hipcc}")
fi

echo "[build_sparta] ${PLATFORM}/${VARIANT}: preset ${PRESET}, arch '${KOKKOS_ARCH}', ${SRC} -> ${BUILD}"
if [ -n "${CHECK:-}" ]; then
  echo cmake -S "${SRC}/cmake" -B "${BUILD}" -C "${SRC}/cmake/presets/${PRESET}.cmake" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" "${ARCH[@]}" "${EXTRA[@]}"
  exit 0
fi
cmake -S "${SRC}/cmake" -B "${BUILD}" -C "${SRC}/cmake/presets/${PRESET}.cmake" \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" "${ARCH[@]}" "${EXTRA[@]}" \
  2>&1 | tee "${BUILD}.configure.log"
cmake --build "${BUILD}" -j "${JOBS:-16}" 2>&1 | tail -20
EXE="$(find "${BUILD}" -maxdepth 2 -type f -perm -u+x -name 'spa_*' | head -1)"
[ -x "${EXE}" ] || { echo "no spa_* executable under ${BUILD}"; exit 1; }
ln -sfn "${EXE}" "${BUILD}/spa"
{
  echo "SPARTA build ${PLATFORM}/${VARIANT} $(date -Iseconds) on $(hostname)"
  echo "source ${SRC} commit $(git -C "${SRC}" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "preset ${PRESET} arch '${KOKKOS_ARCH}' flags '${CXX_FLAGS}' extra '${EXTRA[*]}'"
  echo "modules:"; module -t list 2>&1 | sed 's/^/  /' || true
  "${CXX:-c++}" --version 2>&1 | head -1 || true
  command -v nvcc > /dev/null && nvcc --version | tail -2
  command -v hipcc > /dev/null && hipcc --version | head -2
} > "${BUILD}/BUILD_INFO.txt"
echo "[build_sparta] ${BUILD}/spa -> ${EXE}"
