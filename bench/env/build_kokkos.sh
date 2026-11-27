#!/bin/bash
# Build and install an external Kokkos for one platform, for SPARTA's USE_EXTERNAL_KOKKOS
# (build_sparta.sh KOKKOS_ROOT=...).  Needed where the toolchain is older than the bundled
# Kokkos 5's minimum: Alpine's newest ROCm is 6.1.0, Kokkos 5 needs hipcc >= 6.2.
#
#   PLATFORM=alp-mi100 bash bench/env/build_kokkos.sh      # -> ${BENCH_ROOT}/kokkos/<version>-<platform>
#   KOKKOS_VERSION=4.7.01 (default)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/platforms.sh"
pf_modules "${BUILD_MODULES}"
V="${KOKKOS_VERSION:-4.7.01}"
SRC="${BENCH_ROOT}/src/kokkos-${V}"
PREFIX="${BENCH_ROOT}/kokkos/${V}-${PLATFORM}"
[ -d "${SRC}" ] || git clone --depth 1 -b "${V}" https://github.com/kokkos/kokkos "${SRC}"
ARCH=()
for a in ${KOKKOS_ARCH}; do ARCH+=(-D "Kokkos_ARCH_${a}=ON"); done
case "${KOKKOS_PRESET}" in
  # hipcc explicitly: the openmpi module exports CXX=mpicxx (g++)
  kokkos_hip)  BACKEND=(-D Kokkos_ENABLE_HIP=ON -D CMAKE_CXX_COMPILER="${HIPCXX:-$(command -v hipcc)}") ;;
  kokkos_cuda) BACKEND=(-D Kokkos_ENABLE_CUDA=ON -D CMAKE_CXX_COMPILER="${SRC}/bin/nvcc_wrapper") ;;
  kokkos_omp)  BACKEND=(-D Kokkos_ENABLE_OPENMP=ON) ;;
  *)           BACKEND=() ;;
esac
echo "[build_kokkos] Kokkos ${V} for ${PLATFORM}: ${BACKEND[*]} ${ARCH[*]} -> ${PREFIX}"
cmake -S "${SRC}" -B "${PREFIX}.build" -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
  -D CMAKE_CXX_STANDARD=20 -D Kokkos_ENABLE_SERIAL=ON "${BACKEND[@]}" "${ARCH[@]}"
cmake --build "${PREFIX}.build" -j "${JOBS:-4}"
cmake --install "${PREFIX}.build"
echo "[build_kokkos] installed: ${PREFIX}  (KOKKOS_ROOT=${PREFIX})"
