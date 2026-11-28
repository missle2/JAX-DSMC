#!/bin/bash
# Build SPARTA for one platform and one build variant (login node, or a compute node for HIP
# builds that need the GPU toolchain present):
#
#   PLATFORM=pm-gpu ACCOUNT=m1234 bash bench/env/build_sparta.sh               # Kokkos CUDA, default
#   PLATFORM=pm-gpu VARIANT=nomallocasync bash bench/env/build_sparta.sh
#   PLATFORM=pm-cpu VARIANT=mpi bash bench/env/build_sparta.sh                  # plain MPI (no Kokkos)
#   PLATFORM=pm-cpu VARIANT=kokkos_omp bash bench/env/build_sparta.sh
#   PLATFORM=alp-rtx6000 BUILD_MODULES="gcc cmake openmpi" CUDA_ROOT=/projects/$USER/cuda-12.8 \
#       bash bench/env/build_sparta.sh                                       # own CUDA toolkit
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
#   mallocasync      -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON (CUDA; Kokkos 5 default is OFF)
#   mpi              no Kokkos (SPARTA's own CPU code), MPI           [CPU platforms]
#   mpi_march        mpi with -march=native                            [CPU platforms]
#   mpi_cce          mpi with the Cray compiler (PrgEnv-cray)          [NERSC CPU]
#   kokkos_omp       Kokkos OpenMP backend                             [CPU platforms]
#   kokkos_serial    Kokkos Serial backend (one rank per core)         [CPU platforms]
#   clang / cce      another compiler: CXX for host/MPI builds (cce = Cray CC wrapper on NERSC)
# KOKKOS_ROOT=<install>  use an external Kokkos (bench/env/build_kokkos.sh) instead of lib/kokkos
# BUILD_TAG=<name>       build directory sparta-${PLATFORM}-<name> instead of -${VARIANT}
#                        (e.g. VARIANT=default KOKKOS_ROOT=... BUILD_TAG=k47: same flags, other Kokkos)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/platforms.sh"
pf_modules "${BUILD_MODULES}"
# CUDA_ROOT: a user-installed CUDA toolkit instead of a cuda module (Alpine's newest module is
# 12.1.1; Blackwell sm_120 needs >= 12.8 and Kokkos 5 a recent CUDA).  Toolkit only: the node's
# driver is used, and nvcc links the CUDA runtime statically, so jobs need nothing from it.
if [ -n "${CUDA_ROOT:-}" ]; then
  [ -x "${CUDA_ROOT}/bin/nvcc" ] || { echo "no nvcc in ${CUDA_ROOT}/bin"; exit 1; }
  export PATH="${CUDA_ROOT}/bin:${PATH}" CUDA_HOME="${CUDA_ROOT}" CUDA_PATH="${CUDA_ROOT}"
  export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  CUDA_ARGS=(-D "CUDAToolkit_ROOT=${CUDA_ROOT}")
else
  CUDA_ARGS=()
fi
if [ "${KOKKOS_PRESET}" = kokkos_cuda ] && command -v nvcc > /dev/null; then
  echo "[build_sparta] nvcc: $(nvcc --version | tail -1)"
fi
VARIANT="${VARIANT:-default}"
SRC="${SPARTA_SRC}"
BUILD="${SPARTA_BUILD_DIR}/sparta-${PLATFORM}-${BUILD_TAG:-${VARIANT}}"
[ -f "${SRC}/cmake/CMakeLists.txt" ] || { echo "no SPARTA source at ${SRC} (set SPARTA_SRC)"; exit 1; }

PRESET="${KOKKOS_PRESET}"
EXTRA=()
CXX_FLAGS="-O3"
case "${VARIANT}" in
  default) ;;
  nomallocasync) EXTRA+=(-D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF) ;;
  # Kokkos 5's default is OFF ("CUDA malloc async support disabled" in the build logs)
  mallocasync) EXTRA+=(-D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON) ;;
  march) CXX_FLAGS="-O3 -march=native" ;;
  multikernel) EXTRA+=(-D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON) ;;
  # HIP diagnosis (MI100 sphere: device-side "Surf_collide tsurf <= 0.0" abort, job 36514785):
  # a -O3 miscompile shows as o1 running where default aborts
  o1) CXX_FLAGS="-O1" ;;
  # SPARTA has no size option: its mpi_big preset appends -DSPARTA_BIGBIG to
  # SPARTA_CXX_COMPILE_FLAGS (a -D SPARTA_SIZES was ignored: Perlmutter build 2026-11-27)
  bigbig) SPARTA_FLAGS_APPEND="-DSPARTA_BIGBIG" ;;
  mpi) PRESET=mpi ;;
  mpi_march) PRESET=mpi; CXX_FLAGS="-O3 -march=native" ;;
  # Cray compiler (CCE) through PrgEnv-cray; jobs run these binaries under the same PrgEnv
  # (bench/env/with_modules.sh PrgEnv-cray -- ...)
  mpi_cce) PRESET=mpi; PRGENV=cray ;;
  kokkos_omp) PRESET=kokkos_omp ;;
  kokkos_serial) PRESET=kokkos_mpi_only ;;
  clang) export CXX="${CXX:-clang++}" ;;
  cce) PRGENV=cray ;;
  *) echo "unknown VARIANT ${VARIANT}"; exit 1 ;;
esac

if [ -n "${PRGENV:-}" ]; then
  module load "PrgEnv-${PRGENV}" || { echo "cannot load PrgEnv-${PRGENV}"; exit 1; }
  echo "[build_sparta] PrgEnv-${PRGENV}: $(CC --version 2>&1 | head -1)"
fi
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
  # the Cray wrapper CC is the MPI C++ compiler (gcc of PrgEnv-gnu + cray-mpich + target flags)
  if [ "${PRESET}" = kokkos_cuda ]; then
    export NVCC_WRAPPER_DEFAULT_COMPILER="${CXX:-CC}"
    EXTRA+=(-D MPI_CXX_COMPILER=CC)
  else
    EXTRA+=(-D CMAKE_CXX_COMPILER="${CXX:-CC}" -D CMAKE_C_COMPILER="${CC:-cc}")
  fi
fi
if [ "${PRESET}" = kokkos_hip ]; then
  # hipcc explicitly: the openmpi module exports CXX=mpicxx (g++), which cannot compile HIP
  # (2026-11-25 build log); MPI is found by CMake.  KISS FFT: the decks use no FFT.
  EXTRA+=(-D CMAKE_CXX_COMPILER="${HIPCXX:-$(command -v hipcc)}" -D FFT_KOKKOS=KISS)
  # the GPU target, explicitly: hip-config otherwise probes the build host's GPU (none on
  # acompile: "amdgpu-arch failed ... the default architecture gfx906 will be used")
  HIP_ARCH="${HIP_ARCH:-$(echo "${KOKKOS_ARCH}" | sed -n 's/.*AMD_GFX\([0-9A-Fa-f]*\).*/gfx\1/p' | tr 'A-F' 'a-f')}"
  [ -n "${HIP_ARCH}" ] && EXTRA+=(-D GPU_TARGETS="${HIP_ARCH}" -D AMDGPU_TARGETS="${HIP_ARCH}")
  # hipcc (clang) takes the C++ standard library of the OS's GCC 8.5 unless told otherwise:
  # no C++20 <concepts> (2026-11-27 build log).  Point it at the loaded gcc module's toolchain.
  CXX_FLAGS="${CXX_FLAGS} --gcc-toolchain=$(dirname "$(dirname "$(command -v gcc)")")"
fi
# KOKKOS_ROOT: link an installed Kokkos (build_kokkos.sh) instead of the bundled lib/kokkos,
# e.g. Kokkos 4.7 where the toolchain is older than Kokkos 5's minimum (hipcc 6.2, nvcc 12.2)
if [ -n "${KOKKOS_ROOT:-}" ]; then
  EXTRA+=(-D USE_EXTERNAL_KOKKOS=ON -D "Kokkos_ROOT=${KOKKOS_ROOT}")
  ARCH=()  # the architecture is that of the installed Kokkos
fi

echo "[build_sparta] ${PLATFORM}/${VARIANT}: preset ${PRESET}, arch '${KOKKOS_ARCH}', ${SRC} -> ${BUILD}"
if [ -n "${CHECK:-}" ]; then
  echo cmake -S "${SRC}/cmake" -B "${BUILD}" -C "${SRC}/cmake/presets/${PRESET}.cmake" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" "${ARCH[@]}" "${EXTRA[@]}" "${CUDA_ARGS[@]}"
  exit 0
fi
mkdir -p "${SPARTA_BUILD_DIR}"  # the configure log below lives next to the build (first build: absent)
# the preset, or a wrapper that includes it and appends to SPARTA_CXX_COMPILE_FLAGS (the presets
# accumulate that list -- e.g. Kokkos' MPI C++ skip flags -- so it is extended, not replaced)
PRESET_FILE="${SRC}/cmake/presets/${PRESET}.cmake"
if [ -n "${SPARTA_FLAGS_APPEND:-}" ]; then
  mkdir -p "${SPARTA_BUILD_DIR}"
  PRESET_FILE="${BUILD}.preset.cmake"
  cat > "${PRESET_FILE}" <<PRESET_EOF
include(${SRC}/cmake/presets/${PRESET}.cmake)
set(SPARTA_CXX_COMPILE_FLAGS \${SPARTA_CXX_COMPILE_FLAGS} ${SPARTA_FLAGS_APPEND}
    CACHE STRING "Compiler flags use when building .o files for spa_*" FORCE)
PRESET_EOF
fi
# --fresh: no cache from an earlier configure (another compiler/ROCm) survives
cmake --fresh -S "${SRC}/cmake" -B "${BUILD}" -C "${PRESET_FILE}" \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" "${ARCH[@]}" "${EXTRA[@]}" "${CUDA_ARGS[@]}" \
  2>&1 | tee "${BUILD}.configure.log"
# Parallelism: the CPUs of this allocation (acompile: 4), not a fixed 16 -- 16 hipcc processes in
# a 4-task acompile allocation died without a compiler message (MI100 k47 build, 2026-11-28).
# The whole build output goes to ${BUILD}.build.log; on failure its first errors are printed
# (a `| tail -20` alone had cut them off).
JOBS="${JOBS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
if ! cmake --build "${BUILD}" -j "${JOBS}" > "${BUILD}.build.log" 2>&1; then
  echo "[build_sparta] BUILD FAILED (-j ${JOBS}); first errors from ${BUILD}.build.log:"
  grep -n -m 5 -A 12 -E "error:|Error [0-9]|Killed|killed|signal|No space" "${BUILD}.build.log" | head -80
  tail -n 5 "${BUILD}.build.log"
  exit 1
fi
tail -n 3 "${BUILD}.build.log"
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
