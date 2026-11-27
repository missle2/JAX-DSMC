#!/bin/bash
# A user-space ROCm 6.4.4 for the MI100 runs, built without root from the nodes' own repository.
#
#   bash bench/env/rocm_user.sh            # -> ${ROCM_USER_DIR}/opt/rocm-6.4.4  (~10 GB before trimming)
#
# Why (2026-11-27, Alpine):
#   * the nodes install only the ROCm 6.4.4 RUNTIME (/opt/rocm-6.4.4: HIP runtime, HSA, comgr,
#     rocm-smi; matching the amdgpu 6.4.4 driver) -- no hipcc, no math libraries, no RCCL;
#   * the rocm/6.1.0 module is too old: hipcc < Kokkos 5's 6.2 minimum, and its RCCL lacks
#     ncclCommRegister, which the PyPI jax-rocm60 0.5.0 plugin calls;
#   * the "rocm" (6.4.4) repository is configured and enabled on the nodes.
# So: a tree that symlinks the system runtime (same version as the driver) and adds, unpacked
# from that repository's RPMs, the toolchain SPARTA/Kokkos needs (hipcc, hip-devel, rocm-llvm,
# device libraries) and the libraries AMD's ROCm 6.4.4 JAX wheels link (rocBLAS, rocSOLVER,
# hipSOLVER, hipBLAS, hipSPARSE, hipFFT/rocFFT, MIOpen, RCCL), plus whatever they pull in that
# the nodes lack (dnf --resolve: SuiteSparse, ATLAS, ...).
#
# Output: ROCM_ROOT=${ROCM_USER_DIR}/opt/rocm-6.4.4 (platforms.sh picks it up for alp-mi100) and
# extra OS libraries in ${ROCM_USER_DIR}/usr/lib64[/atlas].
# TRIM=1 (default) deletes the rocBLAS/MIOpen GPU kernels of other architectures (keeps gfx908).
set -euo pipefail
V=6.4.4
ROCM_USER_DIR="${ROCM_USER_DIR:-/scratch/alpine/${USER}/rocm-${V}-user}"
SYS="/opt/rocm-${V}"
ROOT="${ROCM_USER_DIR}/opt/rocm-${V}"
RPMS="${ROCM_USER_DIR}/rpms"
KEEP_ARCH="${KEEP_ARCH:-gfx908}"
PKGS="${ROCM_PKGS:-hipcc hip-devel rocm-llvm rocm-device-libs hsa-rocr-devel rocm-cmake \
rocblas rocsolver hipsolver hipblas hipsparse rocsparse hipfft rocfft miopen-hip rccl rocthrust rocprim}"
[ -d "${SYS}" ] || { echo "no ${SYS} on $(hostname)"; exit 1; }
mkdir -p "${RPMS}" "${ROCM_USER_DIR}"
echo "[rocm_user] ${SYS} (system runtime) + RPMs -> ${ROOT}"

# 1. the system runtime, as symlinks (a later system update stays in effect)
if [ ! -e "${ROOT}/lib/libamdhip64.so.6" ]; then
  mkdir -p "${ROOT}"
  cp -as "${SYS}/." "${ROOT}/"
fi

# 2. the missing packages (only Rocky's and the ROCm repository: other vendor repos are slow)
#    Incremental: only the packages whose RPM is not in ${RPMS} yet (a later addition to PKGS,
#    e.g. rocthrust/rocprim 2026-11-27, does not download the 2.8 GB again).
# an empty .unpacked is the old all-done marker: list the RPMs present BEFORE new downloads
if [ -f "${ROCM_USER_DIR}/.unpacked" ] && [ ! -s "${ROCM_USER_DIR}/.unpacked" ]; then
  ls "${RPMS}"/*.rpm | xargs -n1 basename > "${ROCM_USER_DIR}/.unpacked"
fi
NEED=()
for p in ${PKGS}; do
  ls "${RPMS}/${p}"-[0-9]*.rpm > /dev/null 2>&1 || NEED+=("${p}")
done
if [ "${#NEED[@]}" -gt 0 ]; then
  echo "[rocm_user] downloading: ${NEED[*]}"
  timeout 1800 dnf download --resolve --disablerepo='*' --enablerepo=rocm,baseos,appstream,powertools \
    --destdir "${RPMS}" "${NEED[@]}"
fi
echo "[rocm_user] $(ls "${RPMS}"/*.rpm | wc -l) RPMs, $(du -sh "${RPMS}" | cut -f1)"

# 3. unpack, in parallel (one RPM per allocated task), with progress.  The RPMs are xz-compressed
#    (2.8 GB -> 10+ GB, tens of thousands of files on /scratch): minutes, CPU-bound.  GPU kernel
#    files of other architectures are not written at all (cpio --nonmatching), and -u overwrites
#    whatever a cancelled earlier run left half-written.
#    Incremental too: ${ROCM_USER_DIR}/.unpacked lists the RPMs already unpacked.
DONE_LIST="${ROCM_USER_DIR}/.unpacked"
touch "${DONE_LIST}"
TODO="$(ls -S "${RPMS}"/*.rpm | while read -r r; do grep -qxF "${r##*/}" "${DONE_LIST}" || echo "${r}"; done)"
if [ -n "${TODO}" ]; then
  SKIP=()
  for a in gfx803 gfx900 gfx906 gfx90a gfx940 gfx941 gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 \
           gfx1031 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201; do
    [ "${a}" = "${KEEP_ARCH}" ] || SKIP+=("*${a}*")
  done
  [ "${TRIM:-1}" = 1 ] || SKIP=()
  export ROCM_USER_DIR
  unpack_one() {
    local r="$1" t0; t0=$(date +%s)
    shift
    ( cd "${ROCM_USER_DIR}" && rpm2cpio "${r}" | cpio -idmu --quiet ${1:+--nonmatching} "$@" 2> /dev/null ) \
      || echo "  (cpio warnings: ${r##*/})"
    echo "  unpacked ${r##*/} ($(du -h "${r}" | cut -f1) rpm) in $(( $(date +%s) - t0 )) s"
  }
  export -f unpack_one
  NPAR="${SLURM_NTASKS:-4}"
  echo "[rocm_user] unpacking $(echo "${TODO}" | wc -l) RPM(s), ${NPAR} at a time, keeping ${KEEP_ARCH} kernels"
  # largest first, so the long ones start early
  echo "${TODO}" | xargs -P "${NPAR}" -I{} bash -c 'unpack_one "$@"' _ {} "${SKIP[@]}"
  echo "${TODO}" | xargs -n1 basename >> "${DONE_LIST}"
fi

# 4. trim the GPU kernel libraries to KEEP_ARCH
if [ "${TRIM:-1}" = 1 ]; then
  for d in "${ROOT}/lib/rocblas/library" "${ROOT}/lib/hipblaslt/library" "${ROOT}/share/miopen/db"; do
    [ -d "${d}" ] || continue
    find "${d}" -type f \( -name '*gfx*' \) ! -name "*${KEEP_ARCH}*" -delete
  done
fi
echo "[rocm_user] tree: $(du -sh "${ROCM_USER_DIR}" --exclude=rpms | cut -f1)"

# 4b. development symlinks (lib<x>.so -> lib<x>.so.N): XLA dlopens some ROCm libraries by their
#     unversioned name (libhipblaslt.so: MI100 check job 36500991, 2026-11-27), which only the
#     -devel RPMs provide
for f in "${ROOT}"/lib/lib*.so.[0-9]*; do
  [ -e "${f}" ] || continue
  base="${f%%.so.*}.so"
  [ -e "${base}" ] || ln -s "$(basename "${f}")" "${base}"
done

# 5. checks: hipcc, and every library the JAX plugin links resolves
export ROCM_PATH="${ROOT}" HIP_PATH="${ROOT}"
XLIB="${ROCM_USER_DIR}/usr/lib64"
[ -d "${XLIB}/atlas" ] && XLIB="${XLIB}:${XLIB}/atlas"
export LD_LIBRARY_PATH="${ROOT}/lib:${XLIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ROOT}/bin/hipcc" --version | head -2
bad=0
for l in libamdhip64.so.6 librocblas.so.4 librocsolver.so.0 libhipsolver.so.0 libhipblas.so.2 libhipsparse.so.1 \
         libhipfft.so.0 libMIOpen.so.1 librccl.so.1; do
  f="${ROOT}/lib/${l}"
  if [ ! -e "${f}" ]; then echo "  MISSING ${l}"; bad=1; continue; fi
  m="$(ldd "${f}" | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
  [ -n "${m}" ] && { echo "  ${l}: not found ${m}"; bad=1; }
done
[ "${bad}" = 0 ] && echo "[rocm_user] all plugin libraries resolve"
echo "ROCM_ROOT=${ROOT}"
echo "ROCM_EXTRA_LIBS=${XLIB}"
exit "${bad}"
