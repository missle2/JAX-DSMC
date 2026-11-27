#!/bin/bash
# Libraries the Alpine rocm/6.1.0 module needs but the nodes do not have, unpacked without root.
#
#   bash bench/env/rocm_extra_libs.sh      # -> ${BENCH_ROOT}/rocm-extra/usr/lib64
#   PLATFORM=alp-mi100 ROCM_EXTRA_LIBS=<the path it prints> bash bench/env/build_jax.sh
#
# Why (2026-11-27): the module's libhipsolver.so.0 -- which the ROCm JAX plugin links -- needs
# libsuitesparseconfig.so.4 and libcholmod.so.3 (`ldd`), i.e. the OS's SuiteSparse 4.4.6 (and
# its BLAS, ATLAS: libsatlas.so.3)
# (Rocky Linux 8 PowerTools), which the nodes do not install.  `dnf download --resolve` fetches
# it plus whatever of its dependencies is not installed here; rpm2cpio unpacks the RPMs into a
# private tree.  The script ends with the ldd check of libhipsolver against that tree.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="${PLATFORM:-alp-mi100}"
# shellcheck disable=SC1091
source "${HERE}/platforms.sh"
pf_modules
DEST="${ROCM_EXTRA_DIR:-${BENCH_ROOT}/rocm-extra}"
RPMS="${DEST}/rpms"
mkdir -p "${RPMS}"
cat /etc/redhat-release
if ! ls "${RPMS}"/suitesparse-*.rpm > /dev/null 2>&1; then
  # only Rocky's own repositories: the nodes also list vendor repos (Dell system update,
  # AMDGPU, ROCm, curc) whose metadata refresh can take minutes (2026-11-27)
  if timeout 600 dnf download --resolve --disablerepo='*' --enablerepo=baseos,appstream,powertools \
       --destdir "${RPMS}" suitesparse; then
    :
  else
    # no usable dnf: the Rocky 8 PowerTools package straight from the mirror (dependencies are
    # then found by the ldd check below and fetched the same way)
    base="https://dl.rockylinux.org/pub/rocky/8/PowerTools/x86_64/os/Packages/s/"
    f="$(curl -fsSL "${base}" | grep -o 'suitesparse-4[^"]*\.x86_64\.rpm' | sort -u | tail -1)"
    : "${f:?could not find suitesparse in ${base}}"
    curl -fsSL -o "${RPMS}/${f}" "${base}${f}"
  fi
fi
ls -1 "${RPMS}"
for r in "${RPMS}"/*.rpm; do
  ( cd "${DEST}" && rpm2cpio "${r}" | cpio -idm --quiet )
done
# SuiteSparse links ATLAS (libsatlas.so.3), which lives in usr/lib64/atlas (the OS adds that
# directory through /etc/ld.so.conf.d): fetch it if --resolve did not, and put every unpacked
# library directory on the path (2026-11-27)
if [ ! -e "${DEST}/usr/lib64/atlas/libsatlas.so.3" ] && ! ls "${RPMS}"/atlas-*.rpm > /dev/null 2>&1; then
  timeout 600 dnf download --disablerepo='*' --enablerepo=baseos,appstream,powertools --destdir "${RPMS}" atlas \
    && ( cd "${DEST}" && rpm2cpio "$(ls "${RPMS}"/atlas-[0-9]*.rpm | head -1)" | cpio -idm --quiet )
fi
LIB="${DEST}/usr/lib64"
[ -d "${DEST}/usr/lib64/atlas" ] && LIB="${LIB}:${DEST}/usr/lib64/atlas"
echo "library path: ${LIB}"; ls "${DEST}/usr/lib64" | grep -E 'suitesparse|cholmod' || true
HIPSOLVER="${ROCM_PATH:?no ROCM_PATH (rocm module?)}/lib/libhipsolver.so.0"
echo "ldd ${HIPSOLVER} with ${LIB}:"
missing="$(LD_LIBRARY_PATH="${LIB}:${LD_LIBRARY_PATH:-}" ldd "${HIPSOLVER}" | awk '/not found/ {print $1}' | sort -u)"
if [ -n "${missing}" ]; then
  echo "STILL MISSING: ${missing}"
  exit 1
fi
echo "every library resolves.  Next:"
echo "  PLATFORM=alp-mi100 ROCM_EXTRA_LIBS=${LIB} bash bench/env/build_jax.sh"
