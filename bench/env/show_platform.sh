#!/bin/bash
# Print what platforms.sh resolved for PLATFORM (partition, gres, modules, paths, launch line):
#   PLATFORM=alp-mi100 bash bench/env/show_platform.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/platforms.sh"
for v in PLATFORM SITE ACCOUNT PART GRES GRES_COUNT GPUS_PER_NODE CORES_PER_NODE MODULES KOKKOS_PRESET KOKKOS_ARCH \
         JAX_FLAVOR BENCH_ROOT SPARTA_SRC SPARTA_BUILD_DIR JAX_ENV_DIR RESULTS_DIR JAX_COMPILATION_CACHE_DIR; do
  printf '%-26s %s\n' "${v}" "${!v:-}"
done
echo "sbatch (2 nodes):          $(pf_resources 2)"
[ "${GPUS_PER_NODE}" -gt 0 ] && echo "sbatch (1 GPU):            $(pf_resources_shared 1)"
echo "launch (2 nodes, full):    $(pf_launch 2 "$([ "${GPUS_PER_NODE}" -gt 0 ] && echo "${GPUS_PER_NODE}" || echo 128)")"
