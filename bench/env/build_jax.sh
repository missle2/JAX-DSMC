#!/bin/bash
# Build the JAX environment of one platform (run on a login node; minutes, no GPU needed):
#
#   PLATFORM=pm-gpu ACCOUNT=m1234 bash bench/env/build_jax.sh            # -> $BENCH_ROOT/envs/jax-pm-gpu
#   PLATFORM=alp-h200 JAX_VERSION=0.7.2 bash bench/env/build_jax.sh      # a pinned version
#   PLATFORM=pm-gpu JAX_FLAVOR=cuda12-local VARIANT=local bash bench/env/build_jax.sh
#
# JAX_FLAVOR (platforms.sh sets the default):
#   cuda12 / cuda13       pip wheels with their own CUDA libraries (only the driver is used)
#   cuda12-local          jax[cuda12-local]: the cluster's cudatoolkit/cudnn modules (MODULES)
#   rocm                  ROCm module + the matching jax-rocm plugin wheels; ROCM_JAX_SIF=<image>
#                         uses an Apptainer image instead (see the end of this file)
#   cpu                   plain jax
# The env gets a manifest (pip freeze + module list + nvidia-smi/rocm-smi) next to it, and
# bench/env/check_env.py runs at the end (CPU-only on a login node: import + versions).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/platforms.sh"
pf_modules
NAME="jax-${PLATFORM}${VARIANT:+-${VARIANT}}"
VENV="${JAX_ENV_DIR}/${NAME}"
if [ -z "${VARIANT:-}" ] && [ "${JAX_VENV_DEFAULT}" != "${VENV}" ]; then
  # Alpine NVIDIA: main's existing env is used (platforms.sh); only check it
  echo "[build_jax] ${PLATFORM} uses ${JAX_VENV_DEFAULT} (VARIANT=<name> builds a separate one)"
  # shellcheck disable=SC1091
  source "${JAX_VENV_DEFAULT}/bin/activate"
  JAX_PLATFORMS=cpu python3 "${HERE}/check_env.py" --import-only
  exit 0
fi
PIN="${JAX_VERSION:+==${JAX_VERSION}}"
echo "[build_jax] ${PLATFORM}: flavor ${JAX_FLAVOR}, jax${PIN:-(latest)} -> ${VENV}"
if [ -n "${CHECK:-}" ]; then exit 0; fi
mkdir -p "${JAX_ENV_DIR}"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "need python >= 3.10 for jax (module load python / set MODULES)"; exit 1; }
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip wheel
case "${JAX_FLAVOR}" in
  cuda12|cuda13|cuda12-local|cuda13-local)
    pip install "jax[${JAX_FLAVOR}]${PIN}"
    ;;
  rocm)
    if [ -n "${ROCM_JAX_SIF:-}" ]; then
      echo "ROCm via the Apptainer image ${ROCM_JAX_SIF}: nothing to pip-install (jobs run 'apptainer exec --rocm')"
    else
      # The plugin wheels pin their jaxlib; installing plugin + pjrt + jax in ONE pip call lets
      # the resolver pick a matching jax (installing jax first and the plugin after leaves a
      # newer jax over an older jaxlib).  ROCM_JAX_WHEELS: jax-rocm60-* (built for ROCm 6.0)
      # run on the ROCm 6.x user space of the module (Alpine: rocm/6.1.0; MI100 = gfx908).
      hipconfig --version 2>/dev/null | head -1 || echo "warning: no hipconfig (rocm module not loaded?)"
      W="${ROCM_JAX_WHEELS:-jax-rocm60-plugin jax-rocm60-pjrt}"
      if [ -n "${PIN}" ]; then W="$(for w in ${W}; do printf '%s%s ' "${w}" "${PIN}"; done)"; fi
      # shellcheck disable=SC2086
      pip install ${W} "jax${PIN}"
      python3 - <<'PY'
import importlib.metadata as md
v = {p: md.version(p) for p in ("jax", "jaxlib") }
plug = [d.metadata["Name"] for d in md.distributions() if "rocm" in d.metadata["Name"]]
print("installed:", v, {p: md.version(p) for p in plug})
PY
    fi
    ;;
  cpu)
    pip install "jax${PIN}"
    ;;
  *) echo "unknown JAX_FLAVOR ${JAX_FLAVOR}"; exit 1 ;;
esac
pip install numpy scipy matplotlib pytest
{
  echo "# ${NAME} built $(date -Iseconds) on $(hostname)"
  echo "# PLATFORM=${PLATFORM} JAX_FLAVOR=${JAX_FLAVOR} JAX_VERSION=${JAX_VERSION:-latest}"
  echo "# modules:"; module -t list 2>&1 | sed 's/^/#   /' || true
  pip freeze
} > "${VENV}/MANIFEST.txt"
echo "[build_jax] manifest: ${VENV}/MANIFEST.txt"
# login nodes have no GPU: check the install only (the GPU check runs in a job:
#   sbatch ... bench/jobs/check.sbatch)
JAX_PLATFORMS=cpu python3 "${HERE}/check_env.py" --import-only
