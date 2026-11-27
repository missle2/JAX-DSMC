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
# Alpine NVIDIA: ONE env for the four platforms (same wheels): platforms.sh's JAX_VENV_DEFAULT
# (/projects/$USER/jax-env-313, beside main's jax-env, so main can switch to it too)
if [ -z "${VARIANT:-}" ] && [ -n "${JAX_VENV_SHARED:-}" ]; then
  VENV="${JAX_VENV_SHARED}"
  if [ -x "${VENV}/bin/python3" ] && [ -z "${REBUILD:-}" ]; then
    echo "[build_jax] ${PLATFORM}: shared env ${VENV} exists ($("${VENV}/bin/python3" -c 'import jax; print("jax", jax.__version__)' 2>/dev/null)); REBUILD=1 rebuilds"
    exit 0
  fi
fi
# JAX 0.12.1 everywhere except the ROCm flavour (AMD's wheels for the nodes' ROCm 6.4.4 stop
# at 0.4.35): one JAX version across the NVIDIA platforms and the CPU (2026-11-27 decision)
if [ "${JAX_FLAVOR}" != rocm ]; then JAX_VERSION="${JAX_VERSION:-0.12.1}"; fi
PIN="${JAX_VERSION:+==${JAX_VERSION}}"
echo "[build_jax] ${PLATFORM}: flavor ${JAX_FLAVOR}, jax${PIN:-(latest)} -> ${VENV}"
if [ -n "${CHECK:-}" ]; then exit 0; fi
mkdir -p "$(dirname "${VENV}")"
# the interpreter: PYTHON_BIN, else the module's python3 if new enough (JAX >= 0.12: 3.12+;
# the ROCm 0.4.35 wheels: 3.10/3.12), else a uv-managed 3.13 (get_python.sh)
NEED="3.12"; [ "${JAX_FLAVOR}" = rocm ] && NEED="3.10"
PY="${PYTHON_BIN:-python3}"
if ! "${PY}" -c "import sys; sys.exit(sys.version_info < tuple(map(int, '${NEED}'.split('.'))))" 2> /dev/null; then
  echo "[build_jax] $("${PY}" -V 2>&1) < ${NEED}: fetching a standalone Python 3.13"
  PY="$(bash "${HERE}/get_python.sh" | tail -1)"
fi
echo "[build_jax] interpreter $("${PY}" -V 2>&1): ${PY}"
"${PY}" -m venv "${VENV}"
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
      # jax, jaxlib and the ROCm plugin + pjrt wheels must be the SAME version: JAX ignores a
      # plugin built for another jaxlib (2026-11-25: pip paired jax 0.6.2 with plugin 0.5.0).
      # The jax-rocm60 wheels (ROCm 6.0 builds, gfx908 included) run on Alpine's rocm/6.1.0;
      # 0.5.0 is their last release.
      hipconfig --version 2>/dev/null | head -1 || echo "warning: no hipconfig (rocm module not loaded?)"
      # AMD's wheels built FOR the ROCm release in use (repo.radeon.com/rocm/manylinux/
      # rocm-rel-<ver>; 6.4.4: jax 0.4.35), not PyPI's jax-rocm60 (built for 6.0, linking a
      # comgr/RCCL ABI that 6.4 changed).  ROCM_JAX_WHEELS=pypi restores the PyPI 0.5.0 set.
      if [ "${ROCM_JAX_WHEELS:-amd}" = pypi ]; then
        RJ="${ROCM_JAX_VERSION:-0.5.0}"
        pip install "jax==${RJ}" "jaxlib==${RJ}" "jax-rocm60-plugin==${RJ}" "jax-rocm60-pjrt==${RJ}"
      else
        RJ="${ROCM_JAX_VERSION:-0.4.35}"
        BASE="${ROCM_JAX_BASE:-https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4}"
        PYT="cp$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
        pip install "${BASE}/jax-${RJ}-py3-none-any.whl" \
                    "${BASE}/jaxlib-${RJ}-${PYT}-${PYT}-manylinux_2_28_x86_64.whl" \
                    "${BASE}/jax_rocm60_plugin-${RJ}-${PYT}-${PYT}-manylinux_2_28_x86_64.whl" \
                    "${BASE}/jax_rocm60_pjrt-${RJ}-py3-none-manylinux_2_28_x86_64.whl"
      fi
      # libraries the plugin needs that the loaded modules do not provide (2026-11-25: the
      # rocm/6.1.0 module's libhipsolver wants SuiteSparse, libsuitesparseconfig.so.4).
      # ROCM_EXTRA_LIBS=<dir> is appended to LD_LIBRARY_PATH by the venv's activate script.
      SP="$(python3 -c 'import jax_plugins.xla_rocm60 as m, os; print(os.path.dirname(m.__file__))')"
      MISSING="$(ldd "${SP}/xla_rocm_plugin.so" | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
      if [ -n "${ROCM_EXTRA_LIBS:-}" ]; then
        echo "export LD_LIBRARY_PATH=\"${ROCM_EXTRA_LIBS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"" >> "${VENV}/bin/activate"
        export LD_LIBRARY_PATH="${ROCM_EXTRA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        MISSING="$(ldd "${SP}/xla_rocm_plugin.so" | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
      fi
      if [ -n "${MISSING}" ]; then
        echo "[build_jax] ROCm plugin: libraries NOT FOUND: ${MISSING}"
        echo "            provide them and rebuild with ROCM_EXTRA_LIBS=<dir> (see bench/README.md, MI100)"
      else
        echo "[build_jax] ROCm plugin: every shared library resolves"
      fi
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
