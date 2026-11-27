#!/bin/bash
# A standalone CPython 3.13 (uv-managed, no root), for JAX >= 0.12 where the cluster's python
# module is too old (Alpine: python/3.10.2; JAX >= 0.7 needs 3.11+, 0.12 needs 3.12+).
#
#   bash bench/env/get_python.sh             # prints the interpreter path on its last line
#   PY_DIR=/projects/$USER/python bash bench/env/get_python.sh
#
# The interpreter lives under ${PY_DIR} (default /projects/$USER/python on Alpine), outside
# this repository, so main's environments can use it too.  Idempotent.
set -euo pipefail
PY_VER="${PY_VER:-3.13}"
PY_DIR="${PY_DIR:-/projects/${USER}/python}"
mkdir -p "${PY_DIR}"
found="$(ls -d "${PY_DIR}"/cpython-"${PY_VER}"*/bin/python"${PY_VER}" 2>/dev/null | head -1 || true)"
if [ -z "${found}" ]; then
  if [ ! -x "${PY_DIR}/uv-venv/bin/uv" ]; then
    command -v module > /dev/null 2>&1 && module load python 2> /dev/null || true
    python3 -m venv "${PY_DIR}/uv-venv"
    "${PY_DIR}/uv-venv/bin/pip" install -q uv
  fi
  UV_PYTHON_INSTALL_DIR="${PY_DIR}" "${PY_DIR}/uv-venv/bin/uv" python install "${PY_VER}" >&2
  found="$(ls -d "${PY_DIR}"/cpython-"${PY_VER}"*/bin/python"${PY_VER}" | head -1)"
fi
"${found}" -c 'import sys, ssl, sqlite3; print("python", sys.version.split()[0], "ok", file=sys.stderr)'
echo "${found}"
