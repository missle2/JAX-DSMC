#!/bin/bash
# Queued behind render_all.sh: holds both CPU slots' locks, so it starts only once the
# render chain has released its slot and never runs beside it.
set -uo pipefail
export JAX_PLATFORMS=cpu PYTHONPATH=.
PY=/home/aakash/envs/jax-env/bin/python
"${PY}" scratch/optax/projection_check.py
"${PY}" -m pytest -q -x tests/test_optimizers.py 2>&1 | tail -5
