#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/sparta_jax/verify_audit
mkdir -p $O
PYTHONPATH=. $L cpu python3 -m pytest -q tests/test_sparta_jax.py > $O/tests.log 2>&1
PYTHONPATH=. $L gpu python3 examples/sparta_jax/validate.py --seeds 8 > $O/validate.log 2>&1
echo DONE > $O/done
