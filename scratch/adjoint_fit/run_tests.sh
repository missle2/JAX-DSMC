#!/bin/bash
# The whole test suite in groups of four files on the CPU slots -> scratch/adjoint_fit/tests.log
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
LOG=scratch/adjoint_fit/tests.log
: > $LOG
mapfile -t FILES < <(ls tests/test_*.py)
for ((i = 0; i < ${#FILES[@]}; i += 4)); do
  group="${FILES[@]:i:4}"
  echo "== $group" >> $LOG
  JAX_PLATFORMS=cpu PYTHONPATH=. $L cpu python3 -m pytest -q -p no:cacheprovider $group 2>&1 \
    | grep -E "passed|failed|error|FAILED|ERROR" >> $LOG
done
echo TESTS_DONE >> $LOG
