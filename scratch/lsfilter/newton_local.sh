#!/bin/bash
# The lsfilter group's Newtonian job (study.py "newton") on the laptop GPU, run by run after the
# local CPU comparison finishes (2026-10-26).  Submit the cluster group without it:
#   ONLY='^kn0' GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lsfilter
set -uo pipefail
until grep -q "^done" scratch/lsfilter/local_runs.log; do sleep 30; done
python3 examples/mesh/cluster/study.py --group lsfilter --list | tail -1 | tr ';' '\n' | while read -r cmd; do
  [ -z "${cmd}" ] && continue
  echo "[$(date +%T)] ${cmd%% --filter*}"
  # shellcheck disable=SC2086
  PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu /home/aakash/envs/jax-env/bin/python ${cmd#python3 } 2>&1 \
    | grep -E "^iterate [0-9]*0:|converged|Error|Traceback"
done
echo done
