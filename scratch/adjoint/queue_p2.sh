#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
# Match the module invocation exactly; a loose pattern also matches this script.
while pgrep -f "^python3 -u -m examples.p2_replication" > /dev/null; do sleep 20; done
sleep 10
# Extend the e-vs-N curve to smaller N, where the random error still dominates, so the
# decrease-then-plateau shape [P2] describes is visible rather than inferred from 3 points.
JAX_PLATFORMS=cuda python3 -u -m examples.p2_replication --part replicate \
  --keys 100 --n-list 500 1000 2000 8000 32000 --npc 500 \
  > scratch/adjoint/p2_repl.log 2>&1
echo DONE > scratch/adjoint/p2_repl.done
