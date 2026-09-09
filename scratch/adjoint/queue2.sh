#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
# Match the python invocation exactly.  A loose pattern like `pgrep -f probe_variance`
# also matches this script and any shell whose command line mentions it, so the wait
# never ends -- the same self-match trap that has bitten `pkill` here twice.
while pgrep -f "^python3 -u scratch/adjoint/probe_variance.py" > /dev/null; do sleep 20; done
sleep 15
JAX_PLATFORMS=cuda python3 -u -m examples.shock_adjoint --mode pathwise \
  --save scratch/adjoint/modeA_study.pkl > scratch/adjoint/study2.log 2>&1
echo DONE > scratch/adjoint/study2.done
JAX_PLATFORMS=cuda python3 -u scratch/adjoint/probe_split_variance.py \
  > scratch/adjoint/split_var.log 2>&1
echo DONE > scratch/adjoint/split_var.done
